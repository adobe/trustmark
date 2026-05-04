// TrustMark HTTP component entry point
// Exports wasi:http/incoming-handler; imports wasi:webgpu (via ORT WebGPU EP).
//
// POST /encode[?bits=<100-bit-string>]
//   Body:  image/png or image/jpeg
//   Response: 200 image/png (watermarked)
//
// Configuration is read from wasi:config/store (set via host_interfaces.config
// in .wash/config.yaml). Keys: MODEL_PATH, USE_WEBGPU.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include "image_utils.h"

extern "C" {
#include "wasi_http/trustmark_http.h"
#include "wasi_config/trustmark_config.h"
}

// ── wasi:config/store helper ──────────────────────────────────────────────────

// Read a key from wasi:config/store; return empty string if absent or error.
static std::string config_get(const char* key) {
    trustmark_config_string_t k;
    trustmark_config_string_set(&k, key);
    wasi_config_store_result_option_string_error_t result = {};
    wasi_config_store_error_t err = {};
    trustmark_config_option_string_t opt = {};
    if (wasi_config_store_get(&k, &opt, &err) && opt.is_some) {
        std::string val(reinterpret_cast<char*>(opt.val.ptr), opt.val.len);
        trustmark_config_option_string_free(&opt);
        return val;
    }
    return {};
}

// ── Constants ────────────────────────────────────────────────────────────────

static const float WM_STRENGTH    = 0.95f * 1.25f; // 1.1875
static const float RESIDUAL_CLAMP = 0.2f;

// Default 100-bit BCH-encoded watermark (verified round-trip with Rust CLI).
static const char DEFAULT_BITS[] =
    "1011011110011000111111000000011111011111011100000110110110111"
    "000110010101101111010011011000010000001";

// ── Lazy ORT session ──────────────────────────────────────────────────────────

struct Session {
    Ort::Env     env;
    Ort::Session session;

    Session(const std::string& model_path, bool use_webgpu)
        : env(ORT_LOGGING_LEVEL_WARNING, "TrustMarkHTTP"),
          session(nullptr)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (use_webgpu) {
            std::unordered_map<std::string, std::string> webgpu_opts;
            webgpu_opts["preferredLayout"] = "NCHW";
            opts.AppendExecutionProvider("WebGPU", webgpu_opts);
        }
        session = Ort::Session(env, model_path.c_str(), opts);
    }
};

static std::unique_ptr<Session> g_session;

static Session& get_session() {
    if (!g_session) {
        std::string model_path = config_get("MODEL_PATH");
        if (model_path.empty())
            model_path = "/models/encoder_P.with_runtime_opt.ort";
        bool use_webgpu = !config_get("USE_WEBGPU").empty();
        g_session = std::make_unique<Session>(model_path, use_webgpu);
    }
    return *g_session;
}

// ── Tensor helpers (matches simple.cpp) ──────────────────────────────────────

static std::vector<float> imageToTensor(const ImageUtils::Image& img) {
    int H = img.height, W = img.width, C = img.channels;
    std::vector<float> tensor(C * H * W);
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++) {
                float v = img.data[(h * W + w) * C + c] / 255.0f;
                tensor[c * H * W + h * W + w] = v * 2.0f - 1.0f;
            }
    return tensor;
}

static ImageUtils::Image tensorToImage(const float* data, int H, int W, int C) {
    ImageUtils::Image img(W, H, C);
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++) {
                float v = (data[c * H * W + h * W + w] + 1.0f) * 0.5f * 255.0f;
                img.data[(h * W + w) * C + c] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, v)));
            }
    return img;
}

// ── Stream I/O helpers ────────────────────────────────────────────────────────

// Read all bytes from a wasi:io input-stream (blocking reads).
// Drops own_stream when done.
static std::vector<uint8_t> read_stream(wasi_io_streams_own_input_stream_t own_stream) {
    std::vector<uint8_t> buf;
    // Read up to 1MB per call; loop until stream closes.
    // blocking-read returns ok=false with stream-error::closed when done.
    constexpr uint64_t CHUNK = 1024 * 1024; // 1MB

    while (true) {
        trustmark_http_list_u8_t chunk = {};
        wasi_io_streams_stream_error_t err = {};
        bool ok = wasi_io_streams_method_input_stream_blocking_read(
            wasi_io_streams_borrow_input_stream(own_stream), CHUNK, &chunk, &err);
        if (chunk.ptr && chunk.len > 0) {
            buf.insert(buf.end(), chunk.ptr, chunk.ptr + chunk.len);
        }
        if (chunk.ptr) free(chunk.ptr);
        if (!ok) {
            // stream-error::closed (or last-operation-failed) — done
            break;
        }
    }
    wasi_io_streams_input_stream_drop_own(own_stream);
    return buf;
}

// Write bytes to a wasi:io output-stream.
// Respects check-write() capacity — required by wasi:io spec (trap if violated).
// Uses subscribe+poll to block until capacity is available, then writes in chunks.
// Does NOT drop own_stream — caller is responsible.
//
// IMPORTANT: call response_outparam_set BEFORE calling write_stream so the host
// (hyper) starts consuming the body mpsc channel.  If the outparam has not been
// set yet the channel fills up and write_stream deadlocks waiting for capacity.
static bool write_stream(wasi_io_streams_borrow_output_stream_t stream,
                         const uint8_t* data, size_t len) {
    size_t written = 0;
    while (written < len) {
        uint64_t capacity = 0;
        wasi_io_streams_stream_error_t cap_err = {};
        bool cap_ok = wasi_io_streams_method_output_stream_check_write(
            stream, &capacity, &cap_err);
        if (!cap_ok) return false;

        if (capacity == 0) {
            // Block until the stream has capacity.
            wasi_io_streams_own_pollable_t pollable =
                wasi_io_streams_method_output_stream_subscribe(stream);
            wasi_io_poll_method_pollable_block(
                wasi_io_poll_borrow_pollable(pollable));
            wasi_io_poll_pollable_drop_own(pollable);
            continue;
        }

        size_t chunk = std::min<size_t>(len - written, static_cast<size_t>(capacity));
        chunk = std::min<size_t>(chunk, 4096u); // 4KB max per non-blocking write

        trustmark_http_list_u8_t payload;
        payload.ptr = const_cast<uint8_t*>(data + written);
        payload.len = chunk;
        wasi_io_streams_stream_error_t err = {};
        bool ok = wasi_io_streams_method_output_stream_write(stream, &payload, &err);
        if (!ok) return false;
        written += chunk;
    }
    wasi_io_streams_stream_error_t ferr = {};
    wasi_io_streams_method_output_stream_blocking_flush(stream, &ferr);
    return true;
}

// ── Query-param helper ────────────────────────────────────────────────────────

// Extract value of a named query parameter from a path+query string like
// "/encode?bits=101010...".
static std::string query_param(const std::string& path_and_query, const char* key) {
    auto q = path_and_query.find('?');
    if (q == std::string::npos) return {};
    std::string query = path_and_query.substr(q + 1);
    std::string search = std::string(key) + "=";
    auto pos = query.find(search);
    if (pos == std::string::npos) return {};
    auto start = pos + search.size();
    auto end = query.find('&', start);
    return query.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

// ── HTTP error helper ─────────────────────────────────────────────────────────

// Helper: make a field name/value from a C string literal.
static inline wasi_http_types_field_name_t make_field_name(const char* s) {
    return { const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(s)),
             static_cast<size_t>(strlen(s)) };
}
static inline wasi_http_types_field_value_t make_field_value(const char* s) {
    return { const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(s)),
             static_cast<size_t>(strlen(s)) };
}

static void send_error(wasi_http_types_own_response_outparam_t outparam,
                       uint16_t status, const char* msg) {
    wasi_http_types_own_fields_t hdrs = wasi_http_types_constructor_fields();
    {
        wasi_http_types_field_name_t  name  = make_field_name("content-type");
        wasi_http_types_field_value_t value = make_field_value("text/plain");
        wasi_http_types_header_error_t herr = {};
        wasi_http_types_method_fields_append(
            wasi_http_types_borrow_fields(hdrs), &name, &value, &herr);
    }
    wasi_http_types_own_outgoing_response_t resp =
        wasi_http_types_constructor_outgoing_response(hdrs);
    wasi_http_types_method_outgoing_response_set_status_code(
        wasi_http_types_borrow_outgoing_response(resp), status);

    wasi_http_types_own_outgoing_body_t body = {};
    wasi_http_types_method_outgoing_response_body(
        wasi_http_types_borrow_outgoing_response(resp), &body);

    wasi_http_types_own_output_stream_t out_stream = {};
    wasi_http_types_method_outgoing_body_write(
        wasi_http_types_borrow_outgoing_body(body), &out_stream);

    // Set outparam first so hyper starts consuming the body channel.
    wasi_http_types_result_own_outgoing_response_error_code_t result;
    result.is_err = false;
    result.val.ok = resp;
    wasi_http_types_static_response_outparam_set(outparam, &result);

    write_stream(wasi_io_streams_borrow_output_stream(out_stream),
                 reinterpret_cast<const uint8_t*>(msg), strlen(msg));

    wasi_io_streams_output_stream_drop_own(out_stream);

    wasi_http_types_error_code_t body_err = {};
    wasi_http_types_static_outgoing_body_finish(body, nullptr, &body_err);
}

// ── Exported handle function ──────────────────────────────────────────────────

extern "C"
void exports_wasi_http_incoming_handler_handle(
    exports_wasi_http_incoming_handler_own_incoming_request_t request,
    exports_wasi_http_incoming_handler_own_response_outparam_t outparam)
{
    // ── Extract path+query ────────────────────────────────────────────────────
    std::string path_and_query;
    {
        trustmark_http_string_t pq_str = {};
        if (wasi_http_types_method_incoming_request_path_with_query(
                wasi_http_types_borrow_incoming_request(request), &pq_str)) {
            path_and_query.assign(reinterpret_cast<char*>(pq_str.ptr), pq_str.len);
            free(pq_str.ptr);
        }
    }

    // ── Extract ?bits= param ─────────────────────────────────────────────────
    std::string bits_str = query_param(path_and_query, "bits");
    const char* bits_cstr = bits_str.empty() ? DEFAULT_BITS : bits_str.c_str();
    if (strlen(bits_cstr) != 100) {
        send_error(outparam, 400, "bits parameter must be exactly 100 chars");
        return;
    }

    // ── Read request body ────────────────────────────────────────────────────
    wasi_http_types_own_incoming_body_t in_body = {};
    if (!wasi_http_types_method_incoming_request_consume(
            wasi_http_types_borrow_incoming_request(request), &in_body)) {
        send_error(outparam, 400, "failed to consume request body");
        return;
    }
    wasi_http_types_own_input_stream_t in_stream = {};
    if (!wasi_http_types_method_incoming_body_stream(
            wasi_http_types_borrow_incoming_body(in_body), &in_stream)) {
        send_error(outparam, 400, "failed to get body stream");
        return;
    }

    std::vector<uint8_t> img_bytes = read_stream(in_stream);
    wasi_http_types_incoming_body_drop_own(in_body);

    if (img_bytes.empty()) {
        send_error(outparam, 400, "empty request body");
        return;
    }

    // ── Decode image ──────────────────────────────────────────────────────────
    ImageUtils::Image img = ImageUtils::loadImageFromMemory(
        img_bytes.data(), img_bytes.size());
    if (img.empty()) {
        send_error(outparam, 400, "failed to decode image");
        return;
    }

    // Convert RGBA → RGB if needed
    if (img.channels == 4) {
        ImageUtils::Image rgb(img.width, img.height, 3);
        for (int i = 0; i < img.width * img.height; i++) {
            rgb.data[i * 3 + 0] = img.data[i * 4 + 0];
            rgb.data[i * 3 + 1] = img.data[i * 4 + 1];
            rgb.data[i * 3 + 2] = img.data[i * 4 + 2];
        }
        img = rgb;
    }

    // ── Run TrustMark encoder ─────────────────────────────────────────────────
    // Note: built with -fno-exceptions; ORT errors call std::terminate.
    {
        Session& sess = get_session();
        Ort::AllocatorWithDefaultOptions allocator;

        // Resize to 256×256
        ImageUtils::Image cover = ImageUtils::resizeImage(img, 256, 256);
        std::vector<float> image_data = imageToTensor(cover);

        // Build secret from 100-bit BCH-encoded string
        std::vector<float> secret_data(100);
        for (int i = 0; i < 100; i++)
            secret_data[i] = (bits_cstr[i] == '1') ? 1.0f : 0.0f;

        std::vector<int64_t> image_shape  = {1, 3, 256, 256};
        std::vector<int64_t> secret_shape = {1, 100};

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
            mem, image_data.data(), image_data.size(),
            image_shape.data(), image_shape.size());
        Ort::Value sec_tensor = Ort::Value::CreateTensor<float>(
            mem, secret_data.data(), secret_data.size(),
            secret_shape.data(), secret_shape.size());

        auto in0  = sess.session.GetInputNameAllocated(0, allocator);
        auto in1  = sess.session.GetInputNameAllocated(1, allocator);
        auto out0 = sess.session.GetOutputNameAllocated(0, allocator);
        const char* input_names[]  = {in0.get(), in1.get()};
        const char* output_names[] = {out0.get()};

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(img_tensor));
        inputs.push_back(std::move(sec_tensor));

        auto outputs = sess.session.Run(Ort::RunOptions{nullptr},
                                        input_names, inputs.data(), 2,
                                        output_names, 1);

        const float* stego = outputs[0].GetTensorMutableData<float>();
        const int N = 3 * 256 * 256;
        std::vector<float> final_chw(N);
        for (int i = 0; i < N; i++) {
            float s = std::max(-1.0f, std::min(1.0f, stego[i]));
            float residual = (s - image_data[i]) * WM_STRENGTH;
            residual = std::max(-RESIDUAL_CLAMP, std::min(RESIDUAL_CLAMP, residual));
            float blended = std::max(-1.0f, std::min(1.0f, image_data[i] + residual));
            final_chw[i] = blended;
        }

        ImageUtils::Image output_img = tensorToImage(final_chw.data(), 256, 256, 3);

        // ── Encode output as PNG ──────────────────────────────────────────────
        std::vector<uint8_t> png_bytes;
        if (!ImageUtils::savePNGToMemory(output_img, png_bytes)) {
            send_error(outparam, 500, "failed to encode output PNG");
            return;
        }

        // ── Build response ────────────────────────────────────────────────────
        wasi_http_types_own_fields_t resp_hdrs = wasi_http_types_constructor_fields();
        {
            wasi_http_types_field_name_t  name  = make_field_name("content-type");
            wasi_http_types_field_value_t value = make_field_value("image/png");
            wasi_http_types_header_error_t herr = {};
            wasi_http_types_method_fields_append(
                wasi_http_types_borrow_fields(resp_hdrs), &name, &value, &herr);
        }

        wasi_http_types_own_outgoing_response_t resp =
            wasi_http_types_constructor_outgoing_response(resp_hdrs);
        wasi_http_types_method_outgoing_response_set_status_code(
            wasi_http_types_borrow_outgoing_response(resp), 200);

        // Get body and write stream before consuming resp via set.
        wasi_http_types_own_outgoing_body_t resp_body = {};
        wasi_http_types_method_outgoing_response_body(
            wasi_http_types_borrow_outgoing_response(resp), &resp_body);

        wasi_http_types_own_output_stream_t out_stream = {};
        wasi_http_types_method_outgoing_body_write(
            wasi_http_types_borrow_outgoing_body(resp_body), &out_stream);

        // Set outparam FIRST so the host (hyper) starts consuming the body
        // channel.  Without this, write_stream deadlocks when the small mpsc
        // buffer fills up — hyper only starts reading after the outparam is set.
        wasi_http_types_result_own_outgoing_response_error_code_t result;
        result.is_err = false;
        result.val.ok = resp;  // resp consumed here
        wasi_http_types_static_response_outparam_set(outparam, &result);

        // Now write body data; hyper is already reading from the other end.
        write_stream(wasi_io_streams_borrow_output_stream(out_stream),
                     png_bytes.data(), png_bytes.size());

        // Drop stream before finishing body (child must be released first).
        wasi_io_streams_output_stream_drop_own(out_stream);

        wasi_http_types_error_code_t body_err = {};
        wasi_http_types_static_outgoing_body_finish(resp_body, nullptr, &body_err);
    }
}
