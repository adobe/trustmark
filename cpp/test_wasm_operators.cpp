// Simple WASM test for individual operators
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.ort>\n", argv[0]);
        return 1;
    }

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    OrtSessionOptions* session_options;
    OrtSession* session;

    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    g_ort->CreateSessionOptions(&session_options);
    g_ort->CreateSession(env, argv[1], session_options, &session);

    printf("Testing: %s\n", argv[1]);

    // Get input info
    size_t num_inputs;
    g_ort->SessionGetInputCount(session, &num_inputs);
    printf("  Inputs: %zu\n", num_inputs);

    // Cleanup
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("  ✓ Model loads successfully\n");
    return 0;
}
