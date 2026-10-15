#include "bch.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trustmark {
namespace {

struct Schema {
    const char* name;
    int data_bits;
    int errors;
    int id;
};

constexpr std::array<Schema, 4> schemas{{
    {"BCH_SUPER", 40, 8, 0}, {"BCH_5", 61, 5, 1},
    {"BCH_4", 68, 4, 2}, {"BCH_3", 75, 3, 3},
}};

const Schema& schema_for(const std::string& version) {
    for (const auto& schema : schemas) {
        if (version == schema.name) return schema;
    }
    throw std::runtime_error("unknown BCH version: " + version);
}

class Bch {
public:
    explicit Bch(int t) : t_(t) {
        exp_[0] = 1;
        for (int i = 0; i < 127; ++i) {
            if (i) exp_[i] = exp_[i - 1] << 1;
            if (exp_[i] & 128) exp_[i] ^= 137;
            log_[exp_[i]] = i;
        }
        exp_[127] = 1;

        std::array<bool, 127> roots{};
        for (int i = 0; i < t_; ++i) {
            int r = 2 * i + 1;
            for (int j = 0; j < 7; ++j) {
                roots[r] = true;
                r = 2 * r % 127;
            }
        }
        std::vector<int> generator{1};
        for (int i = 0; i < 127; ++i) {
            if (!roots[i]) continue;
            std::vector<int> next(generator.size() + 1);
            for (size_t j = 0; j < generator.size(); ++j) {
                next[j] ^= multiply(generator[j], exp_[i]);
                next[j + 1] ^= generator[j];
            }
            generator = std::move(next);
        }
        degree_ = static_cast<int>(generator.size()) - 1;
        for (int bit : generator) {
            if (bit != 0 && bit != 1) throw std::runtime_error("invalid BCH generator");
        }
        for (int i = degree_; i >= 0; --i) polynomial_.push_back(generator[i]);
    }

    int degree() const { return degree_; }

    std::string parity(const std::string& data) const {
        std::string padded = data;
        padded.resize((data.size() + 7) / 8 * 8, '0');
        std::string division = padded + std::string(degree_, '0');
        for (size_t i = 0; i < padded.size(); ++i) {
            if (division[i] == '1') {
                for (int j = 0; j <= degree_; ++j) {
                    division[i + j] = division[i + j] == ('0' + polynomial_[j]) ? '0' : '1';
                }
            }
        }
        return division.substr(padded.size(), degree_);
    }

    bool correct(std::string& data, std::string& ecc) const {
        std::string padded = data;
        padded.resize((data.size() + 7) / 8 * 8, '0');
        const int data_length = static_cast<int>(padded.size());
        std::string received = padded + ecc;
        std::vector<int> syndrome(2 * t_ + 1);
        for (int j = 1; j <= 2 * t_; ++j) {
            int value = 0;
            for (int k = 0; k < static_cast<int>(received.size()); ++k) {
                if (received[k] == '1') {
                    value ^= exp_[(j * (static_cast<int>(received.size()) - 1 - k)) % 127];
                }
            }
            syndrome[j] = value;
        }
        std::vector<int> locator(2 * t_ + 1), previous(2 * t_ + 1);
        locator[0] = previous[0] = 1;
        int degree = 0, distance = 1, last = 1;
        for (int n = 0; n < 2 * t_; ++n) {
            int delta = syndrome[n + 1];
            for (int i = 1; i <= degree; ++i) {
                delta ^= multiply(locator[i], syndrome[n + 1 - i]);
            }
            if (delta == 0) {
                ++distance;
                continue;
            }
            auto saved = locator;
            const int factor = divide(delta, last);
            for (int i = 0; i + distance <= 2 * t_; ++i) {
                locator[i + distance] ^= multiply(factor, previous[i]);
            }
            if (2 * degree <= n) {
                degree = n + 1 - degree;
                previous = std::move(saved);
                last = delta;
                distance = 1;
            } else {
                ++distance;
            }
        }
        if (degree > t_) return false;
        int found = 0;
        for (int k = 0; k < static_cast<int>(received.size()); ++k) {
            const int power = static_cast<int>(received.size()) - 1 - k;
            int value = 1;
            for (int j = 1; j <= degree; ++j) {
                value ^= multiply(locator[j], exp_[(127 - j * power % 127) % 127]);
            }
            if (value == 0) {
                received[k] = received[k] == '0' ? '1' : '0';
                ++found;
            }
        }
        if (found != degree || found > t_) return false;
        data = received.substr(0, data.size());
        ecc = received.substr(data_length);
        return ecc == parity(data);
    }

private:
    int multiply(int a, int b) const {
        return a && b ? exp_[(log_[a] + log_[b]) % 127] : 0;
    }

    int divide(int a, int b) const {
        return a ? exp_[(log_[a] + 127 - log_[b]) % 127] : 0;
    }

    int t_, degree_ = 0;
    std::array<int, 128> exp_{};
    std::array<int, 128> log_{};
    std::vector<int> polynomial_;
};

}  // namespace

std::string encode_bits(const std::string& payload, const std::string& version) {
    const auto& schema = schema_for(version);
    if (payload.size() > static_cast<size_t>(schema.data_bits) ||
        payload.find_first_not_of("01") != std::string::npos) {
        throw std::runtime_error("payload must be a binary string of at most " +
                                 std::to_string(schema.data_bits) + " bits");
    }
    std::string data = payload;
    data.resize(schema.data_bits, '0');
    Bch bch(schema.errors);
    std::string ecc = bch.parity(data);
    ecc.resize(96 - schema.data_bits);
    std::string result = data + ecc + "00";
    result += schema.id & 2 ? '1' : '0';
    result += schema.id & 1 ? '1' : '0';
    return result;
}

bool decode_bits(const std::string& packet, Decoded& decoded) {
    if (packet.size() != 100 || packet.find_first_not_of("01") != std::string::npos) {
        throw std::runtime_error("decoder output must contain 100 binary digits");
    }
    const int hinted = (packet[98] - '0') * 2 + packet[99] - '0';
    std::array<int, 4> order{hinted, 0, 1, 2};
    int index = 1;
    for (int i = 0; i < 4; ++i) {
        if (i != hinted) order[index++] = i;
    }
    for (int id : order) {
        const auto& schema = schemas[id];
        std::string data = packet.substr(0, schema.data_bits);
        std::string ecc = packet.substr(schema.data_bits, 96 - schema.data_bits);
        Bch bch(schema.errors);
        if (bch.correct(data, ecc)) {
            decoded = {data, schema.name};
            return true;
        }
    }
    return false;
}

}  // namespace trustmark
