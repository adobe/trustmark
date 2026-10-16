#include "bch.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    const std::string payload = "1011011110011000111111000000011111011111011100000110110110111";
    const auto packet = trustmark::encode_bits(payload, "BCH_5");
    assert(packet == "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001");
    const std::string versions[] = {"BCH_SUPER", "BCH_5", "BCH_4", "BCH_3"};
    const std::string expected[] = {
        "1011010010100000000000000000000000000000001111000100010111110110000001111110001101101011000010000000",
        "1011010010100000000000000000000000000000000000000000000000000111110111111010101111110110000101010001",
        "1011010010100000000000000000000000000000000000000000000000000000000000011011111010010110010010000010",
        "1011010010100000000000000000000000000000000000000000000000000000000000000000000010100100010110110011",
    };
    for (int schema = 0; schema < 4; ++schema) {
        const auto& version = versions[schema];
        auto bits = trustmark::encode_bits("10110100101", version);
        assert(bits == expected[schema]);
        trustmark::Decoded decoded;
        assert(trustmark::decode_bits(bits, decoded));
        assert(decoded.bits.substr(0, 11) == "10110100101");
        assert(decoded.version == version);
        for (int error_count = 1; error_count <= 3; ++error_count) {
            auto corrupted = bits;
            for (int i = 0; i < error_count; ++i) {
                corrupted[i * 9] = corrupted[i * 9] == '0' ? '1' : '0';
            }
            assert(trustmark::decode_bits(corrupted, decoded));
            assert(decoded.bits.substr(0, 11) == "10110100101");
        }
    }
    auto corrupted = packet;
    corrupted.replace(0, 40, 40, '0');
    trustmark::Decoded decoded;
    assert(!trustmark::decode_bits(corrupted, decoded));
    std::cout << "BCH packet and error correction passed\n";
}
