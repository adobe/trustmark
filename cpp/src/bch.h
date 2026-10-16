#pragma once

#include <array>
#include <string>

namespace trustmark {

struct Decoded {
    std::string bits;
    std::string version;
};

std::string encode_bits(const std::string& payload, const std::string& version);
// Returns false when no supported BCH schema validates the packet.
bool decode_bits(const std::string& packet, Decoded& decoded);

}  // namespace trustmark
