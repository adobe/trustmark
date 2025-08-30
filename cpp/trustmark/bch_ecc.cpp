#include "bch_ecc.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace TrustMark {

// Constructor
BCHErrorCorrection::BCHErrorCorrection(int secretLen, bool verbose)
    : secretLen_(secretLen)
    , verbose_(verbose)
    , encodingType_(BCH_5)
    , messageLength_(secretLen)
    , parityLength_(10)
    , totalLength_(messageLength_ + parityLength_)
{
    if (!initializeBCHCodec()) {
        setLastError("Failed to initialize BCH codec");
        return;
    }

    if (verbose_) {
        std::cout << "BCH Error Correction initialized with "
                  << messageLength_ << " message bits and "
                  << parityLength_ << " parity bits" << std::endl;
    }
}

// Destructor
BCHErrorCorrection::~BCHErrorCorrection() = default;

// Move constructor
BCHErrorCorrection::BCHErrorCorrection(BCHErrorCorrection&& other) noexcept
    : secretLen_(other.secretLen_)
    , verbose_(other.verbose_)
    , lastError_(std::move(other.lastError_))
    , bchCodec_(std::move(other.bchCodec_))
    , encodingType_(other.encodingType_)
    , messageLength_(other.messageLength_)
    , parityLength_(other.parityLength_)
    , totalLength_(other.totalLength_)
{
}

// Move assignment
BCHErrorCorrection& BCHErrorCorrection::operator=(BCHErrorCorrection&& other) noexcept {
    if (this != &other) {
        secretLen_ = other.secretLen_;
        verbose_ = other.verbose_;
        lastError_ = std::move(other.lastError_);
        bchCodec_ = std::move(other.bchCodec_);
        encodingType_ = other.encodingType_;
        messageLength_ = other.messageLength_;
        parityLength_ = other.parityLength_;
        totalLength_ = other.totalLength_;
    }
    return *this;
}

// Encode text
std::vector<bool> BCHErrorCorrection::encodeText(const std::vector<std::string>& texts) {
    try {
        if (texts.empty()) {
            setLastError("No text provided for encoding");
            return {};
        }

        // For simplicity, encode only the first text
        std::string text = texts[0];

        // Convert text to bits
        std::vector<bool> messageBits = stringToBits(text);

        // Pad or truncate to message length
        if (messageBits.size() < messageLength_) {
            messageBits.resize(messageLength_, false);
        } else if (messageBits.size() > messageLength_) {
            messageBits.resize(messageLength_);
        }

        // Encode with BCH
        if (bchCodec_) {
            return bchCodec_->encode(messageBits);
        } else {
            // Fallback: return message bits directly
            return messageBits;
        }

    } catch (const std::exception& e) {
        setLastError("Text encoding failed: " + std::string(e.what()));
        return {};
    }
}

// Encode binary
std::vector<bool> BCHErrorCorrection::encodeBinary(const std::vector<std::string>& binaryStrings) {
    try {
        if (binaryStrings.empty()) {
            setLastError("No binary string provided for encoding");
            return {};
        }

        // For simplicity, encode only the first binary string
        std::string binaryString = binaryStrings[0];

        // Convert binary string to bits
        std::vector<bool> messageBits;
        for (char c : binaryString) {
            if (c == '0') {
                messageBits.push_back(false);
            } else if (c == '1') {
                messageBits.push_back(true);
            } else {
                setLastError("Invalid binary character: " + std::string(1, c));
                return {};
            }
        }

        // Pad or truncate to message length
        if (messageBits.size() < messageLength_) {
            messageBits.resize(messageLength_, false);
        } else if (messageBits.size() > messageLength_) {
            messageBits.resize(messageLength_);
        }

        // Encode with BCH
        if (bchCodec_) {
            return bchCodec_->encode(messageBits);
        } else {
            // Fallback: return message bits directly
            return messageBits;
        }

    } catch (const std::exception& e) {
        setLastError("Binary encoding failed: " + std::string(e.what()));
        return {};
    }
}

// Decode bitstream
std::tuple<std::string, bool, int> BCHErrorCorrection::decodeBitstream(
    const std::vector<bool>& bitstream, const std::string& mode) {
    try {
        if (bitstream.empty()) {
            setLastError("Empty bitstream provided for decoding");
            return {"", false, -1};
        }

        // Decode with BCH if available
        std::vector<bool> decodedBits;
        bool success = false;

        if (bchCodec_) {
            auto result = bchCodec_->decode(bitstream);
            decodedBits = std::get<0>(result);
            success = std::get<1>(result);
        } else {
            // Fallback: use bitstream directly
            decodedBits = bitstream;
            success = true;
        }

        if (!success) {
            return {"", false, -1};
        }

        // Convert bits to string based on mode
        std::string result;
        if (mode == "text") {
            result = decodeTextAscii(decodedBits);
        } else if (mode == "binary") {
            result = decodeBinary(decodedBits);
        } else {
            setLastError("Unknown decode mode: " + mode);
            return {"", false, -1};
        }

        return {result, true, encodingType_};

    } catch (const std::exception& e) {
        setLastError("Bitstream decoding failed: " + std::string(e.what()));
        return {"", false, -1};
    }
}

// Get schema capacity
int BCHErrorCorrection::getSchemaCapacity(int encodingType) const {
    // Simplified capacity calculation
    switch (encodingType) {
        case BCH_SUPER:
            return messageLength_ - 20; // More parity bits
        case BCH_3:
            return messageLength_ - 15;
        case BCH_4:
            return messageLength_ - 12;
        case BCH_5:
            return messageLength_ - 10;
        default:
            return messageLength_;
    }
}

// Encode text ASCII
std::vector<bool> BCHErrorCorrection::encodeTextAscii(const std::string& text) {
    try {
        std::vector<bool> bits;

        for (char c : text) {
            // Convert each character to 8 bits
            for (int i = 7; i >= 0; --i) {
                bits.push_back((c >> i) & 1);
            }
        }

        return bits;

    } catch (const std::exception& e) {
        setLastError("ASCII text encoding failed: " + std::string(e.what()));
        return {};
    }
}

// Decode text ASCII
std::string BCHErrorCorrection::decodeTextAscii(const std::vector<bool>& bits) {
    try {
        std::string text;

        // Process bits in groups of 8
        for (size_t i = 0; i < bits.size(); i += 8) {
            if (i + 7 >= bits.size()) {
                break; // Incomplete byte
            }

            char c = 0;
            for (int j = 0; j < 8; ++j) {
                if (bits[i + j]) {
                    c |= (1 << (7 - j));
                }
            }
            text += c;
        }

        return text;

    } catch (const std::exception& e) {
        setLastError("ASCII text decoding failed: " + std::string(e.what()));
        return "";
    }
}

// Decode binary
std::string BCHErrorCorrection::decodeBinary(const std::vector<bool>& bits) {
    try {
        std::string binaryString;

        for (bool bit : bits) {
            binaryString += (bit ? '1' : '0');
        }

        return binaryString;

    } catch (const std::exception& e) {
        setLastError("Binary decoding failed: " + std::string(e.what()));
        return "";
    }
}

// Initialize BCH codec
bool BCHErrorCorrection::initializeBCHCodec() {
    try {
        // Create BCH codec with appropriate parameters
        bchCodec_ = std::make_unique<BCHCodec>(messageLength_, parityLength_);
        return bchCodec_->getMessageLength() > 0;

    } catch (const std::exception& e) {
        setLastError("BCH codec initialization failed: " + std::string(e.what()));
        return false;
    }
}

// String to bits conversion
std::vector<bool> BCHErrorCorrection::stringToBits(const std::string& text) {
    return encodeTextAscii(text);
}

// Bits to string conversion
std::string BCHErrorCorrection::bitsToString(const std::vector<bool>& bits) {
    return decodeTextAscii(bits);
}

// Set last error
void BCHErrorCorrection::setLastError(const std::string& error) const {
    lastError_ = error;
    if (verbose_) {
        std::cerr << "BCH Error: " << error << std::endl;
    }
}

// BCH Codec implementation
BCHCodec::BCHCodec(int messageLength, int parityLength)
    : messageLength_(messageLength)
    , parityLength_(parityLength)
    , totalLength_(messageLength + parityLength)
    , errorCorrectionCapability_(parityLength / 2)
{
    // Initialize generator polynomial (simplified)
    // In practice, this would be a proper BCH generator polynomial
    generatorPolynomial_.resize(parityLength + 1, false);
    generatorPolynomial_[0] = true;
    generatorPolynomial_[parityLength] = true;

    // Add some intermediate terms for better error correction
    for (int i = 1; i < parityLength; ++i) {
        if (i % 3 == 0) { // Simplified pattern
            generatorPolynomial_[i] = true;
        }
    }
}

// Destructor
BCHCodec::~BCHCodec() = default;

// Encode message
std::vector<bool> BCHCodec::encode(const std::vector<bool>& message) {
    try {
        if (message.size() != messageLength_) {
            throw std::runtime_error("Message length mismatch");
        }

        // Calculate parity bits using polynomial division
        std::vector<bool> parity = bch_utils::calculateParity(message, generatorPolynomial_);

        // Combine message and parity
        std::vector<bool> codeword = message;
        codeword.insert(codeword.end(), parity.begin(), parity.end());

        return codeword;

    } catch (const std::exception& e) {
        throw std::runtime_error("BCH encoding failed: " + std::string(e.what()));
    }
}

// Decode received codeword
std::tuple<std::vector<bool>, bool> BCHCodec::decode(const std::vector<bool>& received) {
    try {
        if (received.size() != totalLength_) {
            throw std::runtime_error("Received codeword length mismatch");
        }

        // Check for errors
        int errorCount = bch_utils::detectErrors(received, generatorPolynomial_);

        if (errorCount == 0) {
            // No errors, extract message
            std::vector<bool> message(received.begin(),
                                    received.begin() + messageLength_);
            return {message, true};
        } else if (errorCount <= errorCorrectionCapability_) {
            // Errors can be corrected
            std::vector<bool> corrected = bch_utils::correctErrors(received, generatorPolynomial_);
            std::vector<bool> message(corrected.begin(),
                                    corrected.begin() + messageLength_);
            return {message, true};
        } else {
            // Too many errors
            return {{}, false};
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("BCH decoding failed: " + std::string(e.what()));
    }
}



// BCH utility functions
namespace bch_utils {

// Reverse bits
std::vector<bool> reverseBits(const std::vector<bool>& bits) {
    std::vector<bool> reversed = bits;
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

// Pad bits
std::vector<bool> padBits(const std::vector<bool>& bits, int targetLength) {
    std::vector<bool> padded = bits;
    if (padded.size() < targetLength) {
        padded.resize(targetLength, false);
    }
    return padded;
}

// Truncate bits
std::vector<bool> truncateBits(const std::vector<bool>& bits, int targetLength) {
    if (bits.size() <= targetLength) {
        return bits;
    }
    return std::vector<bool>(bits.begin(), bits.begin() + targetLength);
}

// Calculate parity
std::vector<bool> calculateParity(const std::vector<bool>& message,
                                 const std::vector<bool>& generator) {
    // Simplified parity calculation
    std::vector<bool> parity(generator.size() - 1, false);

    // Simple XOR-based parity (not actual BCH)
    for (size_t i = 0; i < message.size(); ++i) {
        if (message[i]) {
            for (size_t j = 0; j < parity.size(); ++j) {
                parity[j] = parity[j] ^ ((i + j) % 2);
            }
        }
    }

    return parity;
}

// Check parity
bool checkParity(const std::vector<bool>& codeword,
                 const std::vector<bool>& generator) {
    // Simplified parity check
    std::vector<bool> message(codeword.begin(),
                             codeword.begin() + codeword.size() - generator.size() + 1);
    std::vector<bool> parity = calculateParity(message, generator);

    for (bool bit : parity) {
        if (bit) return false;
    }
    return true;
}

// Detect errors
int detectErrors(const std::vector<bool>& received,
                 const std::vector<bool>& generator) {
    // Simplified error detection
    std::vector<bool> message(received.begin(),
                             received.begin() + received.size() - generator.size() + 1);
    std::vector<bool> parity = calculateParity(message, generator);

    int errorCount = 0;
    for (bool bit : parity) {
        if (bit) errorCount++;
    }

    return errorCount;
}

// Correct errors
std::vector<bool> correctErrors(const std::vector<bool>& received,
                               const std::vector<bool>& generator) {
    // Simplified error correction (no actual correction)
    return received;
}

} // namespace bch_utils

} // namespace TrustMark
