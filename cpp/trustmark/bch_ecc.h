#pragma once

#include <string>
#include <vector>
#include <memory>

namespace TrustMark {

// Forward declaration
class BCHCodec;

// BCH Error Correction class
class BCHErrorCorrection {
public:
    // Constructor
    BCHErrorCorrection(int secretLen, bool verbose = true);
    
    // Destructor
    ~BCHErrorCorrection();
    
    // Disable copy
    BCHErrorCorrection(const BCHErrorCorrection&) = delete;
    BCHErrorCorrection& operator=(const BCHErrorCorrection&) = delete;
    
    // Move constructor and assignment
    BCHErrorCorrection(BCHErrorCorrection&&) noexcept;
    BCHErrorCorrection& operator=(BCHErrorCorrection&&) noexcept;
    
    // Main encoding methods
    std::vector<bool> encodeText(const std::vector<std::string>& texts);
    std::vector<bool> encodeBinary(const std::vector<std::string>& binaryStrings);
    
    // Main decoding methods
    std::tuple<std::string, bool, int> decodeBitstream(const std::vector<bool>& bitstream, 
                                                      const std::string& mode = "text");
    
    // Utility methods
    int getSchemaCapacity(int encodingType) const;
    std::vector<bool> encodeTextAscii(const std::string& text);
    std::string decodeTextAscii(const std::vector<bool>& bits);
    std::string decodeBinary(const std::vector<bool>& bits);
    
    // Error handling
    std::string getLastError() const { return lastError_; }
    void clearLastError() { lastError_.clear(); }
    
    // Constants
    static constexpr int BCH_SUPER = 0;
    static constexpr int BCH_3 = 3;
    static constexpr int BCH_4 = 2;
    static constexpr int BCH_5 = 1;

private:
    // Private helper methods
    void setLastError(const std::string& error) const;
    bool initializeBCHCodec();
    std::vector<bool> stringToBits(const std::string& text);
    std::string bitsToString(const std::vector<bool>& bits);
    
    // Member variables
    int secretLen_;
    bool verbose_;
    mutable std::string lastError_;
    
    // BCH codec instance
    std::unique_ptr<BCHCodec> bchCodec_;
    
    // Encoding parameters
    int encodingType_;
    int messageLength_;
    int parityLength_;
    int totalLength_;
};

// BCH Codec implementation (simplified interface)
class BCHCodec {
public:
    // Constructor
    BCHCodec(int messageLength, int parityLength);
    
    // Destructor
    ~BCHCodec();
    
    // Encoding and decoding
    std::vector<bool> encode(const std::vector<bool>& message);
    std::tuple<std::vector<bool>, bool> decode(const std::vector<bool>& received);
    
    // Utility methods
    int getMessageLength() const { return messageLength_; }
    int getParityLength() const { return parityLength_; }
    int getTotalLength() const { return totalLength_; }
    int getErrorCorrectionCapability() const { return errorCorrectionCapability_; }

private:
    // Private helper methods
    std::vector<bool> polynomialDivision(const std::vector<bool>& dividend, 
                                       const std::vector<bool>& divisor);
    std::vector<bool> polynomialMultiplication(const std::vector<bool>& a, 
                                             const std::vector<bool>& b);
    std::vector<bool> polynomialAddition(const std::vector<bool>& a, 
                                       const std::vector<bool>& b);
    
    // Member variables
    int messageLength_;
    int parityLength_;
    int totalLength_;
    int errorCorrectionCapability_;
    
    // Generator polynomial
    std::vector<bool> generatorPolynomial_;
};

// Utility functions for BCH operations
namespace bch_utils {
    // Bit manipulation
    std::vector<bool> reverseBits(const std::vector<bool>& bits);
    std::vector<bool> padBits(const std::vector<bool>& bits, int targetLength);
    std::vector<bool> truncateBits(const std::vector<bool>& bits, int targetLength);
    
    // Polynomial operations
    std::vector<bool> calculateParity(const std::vector<bool>& message, 
                                    const std::vector<bool>& generator);
    bool checkParity(const std::vector<bool>& codeword, 
                    const std::vector<bool>& generator);
    
    // Error detection and correction
    int detectErrors(const std::vector<bool>& received, 
                    const std::vector<bool>& generator);
    std::vector<bool> correctErrors(const std::vector<bool>& received, 
                                  const std::vector<bool>& generator);
}

} // namespace TrustMark
