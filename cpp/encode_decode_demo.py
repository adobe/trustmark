#!/usr/bin/env python3
"""
TrustMark Encode/Decode Demo
Demonstrates encoding a string message, watermarking an image, and recovering the message
"""

import subprocess
import sys

def string_to_bits(s, target_bits=48):
    """Convert string to binary bit string"""
    bits = ''.join(format(ord(c), '08b') for c in s)
    # Pad with zeros
    bits = bits[:target_bits].ljust(target_bits, '0')
    return bits

def bits_to_string(bits):
    """Convert binary bit string back to string"""
    result = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            c = chr(int(byte, 2))
            if 32 <= ord(c) <= 126:  # Printable ASCII
                result += c
            else:
                break  # Stop at non-printable
    return result.rstrip('\x00')  # Remove null padding

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 encode_decode_demo.py <message>")
        print("Example: python3 encode_decode_demo.py 'Adobe!'")
        sys.exit(1)

    message = sys.argv[1][:6]  # Max 6 chars (48 bits)
    print(f"==============================================================================")
    print(f"TrustMark Encode/Decode Demo")
    print(f"==============================================================================\n")
    print(f"Original Message: \"{message}\"")

    # Convert to 48 bits
    message_bits = string_to_bits(message, 48)
    print(f"As 48 bits: {message_bits}")

    # Pad to 100 bits (add zeros for ECC placeholder)
    full_bits = message_bits + ('0' * 52)
    print(f"Padded to 100 bits: {full_bits[:60]}...{full_bits[-20:]}")

    # Run encoding
    print(f"\n--- ENCODING ---")
    print(f"Running: ./trustmark_example ../images/ufo_240.jpg \"{full_bits}\"")

    result = subprocess.run(
        ['./trustmark_example', '../images/ufo_240.jpg', full_bits],
        cwd='/Users/colmurph/workspaces/github/adobe/trustmark/cpp',
        capture_output=True,
        text=True
    )

    # Extract decoded bits from output
    decoded_jpg = None
    decoded_png = None
    for line in result.stdout.split('\n'):
        if 'Decoded JPG (ok=1):' in line:
            decoded_jpg = line.split(': ')[1].strip()
        elif 'Decoded PNG (ok=1):' in line:
            decoded_png = line.split(': ')[1].strip()
        elif 'Watermarked image saved' in line or 'saved as' in line.lower():
            print(f"? {line.strip()}")

    print(f"\n--- DECODING ---")

    if decoded_jpg:
        print(f"\nFrom JPG:")
        print(f"  Raw bits: {decoded_jpg}")
        recovered_msg = bits_to_string(decoded_jpg[:48])
        print(f"  Recovered message (first 48 bits): \"{recovered_msg}\"")
        if recovered_msg.startswith(message[:len(recovered_msg)]):
            print(f"  ??? MATCH! Successfully recovered: \"{recovered_msg}\" ???")
        else:
            print(f"  ? Partial match or degradation")

    if decoded_png:
        print(f"\nFrom PNG:")
        print(f"  Raw bits: {decoded_png}")
        recovered_msg = bits_to_string(decoded_png[:48])
        print(f"  Recovered message (first 48 bits): \"{recovered_msg}\"")
        if recovered_msg.startswith(message[:len(recovered_msg)]):
            print(f"  ??? MATCH! Successfully recovered: \"{recovered_msg}\" ???")
        else:
            print(f"  ? Partial match or degradation")

    print(f"\n==============================================================================")
    print(f"Demo complete!")
    print(f"==============================================================================")

if __name__ == '__main__':
    main()
