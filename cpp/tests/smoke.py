"""Exercise the C++ runner against real encoder and decoder models."""

import math
import pathlib
import struct
import subprocess
import sys
import tempfile
def invoke(runner, *args):
    return subprocess.run([runner, *map(str, args)], capture_output=True, text=True, check=True)


def main(runner, models):
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        payload = "1011011110011000111111000000011111011111011100000110110110111"
        source = pathlib.Path(__file__).resolve().parents[2] / "images" / "ghost.png"
        for variant in "BCPQ":
            size = 224 if variant == "P" else 256
            image = directory / "image.f32"
            image.write_bytes(struct.pack("=f", 0.0) * (3 * size * size))
            decoded = invoke(runner, "decode", models / f"decoder_{variant}.onnx", image).stdout.strip()
            assert len(decoded) == 100 and set(decoded) <= {"0", "1"}, variant

            short_input = directory / "short.f32"
            short_input.write_bytes(b"\x00\x00\x00\x00")
            failure = subprocess.run(
                [runner, "decode", str(models / f"decoder_{variant}.onnx"), str(short_input)],
                capture_output=True, text=True,
            )
            assert failure.returncode != 0 and "exactly" in failure.stderr, variant

            image.write_bytes(struct.pack("=f", 0.0) * (3 * 256 * 256))
            encoded = directory / "encoded.f32"
            invoke(runner, "encode", models / f"encoder_{variant}.onnx", image, "0" * 100, encoded)
            data = encoded.read_bytes()
            assert len(data) == 3 * 256 * 256 * 4, variant
            assert all(math.isfinite(value) for (value,) in struct.iter_unpack("=f", data)), variant

            failure = subprocess.run(
                [runner, "encode", str(models / f"encoder_{variant}.onnx"),
                 str(image), "invalid-bits", str(encoded)],
                capture_output=True, text=True,
            )
            assert failure.returncode != 0 and "BITS100" in failure.stderr, variant

            watermarked = directory / f"{variant}.png"
            invoke(runner, "encode-image", models, variant, source, watermarked, payload, "BCH_5")
            result = invoke(runner, "decode-image", models, variant, watermarked).stdout.strip()
            assert result == f"Found watermark (BCH_5): {payload}", (variant, result)
            print(f"{variant}: tensor inference and image roundtrip passed")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
