"""Verify that Rust and C++ decode each other's watermarked PNGs."""

import pathlib
import subprocess
import sys
import tempfile


PAYLOAD = "1011011110011000111111000000011111011111011100000110110110111"


def run(*command):
    result = subprocess.run(
        list(map(str, command)), capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def make_wide_image(path):
    width, height = 640, 256
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend(((x // 4 + y // 8) % 256,
                           (x // 6 + y // 3) % 256,
                           (x // 5 + y // 5) % 256))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def main(cpp_runner, rust_cli, models):
    root = pathlib.Path(__file__).resolve().parents[2]
    cpp_runner = cpp_runner.resolve()
    rust_cli = rust_cli.resolve()
    models = models.resolve()

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory)
        wide_image = output / "wide.ppm"
        make_wide_image(wide_image)
        cases = ((root / "images/ghost.png", "BCPQ"),
                 (root / "images/ufo_240.jpg", "BCPQ"),
                 (wide_image, "PQ"))
        for source, variants in cases:
            for variant in variants:
                label = f"{source.name} {variant}"
                rust_image = output / f"rust-{source.stem}-{variant}.png"
                cpp_image = output / f"cpp-{source.stem}-{variant}.png"

                run(rust_cli, "-m", models, "encode", "-i", source, "-o", rust_image,
                    "-w", PAYLOAD, "--version", "BCH_5", "--variant", variant)
                actual = run(cpp_runner, "decode-image", models, variant, rust_image)
                expected = f"Found watermark (BCH_5): {PAYLOAD}"
                if actual != expected:
                    raise AssertionError(f"Rust -> C++ failed for {label}: {actual!r}")

                run(cpp_runner, "encode-image", models, variant, source, cpp_image,
                    PAYLOAD, "BCH_5")
                actual = run(rust_cli, "-m", models, "decode", "-i", cpp_image,
                             "--variant", variant)
                expected = f"Found watermark: {PAYLOAD}"
                if actual != expected:
                    raise AssertionError(f"C++ -> Rust failed for {label}: {actual!r}")

                print(f"{label}: Rust -> C++ and C++ -> Rust passed", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: interoperability.py CPP_RUNNER RUST_CLI MODELS_DIR")
    main(*(pathlib.Path(argument) for argument in sys.argv[1:]))
