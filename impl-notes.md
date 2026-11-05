# Lower peak memory in the Rust `Trustmark::encode`

This note explains a change to the Rust crate in `rust/`: what was changed, the arithmetic the code
performs, and where the memory saving comes from. It is written for engineers who know Rust but
not image processing.

## What changed and what did not

Two files changed: `rust/src/image_processing.rs` and `rust/src/lib.rs`.

The public interface of the crate is unchanged. `Trustmark::new`, `Trustmark::encode` and
`Trustmark::decode` keep their signatures, and the error enums keep their variants. `encode`
still returns an `Rgb32F` image, or `Rgba32F` when the input has an alpha channel, with the same
color space metadata as before. Every value in the returned image is bit-for-bit identical to what
trustmark 0.4.0 returns for the same input, strength and model.

The changes are internal:

- `ModelImage` borrows the input image instead of owning a copy, which removes the two
  `img.clone()` calls in `encode`.
- `encode` crops and resizes the input to the 256x256 model input once and reuses that array for
  the residual; 0.4.0 did it twice for the same result.
- `remove_boundary_artifact` stores the padded residual in pixel order, and `residual_image`
  converts it into an image in place. 0.4.0 held three copies of the padded residual at once.
- `apply_residual` writes the result into the input converted to `Rgb32F` (or `Rgba32F` with
  alpha). 0.4.0 made two `Rgba32F` copies of the input, converted the resized residual to
  `Rgba32F`, and copied the result into a third buffer at the end. The residual is still resized
  by the same `image` call as in 0.4.0.
- With the P and Q variants, `encode` returns an error for an image with a width or height of 0,
  where 0.4.0 panicked.

## How a watermark is added

### Pixel values and the residual

An image is a grid of pixels, and each pixel has a red, a green and a blue value (and sometimes an
alpha value for transparency, which the watermark never touches). The code works with these values
as numbers between 0 and 1: 0 is no light, 1 is full intensity.

The watermarking model only accepts images of 256 by 256 pixels. `encode` shrinks the input to that
size, runs the model, and subtracts the model's input from its output. The difference is the
residual: a 256 by 256 grid of small corrections, one per channel, which encode the watermark. The
code multiplies it by the requested strength and limits each correction to plus or minus 0.2.

This is Algorithm 1 of the TrustMark paper (arXiv 2311.18297): compute the residual at the model's
size, enlarge it to the image's size, add it to the original image.

### Padding the residual

For the P variant, and for Q images more than twice as wide as tall (or the reverse), the model
only saw a square cut from the middle of the image. `remove_boundary_artifact` then builds a
larger residual with the proportions of the whole image: the 256 by 256 residual sits in the
middle, and the rest is filled with the residual's average value for each channel. For a 1000 by
250 image this padded residual is 1024 by 256.

The padded residual is an array with the shape `[1, 3, height, width]`: all red values first, then
all green, then all blue. An image buffer stores the three values of each pixel next to each
other. 0.4.0 allocated the array in the first order and copied it twice while converting it into
an image. The array now keeps the same shape and the same values, but its memory is laid out in
pixel order, so `residual_image` converts each value from `[-1, 1]` to `[0, 1]` in place and hands
the buffer to the image without copying it. Arrays in the other order, such as the unpadded 256 by
256 residual, are copied once.

### Enlarging the residual

The residual has to be stretched to the size of the real image, for example from 256 to 6000
columns. Each column of the large image falls somewhere between two columns of the small residual,
and its value is a blend of those two, weighted by distance. A position 30% of the way from column
4 to column 5 gets 70% of column 4's value and 30% of column 5's. Positions past the first or last
column take the edge value. Rows are handled the same way. This is bilinear interpolation, which
`image::imageops::resize` performs with `FilterType::Triangle`. Both 0.4.0 and the new code call it
through `DynamicImage::resize_exact`.

### Adding it to the image

For each pixel and each color channel, 0.4.0 computed, in 32-bit floating point:

```text
x = 2 * pixel - 1          pixel value moved from [0, 1] to [-1, 1]
r = 2 * residual - 1       residual value moved from [0, 1] to [-1, 1]
out = (min(x + r, 1) + 1) / 2
```

The new code runs the same three lines, in the same order, on each pixel of the converted input,
and overwrites the color values in place. Alpha values are not written. 0.4.0 also left alpha
unchanged: it computed the formula on all four channels and then copied the original alpha back.

### Why the output is bit-for-bit identical

The watermark changes a typical pixel by only a few 8-bit steps; the figures in the paper show
residuals scaled up 20 and 100 times to make them visible. Floating point results depend on the
order of operations, and after conversion to 8 bits a rounding difference can move a pixel by one
step, a noticeable part of that signal. So the change keeps every floating point operation of
0.4.0:

- The padded residual holds the same values; only the order of the values in memory changed.
- The conversion of the residual to `[0, 1]` is the same `(value + 1) / 2` on each value. A probe
  ran all 4,294,967,296 possible 32-bit floating point values, NaN and infinities included,
  through the 0.4.0 conversion and the new one, in both memory orders, and found no differing bit.
- The residual is resized by the same `image` call.
- The input is converted by the `image` crate. With alpha, this is the same `Rgba32F` conversion
  0.4.0 used. Without alpha, the new code converts to `Rgb32F`, where 0.4.0 read from an
  `Rgba32F` copy. The two conversions give the same color values: a probe compared them bit for
  bit for every 8-bit and 16-bit value in all ten pixel formats, and for 4,006 floating point
  values including NaN and infinities, in sRGB, Display P3 and linear color spaces, and found no
  difference.
- The sum is the 0.4.0 formula.

An earlier version of this change also replaced `imageops::resize` with code that produced the
resized residual one row at a time. It saved about 13 more bytes per pixel and was bit-identical,
but it reproduced the internals of the `image` crate. It was dropped in favor of calling the crate.

## Memory

### Where 0.4.0 spent it

For an input of N pixels, 0.4.0 allocated, among others:

- two copies of the input image in `encode`, used only to shrink them to 256 by 256;
- three copies of the padded residual while converting it into an image;
- in `apply_residual`, two 32-bit RGBA copies of the input (16 bytes per pixel each);
- the residual enlarged to full size as 32-bit RGB (12 bytes per pixel), then converted to RGBA
  (16 bytes per pixel);
- at the end, a 32-bit RGB copy of the input and a 32-bit RGB copy of the result.

Several of these are alive at the same moment, on top of the decoded input.

### Where the new code spends it

- The input image, as the caller decoded it, consumed by `encode` (3 bytes per pixel for an
  ordinary RGB JPEG). It is freed once converted.
- One copy of the padded residual, freed once resized.
- The residual enlarged to full size as 32-bit RGB (12 bytes per pixel).
- The returned image (12 bytes per pixel for RGB, 16 with alpha), a size set by the public return
  type.
- The intermediate buffer of the resize, 16 bytes per value for the residual's width times the
  image's height.

The padded residual has 256 by (256 times the long side divided by the short side) values per
channel, about 786 KB for each unit of that ratio. For most photographs it is a few megabytes.
For very thin images it is larger than the image itself: a 1 by 2000 image gets a 256 by 512,000
residual, 1.57 GB. 0.4.0 held three of these at once; the new code holds one.

### Measured

Measured with `/usr/bin/time -l` on macOS (Apple silicon), peak resident memory in MB of 10^6
bytes, model Q, two runs each, with nothing else running. "Baseline" loads the models and decodes
the image without encoding.

| Input | Baseline | 0.4.0 `encode` | new `encode` |
| --- | --- | --- | --- |
| 6000 by 4000 RGB JPEG | 221 MB | 1866 to 1875 MB, 0.55 to 0.57 s | 892 to 895 MB, 0.41 s |
| 1 by 2000 RGB PNG | 147 to 150 MB | 4954 to 4957 MB, 3.3 s | 1808 MB, 1.4 s |
| 1 by 4000 RGB PNG | 145 to 146 MB | 9680 MB, 6.4 to 6.5 s | 3389 to 3391 MB, 2.7 s |

Above the baseline, the 6000 by 4000 encode uses 68.5 to 68.9 bytes per pixel in 0.4.0 and 28.0
to 28.1 bytes per pixel in the new code.

The crate's `cargo bench` (`benches/encode.rs`, a 240 by 240 image, dominated by model inference)
measured 92.4 ms (91.8 to 93.0) for 0.4.0 and 93.8 ms (93.0 to 94.8) for the new code, about 1.5%
slower; an earlier run of both gave 94.5 and 95.8 ms. The large images in the table encode faster
with the new code.

Two ways to go lower, neither part of this change:

- Resizing the residual one row at a time avoids the 12-byte full-size residual (measured at 15.4
  bytes per pixel above baseline), at the cost of reproducing the `image` crate's resize.
- An additional method that returns `Rgb8` or `Rgba8`, written into the input buffer, removes the
  floating point output as well.

Memory for very thin images still grows with the ratio of the long side to the short side, about
786 KB per unit, whatever the number of pixels. A service that accepts arbitrary images needs to
limit that ratio itself. Rejecting such images in the crate would change behavior.

## How it was checked

Every check was rerun after each code change.

- During development, unit tests kept the 0.4.0 `apply_residual`, `residual_image`,
  `remove_boundary_artifact` and `encode` bodies as reference functions and required identical
  values, buffer length, color type, dimensions and color space. They covered:
  - `apply_residual` on ten pixel formats, image sizes larger than, equal to, smaller than and
    mixed against the residual, three residual shapes, and floating point inputs outside `[0, 1]`;
  - `remove_boundary_artifact` and `residual_image` on wide, tall, exactly 2:1, one pixel past
    2:1, square and thin sizes, with NaN, infinite and negative zero residual values, and on
    arrays stored in channel order, pixel order, with an offset, with trailing values, with
    reversed rows and empty;
  - `encode` against the reference `encode` with the real B, C, P and Q models, on seven inputs
    and two strengths.

  Those tests failed on changes that only alter rounding: computing `(value + 1) / 2` as
  `(value + 1) * 0.3 / 0.6` in `residual_image` made 49,092 values differ, and multiplying the
  strength into each side of the subtraction in `encode` made 80,826 values differ. The crate's
  tests no longer include the reference code. `residual_image_on_any_layout` checks
  `residual_image` against values computed from the array for each of those storage layouts.
- A comparison program built 0.4.0 and the new code into one binary and ran `encode` with the real
  Q and P models on 28 inputs (the sample images, gray, 16-bit and floating point versions of
  `ripley.jpg`, and synthetic images from 100x60 to 2000x150), and on 112 edge cases: 0x0, 0x10,
  10x0, 1x1, 1x300, 300x1, 1x2000, 2000x1, 4000x10 and 10x4000 images; images exactly at and one
  pixel past the 2:1 limit; NaN, infinite, negative, zero, negative zero and large strengths; all
  ten pixel formats at three sizes with varying alpha; Display P3 and linear color spaces. Each
  version ran each case twice, and both versions were deterministic. All outputs were identical
  except the accepted empty-image error, and the program reported a difference when one input bit
  was changed.
- The same program ran `encode` on 0x0, 0x10 and 10x0 images with all four variants and with a
  valid and an invalid watermark. All outcomes match 0.4.0 except the accepted P and Q error.
- Both versions decoded the 12 images in `images/` and `c2pa/` with each of the B, C, P and Q
  models and returned the same result in all 48 cases.
- Five implementations decoded each other's watermarks: Rust 0.4.0, the new Rust code, Python
  (`python/`), C++ (`cpp/`, built with OpenCV and ONNX Runtime) and JavaScript (`js/`, unmodified,
  in headless Chrome with WebGPU). Rust 0.4.0, the new Rust code, Python and C++ each encoded Q and
  P watermarks into five images, 40 files in all. Every Rust, Python and C++ decoder read the
  watermark from all 40, and the JavaScript decoder read it from all 40 at full size. The new and
  0.4.0 Rust encodings saved as 8-bit PNG were byte-identical in pairs.
- Thirty sequential encodes of the 9-megapixel `ripley.jpg` held resident memory between 682 and
  684 MB. Eight threads sharing one `Trustmark` ran 48 encodes, each followed by a decode, with no
  output differing from the sequential result, no decode failure and no error.
- Independent review passes probed the public interface of 0.4.0 and the new code side by side on
  5,832 further cases (every variant and BCH version, invalid watermark strings, extreme strengths,
  all pixel formats at 15 sizes, special floating point pixels, non-sRGB color spaces, encode and
  decode), and found no difference other than the accepted one.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` report nothing,
  and `cargo test --workspace` passes.

## Behavior to know about

One case changed. With the P and Q variants, `encode` on an image with a width or height of 0
panicked inside `remove_boundary_artifact` in 0.4.0. The new code returns
`Error::ImageProcessing(image_processing::Error::Image)` ("invalid image") at the same point,
after the watermark is validated and the model has run, so an invalid watermark is still reported
as `Error::Bits`. With the B and C variants such an image returns an empty image, as in 0.4.0.
`decode` on an empty image returns `Error::CorruptWatermark`, in 0.4.0 and in the new code.

The following cases behave the same in 0.4.0 and in the new code; they are listed because a reader
may not expect them.

- Thin images encode, but their watermark does not decode. On strips cut from `ripley.jpg` with a
  long side of 2000 pixels, Q watermarks decoded from a short side of 96 pixels and P watermarks
  from 64. A 2000 by 32 Q strip decoded to a different payload, reported as valid. A 100 by 60
  image also encodes without a decodable watermark.
- Decoders report watermarks in some unwatermarked images. The Rust P decoder reports the same
  75-bit payload for `c2pa/example.jpg` and `images/ufo_240.jpg`, which Python reads as
  unwatermarked; the JavaScript decoder reports a payload for both. The Rust and C++ Q decoders
  report one for `images/ghost.png` converted to RGB, and the C++ P decoder for `ripley.jpg`.
- Python and JavaScript find no watermark in `c2pa/example_wm.jpg` or
  `c2pa/example_wm_signed.jpg`. The Rust B and P decoders report payloads for both, and the Rust Q
  decoder reports none.
- The JavaScript example page shrinks images to 500 pixels wide before decoding. After that step it
  read a different payload from one of the 40 files, a P watermark encoded by Python into
  `ripley.jpg`; at full size it read the right one.
