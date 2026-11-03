# Lower peak memory in the Rust `Trustmark::encode`

This note explains a change to the Rust crate in `rust/`: what was changed, the arithmetic the new
code performs, and where the memory saving comes from. It is written for engineers who know Rust
but not image processing.

## What changed and what did not

Two files changed: `rust/src/image_processing.rs` and `rust/src/lib.rs`.

The public interface of the crate is unchanged. `Trustmark::encode` and `Trustmark::decode` keep
their signatures. `encode` still returns an `Rgb32F` image, or `Rgba32F` when the input has an
alpha channel, with the same color space metadata as before. Every value in the returned image is
bit-for-bit identical to what trustmark 0.4.0 returns for the same input, strength and model.

The changes are internal:

- `ModelImage` borrows the input image instead of owning a copy, which removes the two
  `img.clone()` calls in `encode`.
- `encode` crops and resizes the input to the 256x256 model input once and reuses that array for
  the residual; 0.4.0 did it twice for the same result.
- `apply_residual` is rewritten. It takes the residual as the array the model produced and writes
  the result one row at a time, without building full-size floating point copies of the input or
  of the resized residual.
- `encode` returns an error for an image with a width or height of 0, where 0.4.0 panicked.

## How a watermark is added

### Pixel values and the residual

An image is a grid of pixels, and each pixel has a red, a green and a blue value (and sometimes an
alpha value for transparency, which the watermark never touches). The code works with these values
as numbers between 0 and 1: 0 is no light, 1 is full intensity.

The watermarking model only accepts images of 256 by 256 pixels. `encode` shrinks the input to that
size, runs the model, and subtracts the model's input from its output. The difference is the
residual: a 256 by 256 grid of small corrections, one per channel, which encode the watermark. The
code multiplies it by the requested strength and limits each correction to plus or minus 0.2. For
the P variant, and for Q images more than twice as wide as tall (or the reverse), the residual is
padded to match the image's proportions.

This is Algorithm 1 of the TrustMark paper (arXiv 2311.18297): compute the residual at the model's
size, enlarge it to the image's size, add it to the original image.

### Enlarging the residual

The residual has to be stretched to the size of the real image, for example from 256 to 6000
columns. Each column of the large image falls somewhere between two columns of the small residual,
and its value is a blend of those two, weighted by distance. A position 30% of the way from column
4 to column 5 gets 70% of column 4's value and 30% of column 5's. Positions past the first or last
column take the edge value. Rows are handled the same way. This is bilinear interpolation, which
`image::imageops::resize` performs with `FilterType::Triangle`.

When the image is smaller than the residual, the same function averages over a wider window of
residual columns, so every residual value contributes.

`imageops::resize` does the two directions in sequence: it first resizes every column to the new
height, then every row to the new width. Because each output row only needs the two or so residual
rows nearest to it, the new code can produce the resized residual one output row at a time.

### Adding it to the image

For each pixel and each color channel, 0.4.0 computed, in 32-bit floating point:

```text
x = 2 * pixel - 1          pixel value moved from [0, 1] to [-1, 1]
r = 2 * residual - 1       residual value moved from [0, 1] to [-1, 1]
out = (min(x + r, 1) + 1) / 2
```

The residual is stored in the `[0, 1]` form, `(r + 1) / 2`, because 0.4.0 held it in an image
type. The new code keeps that form, and the same three lines, so it performs the same
floating point operations in the same order.

### Why the arithmetic is repeated operation for operation

The planning document for this change proposed a shorter formula that gives the same result in
exact arithmetic, accepting a difference of one 8-bit step in some pixels. That difference is too
large here. The figures in the paper show residuals scaled up 20 and 100 times to make them
visible, so the watermark changes a typical pixel by only a few 8-bit steps, and a rounding change
of one step is a noticeable part of the signal.

Floating point results depend on the order of operations. `(a + b) + c` and `a + (b + c)` can
differ in the last bit, and after conversion to 8 bits that can move a pixel by one step. The new
code therefore computes the blending weights exactly as `imageops::resize` does in image 0.25.10
(`vertical_sample` and `horizontal_sample`):

- the same scale factor and center position;
- the same window of input positions, including those with zero weight;
- the same rescaling of the weights so they add up to 1;
- the same summation order;
- the same limit to `[0, 1]` after the horizontal pass;
- the same shortcut, a plain copy, when the residual already has the image's size.

Input pixels are converted to floating point by the `image` crate itself, one row at a time, with
the same conversion 0.4.0 applied to the whole image.

To check that the tests detect a change of this size, one run removed only the rescaling of the
weights, a change that affects nothing but rounding. The bit-for-bit test then reported 6132 of
1,048,576 values different on `ghost.png`.

## Memory

### Where 0.4.0 spent it

For an input of N pixels, 0.4.0 allocated, among others:

- two copies of the input image in `encode`, used only to shrink them to 256 by 256;
- in `apply_residual`, two 32-bit RGBA copies of the input (16 bytes per pixel each);
- the residual enlarged to full size as 32-bit RGB (12 bytes per pixel), then converted to RGBA
  (16 bytes per pixel);
- at the end, a 32-bit RGB copy of the input and a 32-bit RGB copy of the result.

Several of these are alive at the same moment, on top of the decoded input.

### Where the new code spends it

- The input image, as the caller decoded it, consumed by `encode` (3 bytes per pixel for an
  ordinary RGB JPEG).
- The returned image (12 bytes per pixel for RGB, 16 with alpha), a size set by the public return
  type.
- Per row: one row of the input converted to floating point, and one row of the residual resized
  to the output height.
- The residual itself, 256 by at most a few hundred values.

### Measured

Measured with `/usr/bin/time -l` on macOS, on a 6000 by 4000 RGB JPEG (24 million pixels), model Q,
two runs each. "Baseline" loads the models and decodes the image without encoding.

| Run | Peak resident memory | Above baseline, per pixel |
| --- | --- | --- |
| Baseline | 223 MB | |
| 0.4.0 `encode` | 1868 MB | 68.5 bytes |
| new `encode` | 589 MB | 15.3 bytes |

The 15.3 bytes include the 12-byte output and the inference workspace of the encoder model, which
these runs do not separate. Run time for the same image went from 0.57 to 0.48 seconds. The crate's
`cargo bench` (`benches/encode.rs`, a 240 by 240 image) measured 96.7 ms for 0.4.0 and 90.6 ms for
the new code.

Going lower requires an 8-bit result, and `encode` returns 32-bit floating point values. An
additional method that returns `Rgb8` or `Rgba8`, written into the input buffer, would bring the
cost to about 3 bytes per pixel above the decoded image. It is not part of this change.

## How it was checked

- Unit tests in `image_processing.rs` compare `apply_residual` with the 0.4.0 implementation, kept
  in the test module as `apply_residual_reference`, and require identical bytes, color type,
  dimensions and color space. They cover ten pixel formats (8-bit, 16-bit and 32-bit, gray and
  color, with and without alpha), image sizes larger than, equal to, smaller than and mixed against
  the residual, three residual shapes, values outside `[0, 1]`, and residuals outside the range
  `encode` produces.
- An end-to-end comparison built 0.4.0 and the new code into one program and ran `encode` with the
  real Q and P models on 28 inputs: the sample images, gray, 16-bit and floating point versions of
  `ripley.jpg`, and synthetic images from 1x1 to 4000x10. All outputs were identical, as were the
  decode results. Two runs of 0.4.0 agreed with each other, so the comparison is not hiding
  nondeterminism.
- The same program compared the two versions on NaN, infinite, negative, zero and large strengths,
  on images exactly at and one pixel past the 2:1 aspect limit, and on images with Display P3
  primaries and a linear transfer function. All outputs were identical.
- The Python implementation in `python/` decoded all 20 Rust-encoded images (Q and P, from 0.4.0
  and from the new code), and both Rust versions decoded the 6 images encoded by Python.
  Unwatermarked images decoded as unwatermarked in both languages. The Python encoder uses a
  different algorithm (it removes the per-channel mean of the residual and uses PyTorch's
  interpolation), so its output bytes differ from Rust's, as they did with 0.4.0. Image quality is
  similar: 41.7 to 44.2 dB PSNR for Rust and 41.8 to 44.6 dB for Python with model Q, 46.8 to
  50.5 dB and 47.0 to 50.8 dB with model P.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` report nothing,
  and `cargo test --workspace` passes.

## Behavior to know about

One case changed. `encode` on an image with a width or height of 0 panicked inside `ndarray` in
0.4.0. The new code returns `Error::ImageProcessing(image_processing::Error::Image)` ("invalid image")
before running the model. `decode` on such an image returns `Error::CorruptWatermark`, in 0.4.0
and in the new code.

The following cases behave the same in both versions; they are listed because a reader may not
expect them.

- An image of 100 by 60 pixels encodes, but its watermark does not decode, with either version.
- A residual with an empty dimension makes the internal `apply_residual` return
  `Error::InvalidShape`. 0.4.0 resized such a residual to zeros instead. `encode` cannot produce
  one: the model returns 256 by 256 values, and the padding only adds to that.
