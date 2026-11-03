// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::cmp;

use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
use image::{
    DynamicImage, GenericImageView as _, GrayAlphaImage, GrayImage, ImageBuffer, Rgb32FImage,
    RgbImage, Rgba32FImage, RgbaImage,
};
use ndarray::{s, Array, Array2, Array3, Array4, ArrayD, Axis, ShapeError};
use ort::value::{TensorValueType, Value};

use crate::Variant;

/// Re-normalize a floating point value (either scalar or array) from the range [0,1] to the range
/// [-1, 1].
macro_rules! convert_from_0_1_to_neg1_1 {
    ($f:expr) => {
        $f * 2. - 1.
    };
}

/// Re-normalize a floating point value (either scalar or array) from the range [-1, 1] to the
/// range [0, 1].
macro_rules! convert_from_neg1_1_to_0_1 {
    ($f:expr) => {
        ($f + 1.) / 2.
    };
}

pub(super) struct ModelImage<'a>(
    pub(super) u32,
    pub(super) Variant,
    pub(super) &'a DynamicImage,
);

/// The error type for the `image_processing` module.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Something went wrong during inference.
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),

    /// We were unable to make an `ndarray::Array` of the requested shape.
    #[error("shape error: {0}")]
    Shape(#[from] ShapeError),

    /// The input array has an unexpected shape.
    #[error("invalid shape")]
    InvalidShape,

    // We could not create an `ImageBuffer` with the requested array.
    #[error("invalid image")]
    Image,

    /// We were unable to resize the input image.
    #[error("resize error: {0}")]
    Resize(#[from] fast_image_resize::ResizeError),
}

impl TryFrom<ModelImage<'_>> for Value<TensorValueType<f32>> {
    type Error = Error;

    fn try_from(img: ModelImage<'_>) -> Result<Self, Self::Error> {
        let array: Array4<f32> = img.try_into()?;
        Ok(Value::from_array(array)?)
    }
}

impl TryFrom<ModelImage<'_>> for Array4<f32> {
    type Error = Error;

    fn try_from(ModelImage(size, variant, img): ModelImage<'_>) -> Result<Self, Self::Error> {
        let (w, h, xpos, ypos) = center_crop_size_and_offset(variant, img);

        let options = ResizeOptions::new()
            .crop(xpos as f64, ypos as f64, w as f64, h as f64)
            .resize_alg(ResizeAlg::Interpolation(
                fast_image_resize::FilterType::Bilinear,
            ));
        let modified_img = resize_img(img, size, size, options)?;

        let img = modified_img.into_rgb32f().into_vec();
        let array = Array::from(img);

        // The `image` crate normalizes to `[0,1]`. Trustmark wants images normalized to `[-1,1]`.
        let array = convert_from_0_1_to_neg1_1!(array);

        let mut array = array
            .to_shape([size as usize, size as usize, 3])?
            .insert_axis(Axis(3))
            .reversed_axes();
        array.swap_axes(2, 3);
        assert_eq!(array.shape(), &[1, 3, size as usize, size as usize]);
        Ok(array.to_owned())
    }
}

/// Red, green and blue: the channels the residual has and is added to. Alpha gets no residual.
const COLOR_CHANNELS: usize = 3;

/// Apply `residual` to the `input`.
///
/// `residual` has shape `[1, 3, height, width]` with values in `[-1, 1]`. It is resized to the
/// size of `input` and added to the color channels; alpha is kept.
///
/// The output equals that of the 0.4.0 implementation (kept as `apply_residual_reference`) bit
/// for bit. The watermark signal is only a few 8-bit steps, so this repeats the arithmetic of
/// `image::imageops::resize` with `FilterType::Triangle` operation for operation, instead of an
/// equivalent formula that rounds differently. Unlike 0.4.0, it makes no full-size copies besides
/// the output: the residual is resized and the input converted one row at a time.
pub(super) fn apply_residual(
    input: DynamicImage,
    residual: &ArrayD<f32>,
) -> Result<DynamicImage, Error> {
    let &[1, COLOR_CHANNELS, residual_height, residual_width] = residual.shape() else {
        return Err(Error::InvalidShape);
    };
    // The model never produces an empty residual, and an empty one cannot be resized.
    if residual_height == 0 || residual_width == 0 {
        return Err(Error::InvalidShape);
    }
    // The residual as the 0.4.0 `Rgb32F` image held it: values in `[0, 1]`, indexed
    // `[channel, row, column]`.
    let residual = Array3::from_shape_fn(
        (COLOR_CHANNELS, residual_height, residual_width),
        |(channel, row, column)| convert_from_neg1_1_to_0_1!(residual[[0, channel, row, column]]),
    );

    let (width, height) = input.dimensions();
    // `imageops::resize` returns an unchanged copy when the size already matches.
    let resize =
        ((width as usize, height as usize) != (residual_width, residual_height)).then(|| {
            ResizeWeights {
                columns: bilinear_resize_weights(residual_width, width),
                rows: bilinear_resize_weights(residual_height, height),
            }
        });

    // The conversion gives the output the input's color space, as in 0.4.0; the color values
    // are overwritten.
    Ok(if input.color().has_alpha() {
        let mut output = input.to_rgba32f();
        add_residual(
            &mut output,
            COLOR_CHANNELS + 1,
            &input,
            &residual,
            resize.as_ref(),
        );
        output.into()
    } else {
        let mut output = input.to_rgb32f();
        add_residual(
            &mut output,
            COLOR_CHANNELS,
            &input,
            &residual,
            resize.as_ref(),
        );
        output.into()
    })
}

/// How to resize the residual to the image: one entry per output column and per output row.
struct ResizeWeights {
    columns: Vec<WeightedInputs>,
    rows: Vec<WeightedInputs>,
}

/// How one output position of a resize along one axis is computed: the weighted sum of the
/// input values at `first_input`, `first_input + 1`, and so on, one per weight.
struct WeightedInputs {
    first_input: usize,
    weights: Vec<f32>,
}

impl WeightedInputs {
    /// The weighted sum of the values `input` returns for each position. The additions run in
    /// position order, as in `imageops::resize`, since the order changes the rounding.
    fn sum(&self, input: impl Fn(usize) -> f32) -> f32 {
        let mut sum = 0.0;
        for (offset, weight) in self.weights.iter().enumerate() {
            sum += input(self.first_input + offset) * weight;
        }
        sum
    }
}

/// Resizes `input_size` positions to `output_size` positions by blending neighboring values, as
/// `imageops::resize` does with `FilterType::Triangle` (image 0.25, `vertical_sample` and
/// `horizontal_sample`). Computed the same way, so the results round the same way.
fn bilinear_resize_weights(input_size: usize, output_size: u32) -> Vec<WeightedInputs> {
    // How many input positions one output position covers.
    let scale = input_size as f32 / output_size as f32;
    // How far from its center an output position blends inputs: one position when enlarging,
    // more when shrinking so that every input counts.
    let reach = if scale < 1.0 { 1.0 } else { scale };

    (0..output_size)
        .map(|output_position| {
            // Where the center of this output position falls, in input positions.
            let center = (output_position as f32 + 0.5) * scale;
            let first = ((center - reach).floor() as i64).clamp(0, input_size as i64 - 1);
            let end = ((center + reach).ceil() as i64).clamp(first + 1, input_size as i64);
            let center = center - 0.5;

            // Nearer inputs count more. Inputs past the edge are left out, and the weights
            // rescaled to add up to 1.
            let mut weights: Vec<f32> = (first..end)
                .map(|input_position| triangle((input_position as f32 - center) / reach))
                .collect();
            let total = weights.iter().fold(0.0, |total, weight| total + weight);
            for weight in &mut weights {
                *weight /= total;
            }
            WeightedInputs {
                first_input: first as usize,
                weights,
            }
        })
        .collect()
}

/// 1 at distance 0, falling in a straight line to 0 at distance 1 and beyond.
fn triangle(distance: f32) -> f32 {
    if distance.abs() < 1.0 {
        1.0 - distance.abs()
    } else {
        0.0
    }
}

/// Writes `input` plus the `residual` into the color channels of `output`, one row at a time.
/// `output` holds `channels` values per pixel. The residual is resized with `resize`, or used
/// as is when that is `None`.
fn add_residual(
    output: &mut [f32],
    channels: usize,
    input: &DynamicImage,
    residual: &Array3<f32>,
    resize: Option<&ResizeWeights>,
) {
    let width = input.width();
    let residual_width = residual.dim().2;
    // One row of the residual, resized to the output height but not yet to the output width.
    let mut residual_row = Array2::zeros((COLOR_CHANNELS, residual_width));

    for (row, output_row) in output
        .chunks_exact_mut(width as usize * channels)
        .enumerate()
    {
        // Resize vertically first, as `imageops::resize` does (`vertical_sample`).
        if let Some(resize) = resize {
            for channel in 0..COLOR_CHANNELS {
                for column in 0..residual_width {
                    residual_row[[channel, column]] =
                        resize.rows[row].sum(|source_row| residual[[channel, source_row, column]]);
                }
            }
        }

        // 0.4.0 read the input converted to `Rgba32F`, whatever the output type.
        let input_row = input.crop_imm(0, row as u32, width, 1).into_rgba32f();

        for (column, output_pixel) in output_row.chunks_exact_mut(channels).enumerate() {
            let input_pixel = input_row.get_pixel(column as u32, 0);
            for channel in 0..COLOR_CHANNELS {
                let residual_value = match resize {
                    // Then horizontally (`horizontal_sample`), which limits the result to
                    // `[0, 1]`.
                    Some(resize) => resize.columns[column]
                        .sum(|source_column| residual_row[[channel, source_column]])
                        .clamp(0.0, 1.0),
                    None => residual[[channel, row, column]],
                };

                // The 0.4.0 `apply_residual` closure: both values to `[-1, 1]`, added, capped
                // at 1, and back to `[0, 1]`.
                let pixel_value = convert_from_0_1_to_neg1_1!(input_pixel[channel]);
                let residual_value = convert_from_0_1_to_neg1_1!(residual_value);
                output_pixel[channel] =
                    convert_from_neg1_1_to_0_1!(f32::min(pixel_value + residual_value, 1.0));
            }
        }
    }
}

/// Return the size and offset of the "center-cropped" image.
///
/// Returns `(width, height, xpos, ypos)` for the square to crop.
///
/// For long-skinny images or short-wide images, we want to crop a square image with side length of
/// the shorter side out of the center of the image for the model.
fn center_crop_size_and_offset(variant: Variant, img: &DynamicImage) -> (u32, u32, u32, u32) {
    let (width, height) = img.dimensions();

    if height > width * 2 || width > height * 2 || variant == Variant::P {
        let m = cmp::min(height, width);
        let offset = (cmp::max(height, width) - m) / 2;

        let xpos;
        let ypos;
        if height > width {
            xpos = 0;
            ypos = offset;
        } else {
            ypos = 0;
            xpos = offset;
        }

        (m, m, xpos, ypos)
    } else {
        (width, height, 0, 0)
    }
}

/// Returns a new `DynamicImage`, resized to `width` by `height` with the specified `ResizeOptions`.
fn resize_img(
    img: &DynamicImage,
    width: u32,
    height: u32,
    options: ResizeOptions,
) -> Result<DynamicImage, Error> {
    let mut modified_img = match img {
        DynamicImage::ImageLuma8(_) => DynamicImage::ImageLuma8(GrayImage::new(width, height)),
        DynamicImage::ImageLumaA8(_) => {
            DynamicImage::ImageLumaA8(GrayAlphaImage::new(width, height))
        }
        DynamicImage::ImageRgb8(_) => DynamicImage::ImageRgb8(RgbImage::new(width, height)),
        DynamicImage::ImageRgba8(_) => DynamicImage::ImageRgba8(RgbaImage::new(width, height)),
        DynamicImage::ImageLuma16(_) => DynamicImage::ImageLuma16(ImageBuffer::new(width, height)),
        DynamicImage::ImageLumaA16(_) => {
            DynamicImage::ImageLumaA16(ImageBuffer::new(width, height))
        }
        DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgb16(ImageBuffer::new(width, height)),
        DynamicImage::ImageRgba16(_) => DynamicImage::ImageRgba16(ImageBuffer::new(width, height)),
        DynamicImage::ImageRgb32F(_) => DynamicImage::ImageRgb32F(Rgb32FImage::new(width, height)),
        DynamicImage::ImageRgba32F(_) => {
            DynamicImage::ImageRgba32F(Rgba32FImage::new(width, height))
        }
        // Technically unreachable, but we error for safety.
        _ => return Err(Error::Image),
    };
    Resizer::new().resize(img, &mut modified_img, &options)?;

    Ok(modified_img)
}

/// Applies the mean padding boundary artifact mitigation.
///
/// Center cropped images have a vertical line problem along the boundary of the residual. This
/// transformation makes this boundary less visible.
pub(super) fn remove_boundary_artifact(
    mut residual: ArrayD<f32>,
    (width, height): (usize, usize),
    _variant: Variant,
) -> ArrayD<f32> {
    // We're going to replace the border of the residual with the mean and also pad the non-center
    // areas with the mean value.
    let channel_means: Vec<f32> = (0_usize..3)
        .map(|i| residual.slice(s![.., i, .., ..]).mean().unwrap())
        .collect();

    // We want one dimension of the output to be 256 and we we want the aspect ratio of the output
    // to match the input image.
    let other_dim;
    let mut mean_padded: ndarray::Array4<f32> = if width > height {
        other_dim = ((width as f32 / height as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 3, 256_usize, other_dim])
    } else {
        other_dim = ((height as f32 / width as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 3, other_dim, 256])
    };

    // This softens the transition between the residual area and the rest of the image.
    let border = 2;
    for (i, mean) in channel_means.iter().enumerate() {
        residual.slice_mut(s![0, i, ..border, ..]).fill(*mean);
        residual.slice_mut(s![0, i, -border.., ..]).fill(*mean);
        residual.slice_mut(s![0, i, .., -border..]).fill(*mean);
        residual.slice_mut(s![0, i, .., ..border]).fill(*mean);
        mean_padded.slice_mut(s![0, i, .., ..]).fill(*mean);
    }

    if width > height {
        let leftover = (other_dim - 256) / 2;
        mean_padded
            .slice_mut(s![.., .., .., leftover..(leftover + 256)])
            .assign(&residual);
    } else {
        let leftover = (other_dim - 256) / 2;
        mean_padded
            .slice_mut(s![.., .., leftover..(leftover + 256), ..])
            .assign(&residual);
    }

    mean_padded.into_dyn()
}

#[cfg(test)]
mod tests {
    use image::{
        imageops::{self, FilterType},
        Pixel as _,
    };

    use super::*;

    /// The 0.4.0 conversion of the residual array into an `Rgb32F` image, kept as the reference
    /// for `apply_residual`.
    fn residual_image_reference(mut array: ArrayD<f32>) -> Result<DynamicImage, Error> {
        let &[1, 3, height, width] = &array.shape().to_owned()[..] else {
            return Err(Error::InvalidShape);
        };
        array.swap_axes(2, 3);
        let array = array.reversed_axes().remove_axis(Axis(3));
        let array = array.to_shape([width * height * 3])?;

        // The `image` crate normalizes to `[0,1]`. Trustmark wants images normalized to `[-1,1]`.
        let array = convert_from_neg1_1_to_0_1!(array);

        let image = Rgb32FImage::from_vec(width as u32, height as u32, array.to_vec())
            .ok_or(Error::Image)?;

        Ok(image.into())
    }

    /// The 0.4.0 `apply_residual`, kept as the reference for `apply_residual`.
    fn apply_residual_reference(input: DynamicImage, residual: DynamicImage) -> DynamicImage {
        let has_alpha = input.color().has_alpha();
        let (w, h) = input.dimensions();

        let applied = {
            let input = input.clone().into_rgba32f();
            let mut target = input.clone();

            let residual = residual.resize_exact(w, h, FilterType::Triangle);
            let residual = residual.into_rgba32f();

            for ((target, residual), original) in target
                .pixels_mut()
                .zip(residual.pixels())
                .zip(input.pixels())
            {
                target.apply2(residual, |x, y| {
                    let x = convert_from_0_1_to_neg1_1!(x);
                    let y = convert_from_0_1_to_neg1_1!(y);

                    convert_from_neg1_1_to_0_1!(f32::min(x + y, 1.0))
                });
                target[3] = original[3];
            }

            target
        };

        if has_alpha {
            let mut input = input.into_rgba32f();
            imageops::replace(&mut input, &applied, 0, 0);
            input.into()
        } else {
            let mut input = input.into_rgb32f();
            let applied = DynamicImage::ImageRgba32F(applied).into_rgb32f();
            imageops::replace(&mut input, &applied, 0, 0);
            input.into()
        }
    }

    #[test]
    fn renormalize_from_0_1() {
        assert_eq!(convert_from_0_1_to_neg1_1!(0.), -1.);
        assert_eq!(convert_from_0_1_to_neg1_1!(0.5), 0.);
        assert_eq!(convert_from_0_1_to_neg1_1!(0.99), 0.98);
    }

    #[test]
    fn renormalize_from_neg1_1() {
        assert_eq!(convert_from_neg1_1_to_0_1!(-1.), 0.);
        assert_eq!(convert_from_neg1_1_to_0_1!(0.5), 0.75);
        assert_eq!(convert_from_neg1_1_to_0_1!(-0.1), 0.45);
    }

    #[test]
    fn normal_image() {
        let image = DynamicImage::new(100, 110, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (100, 110, 0, 0)
        );
    }

    #[test]
    fn skinny_image() {
        let image = DynamicImage::new(10, 100, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (10, 10, 0, 45)
        );
    }

    #[test]
    fn wide_image() {
        let image = DynamicImage::new(101, 10, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::Q, &image),
            (10, 10, 45, 0)
        );
    }

    #[test]
    fn always_crop_p() {
        let image = DynamicImage::new(100, 110, image::ColorType::L8);
        assert_eq!(
            center_crop_size_and_offset(Variant::P, &image),
            (100, 100, 0, 5)
        );
    }

    #[test]
    fn remove_boundary_artifact_tall() {
        let residual: Array4<f32> = Array4::ones([1, 3, 256, 256]);
        let width = 256;
        let height = 298;

        let output = remove_boundary_artifact(residual.into_dyn(), (width, height), Variant::P);

        assert_eq!(output.shape(), &[1, 3, 298, 256]);
    }

    /// Deterministic values in `[0, 1)` (xorshift), so the tests need no random number crate.
    fn noise(mut state: u32) -> impl FnMut() -> f32 {
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1 << 24) as f32
        }
    }

    fn residual(height: usize, width: usize, seed: u32) -> ArrayD<f32> {
        let mut noise = noise(seed);
        Array4::from_shape_simple_fn([1, 3, height, width], || noise() * 0.4 - 0.2).into_dyn()
    }

    fn noise_image(width: u32, height: u32, seed: u32) -> Rgba32FImage {
        let mut noise = noise(seed);
        Rgba32FImage::from_fn(width, height, |_, _| {
            image::Rgba([noise(), noise(), noise(), noise()])
        })
    }

    /// `apply_residual` output against the 0.4.0 implementation, bit for bit.
    fn assert_matches_reference(input: &DynamicImage, residual: &ArrayD<f32>) {
        let actual = apply_residual(input.clone(), residual).unwrap();
        let expected = apply_residual_reference(
            input.clone(),
            residual_image_reference(residual.clone()).unwrap(),
        );
        assert_eq!(actual.color(), expected.color());
        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.color_space(), expected.color_space());
        // Both are `f32` images; compare each value's bytes.
        let (actual, expected) = (
            actual.as_bytes().as_chunks::<4>().0,
            expected.as_bytes().as_chunks::<4>().0,
        );
        let mismatches = actual.iter().zip(expected).filter(|(a, e)| a != e).count();
        assert_eq!(
            mismatches,
            0,
            "{:?} {}x{}, residual {:?}: {mismatches} of {} values differ",
            input.color(),
            input.width(),
            input.height(),
            residual.shape(),
            expected.len()
        );
    }

    const RESIDUAL_SHAPES: [(usize, usize); 3] = [(256, 256), (256, 384), (384, 256)];

    #[test]
    fn apply_residual_matches_reference_on_sample_images() {
        for path in ["../images/ghost.png", "../images/ufo_240.jpg"] {
            let img = image::open(path).unwrap();
            for (seed, (h, w)) in RESIDUAL_SHAPES.into_iter().enumerate() {
                assert_matches_reference(&img, &residual(h, w, seed as u32 + 1));
            }
        }
    }

    #[test]
    fn apply_residual_matches_reference_on_every_color_type() {
        // Larger than, equal to, smaller than and mixed against the residual sizes.
        let sizes = [(600, 450), (256, 256), (256, 300), (100, 60), (200, 500)];
        for (i, (w, h)) in sizes.into_iter().enumerate() {
            let float = DynamicImage::ImageRgba32F(noise_image(w, h, 100 + i as u32));
            let inputs = [
                DynamicImage::ImageLuma8(float.to_luma8()),
                DynamicImage::ImageLumaA8(float.to_luma_alpha8()),
                DynamicImage::ImageRgb8(float.to_rgb8()),
                DynamicImage::ImageRgba8(float.to_rgba8()),
                DynamicImage::ImageLuma16(float.to_luma16()),
                DynamicImage::ImageLumaA16(float.to_luma_alpha16()),
                DynamicImage::ImageRgb16(float.to_rgb16()),
                DynamicImage::ImageRgba16(float.to_rgba16()),
                DynamicImage::ImageRgb32F(float.to_rgb32f()),
                float,
            ];
            for input in &inputs {
                for (seed, (rh, rw)) in RESIDUAL_SHAPES.into_iter().enumerate() {
                    assert_matches_reference(input, &residual(rh, rw, seed as u32 + 10));
                }
            }
        }
    }

    #[test]
    fn apply_residual_matches_reference_on_out_of_range_floats() {
        let mut noise = noise(7);
        let img = Rgb32FImage::from_fn(300, 280, |_, _| {
            image::Rgb([
                noise() * 3.0 - 1.0,
                noise() * 3.0 - 1.0,
                noise() * 3.0 - 1.0,
            ])
        });
        assert_matches_reference(&DynamicImage::ImageRgb32F(img), &residual(256, 256, 3));
    }

    #[test]
    fn apply_residual_matches_reference_on_out_of_range_residual() {
        // `encode` limits the residual to `[-0.2, 0.2]`; `apply_residual` must still match 0.4.0
        // beyond that, with and without resizing.
        let residual = residual(256, 256, 4).mapv(|value| value * 8.0);
        for (w, h) in [(256, 256), (300, 280)] {
            let img = DynamicImage::ImageRgba32F(noise_image(w, h, 9));
            assert_matches_reference(&img, &residual);
        }
    }

    #[test]
    fn apply_residual_rejects_bad_shape() {
        for shape in [[1, 4, 256, 256], [1, 3, 0, 256], [1, 3, 256, 0]] {
            let img = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
            let residual = Array4::<f32>::zeros(shape).into_dyn();
            assert!(matches!(
                apply_residual(img, &residual),
                Err(Error::InvalidShape)
            ));
        }
    }
}
