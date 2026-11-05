// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::cmp;

use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
use image::{
    imageops::FilterType, DynamicImage, GenericImageView as _, GrayAlphaImage, GrayImage,
    ImageBuffer, Pixel, Rgb32FImage, RgbImage, Rgba32FImage, RgbaImage,
};
use ndarray::{s, Array, Array4, ArrayD, Axis, IxDyn, ShapeError};
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

/// Converts a `[1, 3, height, width]` residual array into an `Rgb32F` image.
/// Converts in place for pixel-order arrays and copies other layouts once.
pub(super) fn residual_image(array: ArrayD<f32>) -> Result<DynamicImage, Error> {
    let &[1, 3, height, width] = array.shape() else {
        return Err(Error::InvalidShape);
    };
    // `[height, width, 3]`, the pixel order of an image.
    let mut pixels = array
        .permuted_axes(IxDyn(&[0, 2, 3, 1]))
        .remove_axis(Axis(0));
    if !pixels.is_standard_layout() {
        pixels = pixels.as_standard_layout().into_owned();
    }

    // Trustmark's residual is in `[-1,1]`. The `image` crate wants `[0,1]`.
    pixels.mapv_inplace(|value| convert_from_neg1_1_to_0_1!(value));

    // In standard layout the values are contiguous, starting at `offset` in the buffer.
    let length = pixels.len();
    let (mut values, offset) = pixels.into_raw_vec_and_offset();
    values.drain(..offset.unwrap_or(0));
    values.truncate(length);

    let image = Rgb32FImage::from_vec(width as u32, height as u32, values).ok_or(Error::Image)?;

    Ok(image.into())
}

/// The residual affects red, green and blue, but not alpha.
const COLOR_CHANNELS: usize = 3;

/// Resizes the residual to the input size and adds it to the input.
/// Frees the original residual and writes the sum into an `Rgb32F` or `Rgba32F` image.
pub(super) fn apply_residual(input: DynamicImage, residual: DynamicImage) -> DynamicImage {
    let (w, h) = input.dimensions();
    let resized = residual
        .resize_exact(w, h, FilterType::Triangle)
        .into_rgb32f();
    drop(residual);

    if input.color().has_alpha() {
        let mut output = input.into_rgba32f();
        add_residual(&mut output, &resized);
        output.into()
    } else {
        let mut output = input.into_rgb32f();
        add_residual(&mut output, &resized);
        output.into()
    }
}

/// Adds `residual` to the color channels of `image`.
fn add_residual<P: Pixel<Subpixel = f32>>(
    image: &mut ImageBuffer<P, Vec<f32>>,
    residual: &Rgb32FImage,
) {
    for (pixel, residual_pixel) in image.pixels_mut().zip(residual.pixels()) {
        let values = pixel.channels_mut();
        for channel in 0..COLOR_CHANNELS {
            // Both values to `[-1, 1]`, added, capped at 1, and back to `[0, 1]`.
            let pixel_value = convert_from_0_1_to_neg1_1!(values[channel]);
            let residual_value = convert_from_0_1_to_neg1_1!(residual_pixel[channel]);
            values[channel] =
                convert_from_neg1_1_to_0_1!(f32::min(pixel_value + residual_value, 1.0));
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

    // We want one dimension of the output to be 256 and we want the aspect ratio of the output
    // to match the input image.
    // The array has shape `[1, 3, height, width]` but is stored in pixel order,
    // `[1, height, width, 3]`, so `residual_image` needs no copy of it.
    let other_dim;
    let mut mean_padded: ndarray::Array4<f32> = if width > height {
        other_dim = ((width as f32 / height as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, 256_usize, other_dim, 3]).permuted_axes([0, 3, 1, 2])
    } else {
        other_dim = ((height as f32 / width as f32) * 256.0) as usize;
        ndarray::Array4::zeros([1, other_dim, 256, 3]).permuted_axes([0, 3, 1, 2])
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
    use super::*;

    #[test]
    fn apply_residual_preserves_alpha_and_resizes() {
        let residual = DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(
            1,
            1,
            image::Rgb([0.25, 0.25, 0.75]),
        ));
        let inputs = [
            DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(2, 3, image::Rgb([0.0, 0.5, 0.75]))),
            DynamicImage::ImageRgba32F(Rgba32FImage::from_pixel(
                2,
                3,
                image::Rgba([0.0, 0.5, 0.75, 0.375]),
            )),
        ];
        for input in inputs {
            let has_alpha = input.color().has_alpha();
            let output = apply_residual(input, residual.clone());
            assert_eq!(output.dimensions(), (2, 3));
            assert_eq!(output.color().has_alpha(), has_alpha);
            for pixel in output.to_rgba32f().pixels() {
                assert_eq!(
                    pixel.0,
                    [-0.25, 0.25, 1.0, if has_alpha { 0.375 } else { 1.0 }]
                );
            }
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

    #[test]
    fn residual_image_on_any_layout() {
        let value = |n: usize, c: usize, y: usize, x: usize| {
            ((n * 3 + c) * 600 + y * 30 + x) as f32 / 4000.0 - 0.5
        };
        let channel_order = Array4::from_shape_fn([1, 3, 20, 30], |(n, c, y, x)| value(n, c, y, x));
        let mut reversed_rows = channel_order.clone();
        reversed_rows.invert_axis(Axis(2));
        let pixel_order = Array4::from_shape_fn([2, 20, 30, 3], |(n, y, x, c)| value(n, c, y, x));
        let mut at_offset = pixel_order.clone();
        at_offset.slice_collapse(s![1.., .., .., ..]);
        let mut with_trailing_values = pixel_order;
        with_trailing_values.slice_collapse(s![..1, .., .., ..]);
        let arrays = [
            channel_order,
            reversed_rows,
            at_offset.permuted_axes([0, 3, 1, 2]),
            with_trailing_values.permuted_axes([0, 3, 1, 2]),
            Array4::zeros([1, 3, 0, 30]),
        ];
        for array in arrays {
            let (height, width) = (array.shape()[2], array.shape()[3]);
            let expected = Rgb32FImage::from_fn(width as u32, height as u32, |x, y| {
                image::Rgb(std::array::from_fn(|c| {
                    convert_from_neg1_1_to_0_1!(array[[0, c, y as usize, x as usize]])
                }))
            });
            let actual = residual_image(array.into_dyn()).unwrap();
            assert_eq!(actual, DynamicImage::ImageRgb32F(expected));
        }
        assert!(matches!(
            residual_image(Array4::zeros([1, 4, 2, 2]).into_dyn()),
            Err(Error::InvalidShape)
        ));
    }
}
