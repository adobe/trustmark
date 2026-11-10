// Copyright 2026 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{path::Path, sync::Mutex};

use image::DynamicImage;
use ndarray::ArrayD;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
    value::{TensorValueType, Value, ValueType},
};

use crate::{image_processing, image_processing::ModelImage, Error, Variant};

const MODEL_SIZE: u32 = 256;

/// An explicitly loaded TrustMark watermark remover.
///
/// Loads only `remover_{variant}.onnx`. Callers using only [`crate::Trustmark`] incur
/// no additional removal cost. Constructing a remover loads a separate model session,
/// and removal allocates image and inference buffers. Removal is lossy and does not
/// guarantee restoration of the original pixels or removal of arbitrary watermarks.
///
/// ```no_run
/// use trustmark::{Variant, WatermarkRemover};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let remover = WatermarkRemover::new("./models", Variant::Q)?;
/// let recovered = remover.remove(image::open("watermarked.png")?)?;
/// recovered.to_rgba8().save("recovered.png")?;
/// # Ok(())
/// # }
/// ```
pub struct WatermarkRemover {
    session: Mutex<Session>,
    variant: Variant,
}

impl WatermarkRemover {
    /// Load the remover for the variant used to embed the watermark.
    ///
    /// Download P/Q models with `cargo xtask fetch-removers` from the Rust workspace.
    /// Other variants require separately supplied, matching `remover_{variant}.onnx` files.
    /// The model must accept and return one float32 NCHW image in `[-1, 1]`.
    pub fn new(models: impl AsRef<Path>, variant: Variant) -> Result<Self, Error> {
        let path = models.as_ref().join(format!("remover_{variant}.onnx"));
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(&path)
            .map_err(|source| Error::RemoverModel { path, source })?;
        validate_model(&session)?;
        Ok(Self {
            session: Mutex::new(session),
            variant,
        })
    }

    /// Remove a supported TrustMark watermark, preserving dimensions and alpha.
    ///
    /// RGB is processed at 8-bit precision, as in the Python reference; alpha is
    /// preserved separately. An image need not contain a detectable watermark: inference still succeeds,
    /// but may change its pixels. Uses Python's default removal strength of 1.0
    /// (1.25 for variant P). Inference on a shared remover is serialized.
    ///
    /// Returns an `ImageRgb32F` or `ImageRgba32F`, like [`crate::Trustmark::encode`].
    /// Convert with [`DynamicImage::to_rgb8`] or [`DynamicImage::to_rgba8`] before
    /// saving PNG or JPEG, whose encoders do not accept floating-point pixels.
    pub fn remove(&self, img: DynamicImage) -> Result<DynamicImage, Error> {
        if img.width() == 0 || img.height() == 0 {
            return Err(image_processing::Error::Image.into());
        }
        let rgb = img.to_rgb8().into();
        let input: Value<TensorValueType<f32>> =
            ModelImage(MODEL_SIZE, self.variant, &rgb).try_into()?;
        drop(rgb);
        let original = input.try_extract_array::<f32>()?.to_owned();
        let output = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::SessionLockPoisoned)?;
            let input_name = session.inputs[0].name.clone();
            let outputs = session.run(ort::inputs![input_name.as_str() => input])?;
            outputs[0].try_extract_array::<f32>()?.to_owned()
        };
        let residual = removal_residual(original, output, self.variant)?;
        Ok(image_processing::apply_removal_residual(
            img,
            residual,
            self.variant,
        ))
    }
}

fn removal_residual(
    input: ArrayD<f32>,
    output: ArrayD<f32>,
    variant: Variant,
) -> Result<ArrayD<f32>, Error> {
    if output.shape() != input.shape() || output.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidRemoverModel);
    }
    Ok((output.mapv(|v| v.clamp(-1., 1.)) - input) * variant.strength_multiplier())
}

fn validate_model(session: &Session) -> Result<(), Error> {
    let valid = |ty: &ValueType| {
        matches!(ty, ValueType::Tensor { ty: TensorElementType::Float32, shape, .. }
            if shape.as_ref() == [1, 3, MODEL_SIZE as i64, MODEL_SIZE as i64])
    };
    if session.inputs.len() != 1
        || session.outputs.len() != 1
        || !valid(&session.inputs[0].input_type)
        || !valid(&session.outputs[0].output_type)
    {
        return Err(Error::InvalidRemoverModel);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_has_path() {
        let result = WatermarkRemover::new("./models/missing", Variant::Q);
        assert!(matches!(result, Err(Error::RemoverModel { path, .. })
            if path.ends_with("remover_Q.onnx")));
    }

    #[test]
    fn encoder_is_not_a_remover() {
        let session = Session::builder()
            .unwrap()
            .commit_from_file("./models/encoder_Q.onnx")
            .unwrap();
        assert!(matches!(
            validate_model(&session),
            Err(Error::InvalidRemoverModel)
        ));
    }

    #[test]
    fn decoder_is_not_a_remover() {
        let session = Session::builder()
            .unwrap()
            .commit_from_file("./models/decoder_Q.onnx")
            .unwrap();
        assert!(matches!(
            validate_model(&session),
            Err(Error::InvalidRemoverModel)
        ));
    }

    #[test]
    fn residual_rejects_bad_outputs() {
        let input = ndarray::Array4::zeros([1, 3, 2, 2]).into_dyn();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let output = ndarray::Array4::from_elem([1, 3, 2, 2], bad).into_dyn();
            assert!(matches!(
                removal_residual(input.clone(), output, Variant::Q),
                Err(Error::InvalidRemoverModel)
            ));
        }
        let output = ndarray::Array4::zeros([1, 3, 1, 1]).into_dyn();
        assert!(matches!(
            removal_residual(input, output, Variant::Q),
            Err(Error::InvalidRemoverModel)
        ));
    }

    #[test]
    fn residual_clamps_model_output_and_scales_p() {
        for variant in [Variant::B, Variant::C, Variant::P, Variant::Q] {
            let input = ndarray::Array4::from_elem([1, 3, 2, 2], -1.).into_dyn();
            let output = ndarray::Array4::from_elem([1, 3, 2, 2], 2.).into_dyn();
            let residual = removal_residual(input, output, variant).unwrap();
            let expected = if variant == Variant::P { 2.5 } else { 2. };
            assert!(residual.iter().all(|&v| v == expected));
        }
    }
}
