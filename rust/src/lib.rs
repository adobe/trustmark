// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

//! # Trustmark
//!
//! An implementation of TrustMark watermarking for the Content Authenticity Initiative (CAI) in
//! Rust, as described in:
//!
//! ---
//!
//! **TrustMark - Universal Watermarking for Arbitrary Resolution Images**
//!
//! <https://arxiv.org/abs/2311.18297>
//!
//! [Tu Bui]<sup>1</sup>, [Shruti Agarwal]<sup>2</sup>, [John Collomosse]<sup>1,2</sup>
//!
//! <sup>1</sup>DECaDE Centre for the Decentralized Digital Economy, University of Surrey, UK.\
//! <sup>2</sup>Adobe Research, San Jose CA.
//!
//! ---
//!
//! This is a re-implementation of the [trustmark] Python library.
//!
//! [Tu Bui]: https://www.surrey.ac.uk/people/tu-bui
//! [Shruti Agarwal]: https://research.adobe.com/person/shruti-agarwal/
//! [John Collomosse]: https://www.collomosse.com/
//! [trustmark]: https://pypi.org/project/trustmark/
//!
//! ## Example
//!
//! ```rust
//! use trustmark::{Trustmark, Version, Variant};
//!
//! # fn main() {
//! let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
//! let input = image::open("../images/ghost.png").unwrap();
//! let output = tm.encode("0010101".to_owned(), input, 0.95);
//! # }
//! ```
use std::path::Path;

use image::{DynamicImage, GenericImageView as _};
use ort::{GraphOptimizationLevel, Session};

use self::{bits::Bits, image_processing::ModelImage};

mod bits;
mod image_processing;
mod model;

/// A loaded Trustmark model.
pub struct Trustmark {
    encoder: Session,
    decoder: Session,
    version: Version,
    variant: Variant,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("watermark is corrupt or missing")]
    CorruptWatermark,
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),
    #[error("image processing error: {0}")]
    ImageProcessing(#[from] image_processing::Error),
    #[error("bits processing error: {0}")]
    Bits(bits::Error),
    #[error("invalid model variant")]
    InvalidModelVariant,
}

impl From<bits::Error> for Error {
    fn from(value: bits::Error) -> Self {
        match value {
            bits::Error::CorruptWatermark => Error::CorruptWatermark,
            err => Error::Bits(err),
        }
    }
}

pub use bits::Version;
pub use model::Variant;

impl Trustmark {
    /// Load a Trustmark model.
    pub fn new<P: AsRef<Path>>(
        models: P,
        variant: Variant,
        version: Version,
    ) -> Result<Self, Error> {
        let encoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(models.as_ref().join(variant.encoder_filename()))?;
        let decoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .commit_from_file(models.as_ref().join(variant.decoder_filename()))?;
        Ok(Self {
            encoder,
            decoder,
            version,
            variant,
        })
    }

    /// Encode a watermark into an image.
    ///
    /// `watermark` is a bitstring encoding the watermark identifier to encode. `img` is the image
    /// which will be watermarked. `strength` is a number between 0 and 1 indicating how strong the
    /// resulting watermark should be. 0.95 is a normal strength.
    pub fn encode(
        &self,
        watermark: String,
        img: DynamicImage,
        strength: f32,
    ) -> Result<DynamicImage, Error> {
        let (original_width, original_height) = img.dimensions();
        let aspect_ratio = original_width as f32 / original_height as f32;

        // the image is always encoded with size 256x256
        let encode_size = 256;

        // Debug: dump image tensor and bits before running encoder
        {
            let dbg_img: ort::Value<ort::TensorValueType<f32>> =
                ModelImage(encode_size, self.variant, img.clone()).try_into()?;
            let arr = dbg_img.try_extract_tensor::<f32>()?.to_owned();
            let slice = arr.as_slice().unwrap();
            eprintln!("RUST ENC DEBUG: image tensor NCHW shape: {:?}", arr.shape());
            eprintln!(
                "RUST ENC DEBUG: image tensor first 16 vals: {}",
                slice.iter().take(16).map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
            );
        }
        {
            let bits_struct = Bits::apply_error_correction_and_schema(watermark.clone(), self.version)?;
            eprintln!("RUST ENC DEBUG: full bits: {}", bits_struct.clone().into_string());
            let dbg_bits: ort::Value<ort::TensorValueType<f32>> = bits_struct.into();
            let arr = dbg_bits.try_extract_tensor::<f32>()?.to_owned();
            let slice = arr.as_slice().unwrap();
            eprintln!("RUST ENC DEBUG: bits tensor shape: {:?}", arr.shape());
            eprintln!(
                "RUST ENC DEBUG: bits tensor first 32 vals: {}",
                slice.iter().take(32).map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
            );
        }

        // Build inputs for encoder
        let input_img: ort::Value<ort::TensorValueType<f32>> =
            ModelImage(encode_size, self.variant, img.clone()).try_into()?;
        let bits: ort::Value<ort::TensorValueType<f32>> =
            Bits::apply_error_correction_and_schema(watermark, self.version)?.into();
        let outputs = self.encoder.run(ort::inputs![
            "onnx::Concat_0" => input_img,
            "onnx::Gemm_1" => bits,
        ]?)?;
        let output_img = outputs["image"].try_extract_tensor::<f32>()?.to_owned();

        // Need to calculate and apply the residual.
        let input_img: ort::Value<ort::TensorValueType<f32>> =
            ModelImage(encode_size, self.variant, img.clone()).try_into()?;
        let residual = (self.variant.strength_multiplier() * strength)
            * (output_img - input_img.try_extract_tensor::<f32>()?);

        // Residual should be small perturbations.
        let mut residual = residual.clamp(-0.2, 0.2);
        if (self.variant == Variant::Q && !(0.5..=2.0).contains(&aspect_ratio))
            || self.variant == Variant::P
        {
            residual = image_processing::remove_boundary_artifact(
                residual,
                (original_width as usize, original_height as usize),
                self.variant,
            );
        }

        let ModelImage(_, _, residual) = (encode_size, self.variant, residual).try_into()?;

        Ok(image_processing::apply_residual(img, residual))
    }

    /// Decode a watermark from an image.
    pub fn decode(&self, img: DynamicImage) -> Result<String, Error> {
        // P variant has a smaller decode size
        let decode_size = if self.variant == Variant::P { 224 } else { 256 };
        // Debug: dump decoder input tensor
        {
            let dbg: ort::Value<ort::TensorValueType<f32>> =
                ModelImage(decode_size, self.variant, img.clone()).try_into()?;
            let arr = dbg.try_extract_tensor::<f32>()?.to_owned();
            let shape = arr.shape(); // [1,3,H,W]
            let slice = arr.as_slice().unwrap();
            eprintln!("RUST DEC DEBUG: image tensor NCHW shape: {:?}", shape);
            let hw = (shape[2] * shape[3]) as usize;
            let c0 = &slice[0..hw.min(slice.len())];
            let c1 = &slice[hw..(2*hw).min(slice.len())];
            let c2 = &slice[(2*hw)..(3*hw).min(slice.len())];
            eprintln!(
                "RUST DEC DEBUG: C0 first 16: {}",
                c0.iter().take(16).map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
            );
            eprintln!(
                "RUST DEC DEBUG: C1 first 16: {}",
                c1.iter().take(16).map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
            );
            eprintln!(
                "RUST DEC DEBUG: C2 first 16: {}",
                c2.iter().take(16).map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
            );
        }
        let img: ort::Value<ort::TensorValueType<f32>> =
            ModelImage(decode_size, self.variant, img).try_into()?;
        let outputs = self.decoder.run(ort::inputs![
            "image" => img,
        ]?)?;
        let watermark = outputs["output"].try_extract_tensor::<f32>()?.to_owned();
        let watermark: Bits = watermark.try_into()?;
        Ok(watermark.get_data())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_models() {
        Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
    }

    fn roundtrip(path: impl AsRef<Path>) {
        let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        let input = image::open(path.as_ref()).unwrap();
        let watermark = "1011011110011000111111000000011111011111011100000110110110111".to_owned();
        let encoded = tm.encode(watermark.clone(), input, 0.95).unwrap();
        encoded.to_rgba8().save("./test.png").unwrap();
        let input = image::open("./test.png").unwrap();
        let decoded = tm.decode(input).unwrap();
        assert_eq!(watermark, decoded);
    }

    #[test]
    fn roundtrip_ghost() {
        roundtrip("../images/ghost.png");
    }

    #[test]
    fn roundtrip_ufo() {
        roundtrip("../images/ufo_240.jpg");
    }
}
