// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{fs::OpenOptions, path::PathBuf};

use clap::{Parser, Subcommand};
use image::{codecs::jpeg::JpegEncoder, ImageFormat};
use rand::{distributions::Standard, prelude::Distribution as _};
use trustmark::{Trustmark, Variant, Version, WatermarkRemover};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    models: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Encode a watermark into an image
    Encode {
        /// The image to encode.
        #[arg(short)]
        input: PathBuf,
        /// The path to save the watermarked image.
        #[arg(short)]
        output: PathBuf,
        /// The watermark to encode. Defaults to random if not specified.
        #[arg(short, long)]
        watermark: Option<String>,
        /// The BCH version to encode with. Defaults to BchSuper.
        #[arg(long)]
        version: Option<Version>,
        /// The model variant to encode with.
        #[arg(long)]
        variant: Option<Variant>,
        /// If the requested output is JPEG, the quality to use for encoding.
        #[arg(long)]
        quality: Option<u8>,
    },
    /// Decode a watermark from an image
    Decode {
        #[arg(short)]
        input: PathBuf,
        /// The model variant to decode with.
        #[arg(long)]
        variant: Option<Variant>,
    },
    /// Remove a supported TrustMark watermark (JPG/PNG only; lossy)
    Remove {
        #[arg(short)]
        input: PathBuf,
        #[arg(short)]
        output: PathBuf,
        /// The model variant used to embed the watermark (default: Q).
        #[arg(long)]
        variant: Option<Variant>,
        /// JPEG output quality (default: 90).
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100))]
        quality: Option<u8>,
    },
}

impl Command {
    /// Extract the version to use from this `Command`.
    fn get_version(&self) -> Version {
        match self {
            Command::Encode {
                version: Some(version),
                ..
            } => *version,
            _ => Version::Bch5,
        }
    }

    /// Extract the variant to use from this `Command`.
    fn get_variant(&self) -> Variant {
        match self {
            Command::Encode {
                variant: Some(variant),
                ..
            } => *variant,
            Command::Decode {
                variant: Some(variant),
                ..
            }
            | Command::Remove {
                variant: Some(variant),
                ..
            } => *variant,
            _ => Variant::Q,
        }
    }
}

/// Generate a random watermark with as many bits as specified by `bits`.
///
/// # Example
///
/// ```rust
/// # fn main() {
/// println!("{}", gen_watermark(4)); // 1010
/// # }
/// ```
fn gen_watermark(bits: usize) -> String {
    let mut rng = rand::thread_rng();
    let v: Vec<bool> = Standard.sample_iter(&mut rng).take(bits).collect();
    v.into_iter()
        .map(|bit| if bit { '1' } else { '0' })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if let Command::Remove {
        input,
        output,
        quality,
        ..
    } = &args.command
    {
        return remove_image(
            &args.models,
            args.command.get_variant(),
            input,
            output,
            *quality,
        );
    }
    let tm = Trustmark::new(
        &args.models,
        args.command.get_variant(),
        args.command.get_version(),
    )
    .unwrap();
    match args.command {
        Command::Encode {
            input,
            output,
            watermark,
            version,
            quality,
            ..
        } => {
            let input = image::open(input).unwrap();
            let watermark = watermark.unwrap_or_else(|| {
                gen_watermark(version.unwrap_or(Version::Bch5).data_bits().into())
            });
            let encoded = tm.encode(watermark.clone(), input, 0.95).unwrap();

            let format = ImageFormat::from_path(&output).unwrap();
            match format {
                // JPEG encoding can make visual artifacts worse, so we encode with a higher
                // quality than the default (or the quality requested by the user).
                ImageFormat::Jpeg => {
                    let quality = quality.unwrap_or(90);
                    let mut writer = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&output)
                        .unwrap();
                    let encoder = JpegEncoder::new_with_quality(&mut writer, quality);
                    encoded.to_rgb8().write_with_encoder(encoder).unwrap();
                }
                _ => {
                    encoded.to_rgba8().save(&output).unwrap();
                }
            }
        }
        Command::Decode { input, .. } => {
            let input = image::open(input).unwrap();
            match tm.decode(input) {
                Ok(decoded) => println!("Found watermark: {decoded}"),
                Err(trustmark::Error::CorruptWatermark) => {
                    println!("Corrupt or missing watermark")
                }
                err => panic!("{err:?}"),
            }
        }
        Command::Remove { .. } => {
            unreachable!("removal was dispatched before loading encode/decode models")
        }
    }
    Ok(())
}

fn remove_image(
    models: &std::path::Path,
    variant: Variant,
    input: &std::path::Path,
    output: &std::path::Path,
    quality: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = image::ImageReader::open(input)?.with_guessed_format()?;
    if !matches!(reader.format(), Some(ImageFormat::Jpeg | ImageFormat::Png)) {
        return Err("watermark removal supports only JPG and PNG input".into());
    }
    let format = ImageFormat::from_path(output)?;
    if !matches!(format, ImageFormat::Jpeg | ImageFormat::Png) {
        return Err("watermark removal supports only JPG and PNG output".into());
    }
    if quality.is_some() && format != ImageFormat::Jpeg {
        return Err("--quality is supported only for JPEG output".into());
    }
    let input = reader.decode()?;
    // No encoder or decoder is needed for removal.
    let remover = WatermarkRemover::new(models, variant)?;
    let recovered = remover.remove(input)?;
    save_removed_image(&recovered, output, format, quality)
}

fn save_removed_image(
    recovered: &image::DynamicImage,
    output: &std::path::Path,
    format: ImageFormat,
    quality: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    // Encode beside the destination, then atomically replace it. A failure
    // leaves the old output intact and removes the temporary file on drop.
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    if format == ImageFormat::Jpeg {
        recovered
            .to_rgb8()
            .write_with_encoder(JpegEncoder::new_with_quality(
                file.as_file_mut(),
                quality.unwrap_or(90),
            ))?;
    } else {
        image::DynamicImage::ImageRgba8(recovered.to_rgba8())
            .write_to(file.as_file_mut(), format)?;
    }
    file.as_file().sync_all()?;
    file.persist(output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::GenericImageView as _;

    use super::*;

    #[test]
    fn removal_cli_selects_variant() {
        let args = Args::try_parse_from([
            "trustmark-cli",
            "-m",
            "models",
            "remove",
            "-i",
            "in.png",
            "-o",
            "out.png",
            "--variant",
            "P",
        ])
        .unwrap();
        assert_eq!(args.command.get_variant(), Variant::P);
        assert!(matches!(args.command, Command::Remove { .. }));
    }

    #[test]
    fn removal_rejects_invalid_quality() {
        assert!(Args::try_parse_from([
            "trustmark-cli",
            "-m",
            "models",
            "remove",
            "-i",
            "in.png",
            "-o",
            "out.jpg",
            "--quality",
            "101"
        ])
        .is_err());
    }

    #[test]
    fn removal_rejects_unsupported_output_before_loading_models() {
        let error = remove_image(
            std::path::Path::new("./models/missing"),
            Variant::Q,
            std::path::Path::new("../../../images/ghost.png"),
            std::path::Path::new("out.bmp"),
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "watermark removal supports only JPG and PNG output"
        );
    }

    #[test]
    fn removal_rejects_unsupported_input_before_loading_models() {
        let directory = tempfile::tempdir().unwrap();
        // The extension deliberately says PNG: rejection must use the detected
        // content format, not just the filename.
        let input = directory.path().join("input.png");
        image::RgbImage::new(1, 1)
            .save_with_format(&input, ImageFormat::Bmp)
            .unwrap();
        let output = directory.path().join("output.png");
        let error = remove_image(
            &directory.path().join("missing-models"),
            Variant::Q,
            &input,
            &output,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "watermark removal supports only JPG and PNG input"
        );
        assert!(!output.exists());
    }

    #[test]
    fn removal_rejects_jpeg_quality_for_png_before_loading_models() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        image::RgbImage::new(2, 3).save(&input).unwrap();
        let output = directory.path().join("output.png");
        let error = remove_image(
            &directory.path().join("missing-models"),
            Variant::Q,
            &input,
            &output,
            Some(95),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "--quality is supported only for JPEG output"
        );
        assert!(!output.exists());
    }

    fn test_image() -> image::DynamicImage {
        image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                ((7 * x + 11 * y) % 256) as u8,
                ((13 * x + 3 * y) % 256) as u8,
                ((5 * x + 17 * y) % 256) as u8,
            ])
        })
        .into()
    }

    #[test]
    fn output_encoding_failure_preserves_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("existing.png");
        std::fs::write(&output, b"original output").unwrap();
        // An 8-bit image cannot be encoded as OpenEXR, forcing an encoder error
        // after the temporary file has been created.
        assert!(save_removed_image(&test_image(), &output, ImageFormat::OpenExr, None).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"original output");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn jpeg_output_honors_quality_and_default() {
        let directory = tempfile::tempdir().unwrap();
        let low = directory.path().join("low.jpg");
        let high = directory.path().join("high.jpg");
        let default = directory.path().join("default.jpg");
        let explicit_default = directory.path().join("explicit-default.jpg");
        let original = test_image();
        save_removed_image(&original, &low, ImageFormat::Jpeg, Some(10)).unwrap();
        save_removed_image(&original, &high, ImageFormat::Jpeg, Some(95)).unwrap();
        save_removed_image(&original, &default, ImageFormat::Jpeg, None).unwrap();
        save_removed_image(&original, &explicit_default, ImageFormat::Jpeg, Some(90)).unwrap();
        assert_eq!(
            std::fs::read(default).unwrap(),
            std::fs::read(explicit_default).unwrap()
        );
        let original = original.to_rgb8();
        let error = |path: &std::path::Path| {
            let decoded = image::open(path).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), original.dimensions());
            original
                .as_raw()
                .iter()
                .zip(decoded.as_raw())
                .map(|(&a, &b)| (i64::from(a) - i64::from(b)).pow(2) as u64)
                .sum::<u64>()
        };
        assert!(error(&high) < error(&low));
        assert!(std::fs::metadata(high).unwrap().len() > std::fs::metadata(low).unwrap().len());
    }

    #[test]
    #[ignore = "requires the Q remover model; run cargo xtask fetch-removers"]
    fn removal_writes_jpeg_end_to_end() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.png");
        test_image().save(&input).unwrap();
        let output = directory.path().join("removed.jpg");
        let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        remove_image(&models, Variant::Q, &input, &output, Some(95)).unwrap();
        let reader = image::ImageReader::open(&output)
            .unwrap()
            .with_guessed_format()
            .unwrap();
        assert_eq!(reader.format(), Some(ImageFormat::Jpeg));
        assert_eq!(reader.decode().unwrap().dimensions(), (32, 32));
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 2);
    }
}
