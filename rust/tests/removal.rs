// Copyright 2026 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::io::Cursor;

use image::{DynamicImage, GenericImageView as _, ImageFormat};
use trustmark::{Error, Trustmark, Variant, Version, WatermarkRemover};

const ORIGINAL: &str = "1011011110011000111111000000011111011111011100000110110110111";
const REPLACEMENT: &str = "0011011110011000111111000000011111011111011100000110110110111";

fn image_roundtrip(image: &DynamicImage, format: ImageFormat) -> DynamicImage {
    let mut buffer = Cursor::new(Vec::new());
    if format == ImageFormat::Jpeg {
        image
            .to_rgb8()
            .write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut buffer,
                95,
            ))
            .unwrap();
    } else {
        DynamicImage::ImageRgba8(image.to_rgba8())
            .write_to(&mut buffer, format)
            .unwrap();
    }
    image::load_from_memory(buffer.get_ref()).unwrap()
}

fn removal_roundtrip(variant: Variant, path: &str) {
    let tm = Trustmark::new("./models", variant, Version::Bch5).unwrap();
    let remover = WatermarkRemover::new("./models", variant).unwrap();
    let input = image::open(path).unwrap();
    let unwatermarked = remover.remove(input.clone()).unwrap();
    assert_eq!(unwatermarked.dimensions(), input.dimensions());
    let encoded = image_roundtrip(
        &tm.encode(ORIGINAL.to_owned(), input, 0.95).unwrap(),
        ImageFormat::from_path(path).unwrap(),
    );
    assert_eq!(tm.decode(encoded.clone()).unwrap(), ORIGINAL);
    let recovered = image_roundtrip(&remover.remove(encoded.clone()).unwrap(), ImageFormat::Png);
    assert_eq!(recovered.dimensions(), encoded.dimensions());
    match tm.decode(recovered.clone()) {
        // The decoder may produce another payload after removal. The requirement
        // is that the original identifier cannot be recovered, not that every
        // decoded bitstring must fail error correction.
        Ok(decoded) => assert_ne!(
            decoded, ORIGINAL,
            "old watermark still decodes for {variant} / {path}"
        ),
        Err(Error::CorruptWatermark) => {}
        Err(error) => panic!("decode failed for {variant} / {path}: {error}"),
    }
    let replacement = image_roundtrip(
        &tm.encode(REPLACEMENT.to_owned(), recovered, 0.95).unwrap(),
        ImageFormat::Png,
    );
    assert_eq!(tm.decode(replacement).unwrap(), REPLACEMENT);
    if encoded.color().has_alpha() {
        let output = remover.remove(encoded.clone()).unwrap().to_rgba8();
        assert!(encoded
            .to_rgba8()
            .pixels()
            .zip(output.pixels())
            .all(|(a, b)| a[3] == b[3]));
    }
}

macro_rules! removal_test {
    ($name:ident, $variant:ident) => {
        #[test]
        #[ignore = "requires matching remover ONNX models; fetch P/Q with cargo xtask fetch-removers"]
        fn $name() {
            for path in ["../images/ufo_240.jpg", "../images/ghost.png"] {
                removal_roundtrip(Variant::$variant, path);
            }
        }
    };
}

removal_test!(removes_and_rewatermarks_b, B);
removal_test!(removes_and_rewatermarks_c, C);
removal_test!(removes_and_rewatermarks_p, P);
removal_test!(removes_and_rewatermarks_q, Q);

#[test]
#[ignore = "requires the P remover model; run cargo xtask fetch-removers"]
fn shared_remover_preserves_dimensions_and_alpha() {
    let remover = WatermarkRemover::new("./models", Variant::P).unwrap();
    assert!(remover.remove(DynamicImage::new_rgb8(0, 0)).is_err());
    std::thread::scope(|scope| {
        let handles: Vec<_> = [(1, 1), (101, 10), (10, 101), (100, 110)]
            .into_iter()
            .map(|(w, h)| {
                let remover = &remover;
                scope.spawn(move || {
                    let input = image::RgbaImage::from_pixel(w, h, image::Rgba([120, 80, 160, 64]));
                    let output = remover.remove(input.into()).unwrap();
                    assert_eq!(output.dimensions(), (w, h));
                    assert!(output.to_rgba8().pixels().all(|p| p[3] == 64));
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}
