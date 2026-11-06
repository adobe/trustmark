// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
};

use clap::Parser;

// Published remover models are approximately 155 MB each.
const MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Parser)]
enum Args {
    FetchModels,
    /// Fetch the published P and Q watermark-removal ONNX models.
    FetchRemovers,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    fs::create_dir_all("models")?;
    match args {
        Args::FetchModels => fetch_models()?,
        Args::FetchRemovers => fetch_removers()?,
    }
    Ok(())
}

/// Fetch the encoder and decoder models.
fn fetch_models() -> Result<(), Box<dyn std::error::Error>> {
    for filename in [
        "decoder_Q.onnx",
        "encoder_Q.onnx",
        "decoder_P.onnx",
        "encoder_P.onnx",
        "decoder_B.onnx",
        "encoder_B.onnx",
        "decoder_C.onnx",
        "encoder_C.onnx",
    ] {
        fetch_model(filename)?;
    }
    Ok(())
}

fn fetch_removers() -> Result<(), Box<dyn std::error::Error>> {
    for filename in ["remover_P.onnx", "remover_Q.onnx"] {
        fetch_model(filename)?;
    }
    Ok(())
}

/// Fetch a single model identified by `filename`.
///
/// Models are fetched from a hardcoded CDN URL.
fn fetch_model(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = "https://cai-watermark.adobe.net/watermarking/trustmark-models";
    let model_url = format!("{root}/{filename}",);
    let response = ureq::get(&model_url)
        .set("Accept-Encoding", "identity")
        .call()?;
    let expected_size = response
        .header("Content-Length")
        .map(str::parse::<u64>)
        .transpose()?;
    if expected_size.is_some_and(|size| size > MAX_MODEL_BYTES) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("model exceeds download size limit: {model_url}"),
        )
        .into());
    }
    let path = std::path::Path::new("models").join(filename);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let result = copy_model(
        response.into_reader(),
        &mut file,
        expected_size,
        MAX_MODEL_BYTES,
    );
    drop(file);
    if let Err(error) = result {
        // Only remove the partial file created by this invocation, never an
        // existing asset (create_new above refuses to overwrite one).
        fs::remove_file(&path)?;
        return Err(io::Error::new(
            error.kind(),
            format!("download failed for {model_url}: {error}"),
        )
        .into());
    }
    Ok(())
}

fn copy_model(
    reader: impl Read,
    mut writer: impl Write,
    expected_size: Option<u64>,
    limit: u64,
) -> io::Result<()> {
    let copied = io::copy(&mut reader.take(limit + 1), &mut writer)?;
    if copied > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "model exceeds download size limit",
        ));
    }
    if expected_size.is_some_and(|size| size != copied) {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "model download size does not match Content-Length",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_remover_download_command() {
        assert!(matches!(
            Args::try_parse_from(["xtask", "fetch-removers"]).unwrap(),
            Args::FetchRemovers
        ));
        assert!(matches!(
            Args::try_parse_from(["xtask", "fetch-models"]).unwrap(),
            Args::FetchModels
        ));
    }

    #[test]
    fn supports_published_remover_sizes() {
        // Stream the actual published size to a sink, without allocating a
        // model-sized buffer. A regression to the old 100 MB cap fails here.
        let size = 154_859_578;
        copy_model(
            io::repeat(0).take(size),
            io::sink(),
            Some(size),
            MAX_MODEL_BYTES,
        )
        .unwrap();
    }

    #[test]
    fn copies_complete_model() {
        let mut output = Vec::new();
        copy_model(&b"model"[..], &mut output, Some(5), 5).unwrap();
        assert_eq!(output, b"model");
    }

    #[test]
    fn rejects_truncated_model() {
        let error = copy_model(&b"model"[..], Vec::new(), Some(6), 10).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn rejects_oversized_model_without_content_length() {
        let error = copy_model(&b"model"[..], Vec::new(), None, 4).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }
}
