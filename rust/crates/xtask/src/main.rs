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
    let path = std::path::Path::new("models").join(filename);
    fetch_model_from_url(&model_url, &path, MAX_MODEL_BYTES)
}

fn fetch_model_from_url(
    model_url: &str,
    path: &std::path::Path,
    limit: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = ureq::get(model_url)
        .set("Accept-Encoding", "identity")
        .call()?;
    let expected_size = response
        .header("Content-Length")
        .map(str::parse::<u64>)
        .transpose()?;
    if expected_size.is_some_and(|size| size > limit) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("model exceeds download size limit: {model_url}"),
        )
        .into());
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let result = copy_model(response.into_reader(), &mut file, expected_size, limit);
    drop(file);
    if let Err(error) = result {
        // Only remove the partial file created by this invocation, never an
        // existing asset (create_new above refuses to overwrite one).
        return Err(cleanup_failed_download(path, model_url, error).into());
    }
    Ok(())
}

fn cleanup_failed_download(path: &std::path::Path, model_url: &str, error: io::Error) -> io::Error {
    let mut message = format!("download failed for {model_url}: {error}");
    if let Err(cleanup_error) = fs::remove_file(path) {
        message.push_str(&format!(
            "; could not remove partial file {}: {cleanup_error}",
            path.display()
        ));
    }
    io::Error::new(error.kind(), message)
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
    use std::{io::BufRead, net::TcpListener, time::Duration};

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

    fn serve(response: &str) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
        let response = response.to_owned();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            {
                let mut reader = io::BufReader::new(&mut stream);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                }
            }
            // Malformed/truncated response tests may make the client close early.
            let _ = stream.write_all(response.as_bytes());
        });
        (url, server)
    }

    #[test]
    fn http_download_writes_complete_model_and_refuses_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        let response = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nmodel";
        let (url, server) = serve(response);
        fetch_model_from_url(&url, &path, 10).unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"model");
        let (url, server) = serve(response);
        assert!(fetch_model_from_url(&url, &path, 10).is_err());
        server.join().unwrap();
        assert_eq!(fs::read(path).unwrap(), b"model");
    }

    #[test]
    fn http_bad_responses_leave_no_partial_model() {
        for response in [
            "HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nshort",
            "HTTP/1.1 200 OK\r\nContent-Length: invalid\r\nConnection: close\r\n\r\nmodel",
            "HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n",
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nlong-model-data",
            "HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("model.onnx");
            let (url, server) = serve(response);
            assert!(fetch_model_from_url(&url, &path, 10).is_err());
            server.join().unwrap();
            assert!(!path.exists(), "partial file left after {response:?}");
        }
    }

    #[test]
    fn cleanup_failure_retains_original_error_and_context() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("already-missing.onnx");
        let original = io::Error::new(io::ErrorKind::UnexpectedEof, "original truncated download");
        let error = cleanup_failed_download(&path, "https://example.test/model.onnx", original);
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        let message = error.to_string();
        assert!(message.contains("original truncated download"));
        assert!(message.contains("https://example.test/model.onnx"));
        assert!(message.contains("could not remove partial file"));
    }
}
