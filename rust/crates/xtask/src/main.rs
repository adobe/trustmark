// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{
    fs::OpenOptions,
    io::{self, Read as _},
};

use clap::Parser;

#[derive(Debug, Parser)]
enum Args {
    FetchModels,
}

fn main() {
    let args = Args::parse();
    match args {
        Args::FetchModels => fetch_models(),
    }
}

/// Fetch all known models.
fn fetch_models() {
    fetch_model("decoder_Q.onnx");
    fetch_model("encoder_Q.onnx");
    fetch_model("decoder_P.onnx");
    fetch_model("encoder_P.onnx");
    fetch_model("decoder_B.onnx");
    fetch_model("encoder_B.onnx");
    fetch_model("decoder_C.onnx");
    fetch_model("encoder_C.onnx");
}

/// Fetch a single model identified by `filename`.
///
/// Models are fetched from a hardcoded CDN URL.
fn fetch_model(filename: &str) {
    let root = "https://cai-watermark.adobe.net/watermarking/trustmark-models";
    let model_url = format!("{root}/{filename}",);
    let mut decoder = ureq::get(&model_url)
        .call()
        .unwrap()
        .into_reader()
        .take(100_000_000);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(format!("models/{filename}"))
        .unwrap();
    io::copy(&mut decoder, &mut file).unwrap();
}
