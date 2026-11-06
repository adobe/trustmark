// Copyright 2026 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use criterion::{criterion_group, criterion_main, Criterion};
use trustmark::{Variant, WatermarkRemover};

fn remove_main(c: &mut Criterion) {
    let remover = WatermarkRemover::new("./models", Variant::Q).unwrap();
    let input = image::open("../images/ufo_240.jpg").unwrap();
    c.bench_function("remove", |b| {
        b.iter(|| remover.remove(input.clone()).unwrap())
    });
}

criterion_group!(benches, remove_main);
criterion_main!(benches);
