#![feature(test)]

extern crate test;

use candle_core::{Device, Tensor};

use test::Bencher;

#[bench]
fn candle_core(b: &mut Bencher) {
    let device = Device::Cpu;
    b.iter(|| {
        let a = Tensor::randn(0f32, 1., (2, 3), &device).unwrap();
        let b = Tensor::randn(0f32, 1., (3, 4), &device).unwrap();
        a.add(&b)
    })
}
