mod data;
mod inference;
mod model;
mod training;

use crate::{model::gpt::SimpleGptConfig, model::ModelConfig, training::TrainingConfig};
use burn::backend::{Autodiff, Wgpu};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffbackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    //  let artifact_dir = "./tmp";

    let my_model = SimpleGptConfig::new(50257, 768).init::<MyAutodiffbackend>(&device);
    println!("{:?}", my_model);
}
