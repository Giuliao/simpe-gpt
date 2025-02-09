use burn::{
    module::Module,
    nn::{Gelu, Linear, LinearConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activiation: Gelu,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.input_layer.forward(input);
        let x = self.activiation.forward(x);
        self.output_layer.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    hidden_size: usize,
    out_size: usize,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            input_layer: LinearConfig::new(self.hidden_size, self.out_size).init(device),
            output_layer: LinearConfig::new(self.hidden_size, self.out_size).init(device),
            activiation: Gelu::new(),
        }
    }
}
