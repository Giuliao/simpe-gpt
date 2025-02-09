use burn::{
    module::Module,
    nn::{Dropout, LayerNorm},
    prelude::*,
};
use nn::{DropoutConfig, LayerNormConfig};

use super::feedforward::{FeedForward, FeedForwardConfig};
use super::mha::{MultiHeadAttention, MultiHeadAttentionConfig};

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    att: MultiHeadAttention<B>,
    ff: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    drop_shortcut: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x);
        let x = self.att.forward(x);
        let x = self.drop_shortcut.forward(x);
        let x = x + shortcut;

        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.ff.forward(x);
        let x = self.drop_shortcut.forward(x);
        x + shortcut
    }
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    d_in: usize,
    d_out: usize,
    context_len: usize,
    num_heads: usize,
    dropout: f64,
    bias: bool,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            att: MultiHeadAttentionConfig::new(
                self.d_in,
                self.d_out,
                self.context_len,
                self.dropout,
                self.num_heads,
                self.bias,
            )
            .init(device),
            ff: FeedForwardConfig::new(self.d_in, self.d_out).init(device),
            norm1: LayerNormConfig::new(self.d_in).init(device),
            norm2: LayerNormConfig::new(self.d_in).init(device),
            drop_shortcut: DropoutConfig::new(self.dropout).init(),
        }
    }
}
