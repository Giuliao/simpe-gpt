use burn::{
    module::Module,
    nn::{Dropout, Embedding, LayerNorm, Linear},
    prelude::*,
};

use super::transformer::{TransformerBlock, TransformerBlockConfig};

#[derive(Module, Debug)]
pub struct SimpleGptModel<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    drop_emb: Dropout,
    trf_blocks: Vec<TransformerBlock<B>>,
    final_norm: LayerNorm<B>,
    out_head: Linear<B>,
}

impl<B: Backend> SimpleGptModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.shape().dims();
        let tok_embeds = self.tok_emb.forward(input.clone());
        let pos_indices = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &input.device());

        let pos_embeds = self
            .pos_emb
            .forward(pos_indices.unsqueeze().repeat(&[batch_size, 0]));

        let mut x = tok_embeds + pos_embeds;
        x = self.drop_emb.forward(x);
        for block in &self.trf_blocks {
            x = block.forward(x);
        }
        x = self.final_norm.forward(x);
        self.out_head.forward(x)
    }
}

#[derive(Config)]
pub struct SimpleGptConfig {
    pub vocab_size: usize,
    pub emb_dim: usize,
    #[config(default = 256)]
    pub context_len: usize,
    #[config(default = 0.1)]
    pub drop_rate: f64,
    #[config(default = 12)]
    pub n_layers: usize,
    #[config(default = 12)]
    pub n_heads: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl SimpleGptConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleGptModel<B> {
        let mut trf_blocks = Vec::new();
        for _ in 0..self.n_layers {
            trf_blocks.push(
                TransformerBlockConfig::new(
                    self.emb_dim,
                    self.emb_dim,
                    self.context_len,
                    self.n_heads,
                    self.drop_rate,
                    self.bias,
                )
                .init(device),
            );
        }

        SimpleGptModel {
            tok_emb: nn::EmbeddingConfig::new(self.vocab_size, self.emb_dim).init(device),
            pos_emb: nn::EmbeddingConfig::new(self.context_len, self.emb_dim).init(device),
            drop_emb: nn::DropoutConfig::new(self.drop_rate).init(),
            trf_blocks: trf_blocks,
            final_norm: nn::LayerNormConfig::new(self.emb_dim).init(device),
            out_head: nn::LinearConfig::new(self.emb_dim, self.vocab_size)
                .with_bias(self.bias)
                .init(device),
        }
    }
}
