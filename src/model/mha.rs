use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{activation::softmax, Tensor},
};

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    w_query: Linear<B>,
    w_key: Linear<B>,
    w_value: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
    mask: Tensor<B, 2>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, num_tokens] = x.shape().dims();

        let keys =
            self.w_key
                .forward(x.clone())
                .reshape([b, num_tokens, self.num_heads, self.head_dim]);
        let queries =
            self.w_query
                .forward(x.clone())
                .reshape([b, num_tokens, self.num_heads, self.head_dim]);
        let values =
            self.w_value
                .forward(x)
                .reshape([b, num_tokens, self.num_heads, self.head_dim]);

        let keys = keys.swap_dims(1, 2);
        let queries = queries.swap_dims(1, 2);
        let values = values.swap_dims(1, 2);

        let mut attn_scores = queries.matmul(keys.swap_dims(2, 3));
        attn_scores = attn_scores.mask_fill(
            self.mask
                .clone()
                .bool()
                .slice([0..num_tokens, 0..num_tokens])
                .unsqueeze()
                .repeat(&[b, 0]),
            f64::NEG_INFINITY,
        );

        let mut attn_weights = softmax(attn_scores.div_scalar((self.head_dim as f64).sqrt()), 3);
        attn_weights = self.dropout.forward(attn_weights);
        let context_vec = attn_weights.matmul(values).swap_dims(1, 2).reshape([
            b,
            num_tokens,
            self.num_heads * self.head_dim,
        ]);

        self.out_proj.forward(context_vec)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    d_in: usize,
    d_out: usize,
    context_length: usize,
    dropout: f64,
    num_heads: usize,
    qkv_bias: bool,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(
            self.d_out % self.num_heads == 0,
            "d_out must be divisible by num_heads"
        );

        MultiHeadAttention {
            num_heads: self.num_heads,
            head_dim: self.d_out / self.num_heads,
            w_query: LinearConfig::new(self.d_in, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            w_key: LinearConfig::new(self.d_in, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            w_value: LinearConfig::new(self.d_in, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            out_proj: LinearConfig::new(self.d_out, self.d_out).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            mask: Tensor::ones(
                Shape::new([self.context_length, self.context_length]),
                device,
            )
            .triu(1),
        }
    }
}
