"""keras_core implementation of [Llama](https://arxiv.org/abs/2302.13971)
Based on [minimal-llama](https://github.com/zphang/minimal-llama/blob/main/minimal_llama/model.py)"""
import math
import os
from typing import Callable, TypedDict

os.environ["KERAS_BACKEND"] = "jax"

import keras_core as keras
from keras_core import ops, Layer, Model

MULTIPLE_OF = 256


def flash_attention(q, k, v, mask):
    backend = os.environ["KERAS_BACKEND"]
    if backend == "jax":
        #from jax.experimental.pallas.ops import attention
        from flash_attention_jax import flash_attention
        #from flash_attention_jax import plain_attention as flash_attention
        return flash_attention(q, k, v, mask)
    else:
        raise NotImplementedError("Pytorch and Tensorflow Flash Attention is not implemented")



class BaseLayerKwargs(TypedDict):
    activity_regularizer: Callable
    trainable: bool
    dtype: str
    autocast: bool
    name: str


class RMSNorm(Layer):
    def __init__(self, eps: float = 1e-6, **layer_kwargs: BaseLayerKwargs):
        super().__init__(**layer_kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(input_shape[-1],), initializer="ones", trainable=True, name="weight"
        )

    def call(self, inputs):
        def norm(x):
            return x * ops.rsqrt(
                ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps
            )

        output = ops.cast(norm(ops.cast(inputs, dtype="float32")), dtype=inputs.dtype)
        return output * self.weight


class FeedForward(Layer):
    def __init__(self, hidden_dim: int, p_dropout=0.0, **layer_kwargs: BaseLayerKwargs):
        super().__init__(**layer_kwargs)
        hidden_dim = int(2 * hidden_dim / 3)
        self.hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) // MULTIPLE_OF)
        self.p_dropout = p_dropout

    def build(self, input_shape):
        self.w1 = keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.w2 = keras.layers.Dense(input_shape[-1], use_bias=False)
        self.w3 = keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.dropout = keras.layers.Dropout(rate=self.p_dropout)

    def call(self, inputs):
        return self.dropout(self.w2(ops.silu(self.w1(inputs)) * self.w3(inputs)))


def apply_rotary_pos_emb(x, cos, sin):
    bsz, seq_len, nh, hd = x.shape
    x_shaped = ops.cast(x, dtype="float32").reshape(bsz, seq_len, nh, -1, 2)
    cos = cos[:seq_len].reshape(1, seq_len, 1, hd // 2)
    sin = sin[:seq_len].reshape(1, seq_len, 1, hd // 2)
    x_1 = x_shaped[..., 0] * cos - x_shaped[..., 1] * sin
    x_2 = x_shaped[..., 1] * cos + x_shaped[..., 0] * sin
    x_out = ops.stack((x_1, x_2), -1)
    x_out = ops.reshape(x_out, (bsz, seq_len, nh, hd))
    x_out = ops.cast(x_out, dtype=x.dtype)
    return x_out

class Attention(Layer):
    def __init__(
        self, n_heads: int, p_dropout: float = 0.0, **layer_kwargs: BaseLayerKwargs
    ):
        super().__init__(**layer_kwargs)
        self.n_heads = n_heads
        self.p_dropout = p_dropout

    def build(self, input_shape):
        head_dim = input_shape[-1] // self.n_heads
        self.head_dim = head_dim
        self.wq = keras.layers.Dense(self.n_heads * head_dim, use_bias=False)
        self.wk = keras.layers.Dense(self.n_heads * head_dim, use_bias=False)
        self.wv = keras.layers.Dense(self.n_heads * head_dim, use_bias=False)
        self.wo = keras.layers.Dense(input_shape[-1], use_bias=False)
        self.attn_dropout = keras.layers.Dropout(rate=self.p_dropout)
        self.resid_dropout = keras.layers.Dropout(rate=self.p_dropout)

    def call(self, inputs, cos, sin, mask, flash=False):
        bsz, seqlen, _ = inputs.shape
        xq, xk, xv = self.wq(inputs), self.wk(inputs), self.wv(inputs)
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, cos, sin), apply_rotary_pos_emb(xk, cos, sin)
        keys = xk[:, :seqlen]
        values = xv[:, :seqlen]

        xq = ops.transpose(xq, axes=(0, 2, 1, 3))
        keys = ops.transpose(keys, axes=(0, 2, 1, 3))
        values = ops.transpose(values, axes=(0, 2, 1, 3))
        if flash:
            # ! flash_attention_jax doesn't support dropout
            mask = mask[0,0, :seqlen, :seqlen]
            output = flash_attention(xq, keys, values, mask)
        else:
            scores = ops.matmul(xq, ops.transpose(keys, axes=(0, 1, 3, 2))) / math.sqrt(
                self.head_dim
            )
            if mask is not None:
                scores = (
                    scores + mask[..., :seqlen, :seqlen]
                )  
            scores = ops.cast(
                ops.softmax(ops.cast(scores, dtype="float32"), axis=-1), dtype=xq.dtype
            )
            scores = self.attn_dropout(scores)
            output = ops.matmul(scores, values)  # (bsz, n_heads, seqlen, head_dim)
        output = ops.transpose(output, axes=(0, 2, 1, 3)).reshape(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(output))


class LlamaBlock(Layer):
    def __init__(
        self,
        n_heads: int,
        norm_eps: float = 1e-6,
        p_dropout: float = 0.1,
        **layer_kwargs: BaseLayerKwargs,
    ):
        super().__init__(**layer_kwargs)
        self.n_heads = n_heads
        self.norm_eps = norm_eps
        self.p_dropout = p_dropout

    def build(self, input_shape):
        # self.attention = keras.layers.MultiHeadAttention(num_heads=self.n_heads, key_dim=input_shape[-1]//self.n_heads)
        self.attention = Attention(n_heads=self.n_heads, p_dropout=self.p_dropout)
        self.feed_forward = FeedForward(
            hidden_dim=4 * input_shape[-1], p_dropout=self.p_dropout
        )
        self.attention_norm = RMSNorm(eps=self.norm_eps)
        self.ffn_norm = RMSNorm(eps=self.norm_eps)

    def call(self, inputs, cos, sin, mask):
        # q=k=v=self.attention_norm(inputs)
        # h = inputs + self.attention(q,k,v,use_causal_mask=True)
        h = inputs + self.attention(inputs, cos, sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


def precompute_cos_sin(seq_len: int, dim: int, dtype, base: int = 10_000):
    theta = 1.0 / (base ** (ops.arange(0, dim, 2, dtype=dtype) / dim))
    seq_idx = ops.arange(seq_len, dtype=dtype)
    idx_theta = ops.cast(ops.outer(seq_idx, theta), dtype="float32")
    return ops.cos(idx_theta), ops.sin(idx_theta)


def get_functional_llama(
    batch_size, seq_len, vocab_size, n_layers, dim, n_heads, norm_eps
):
    inputs = keras.layers.Input(shape=(None,))  # set seq len at runtime, mostly for generation
    h = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=dim, name="token_embeddings"
    )(inputs)
    cos, sin = precompute_cos_sin(seq_len, dim // n_heads, dtype="float32")
    mask = ops.full((1, 1, seq_len, seq_len), fill_value=-1e10)
    mask = ops.triu(mask, k=1)
    blocks = [LlamaBlock(n_heads=n_heads, norm_eps=norm_eps) for _ in range(n_layers)]
    for idx, block in enumerate(blocks):
        h = block(h, cos[:seq_len], sin[:seq_len], mask)
    h = RMSNorm(eps=norm_eps)(h)
    outputs = keras.layers.Dense(vocab_size, use_bias=False, name="lm_head")(h)
    return keras.Model(inputs=inputs, outputs=outputs, name="llama")


if __name__ == "__main__":
    import os.path
    import numpy as np
    import wandb
    from keras_core.utils import plot_model, PyDataset, set_random_seed
    from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint

    set_random_seed(1)

    def profile_jax_model(model):
        import jax
        data = [(ops.ones((16,128)), ops.ones((16,128)))]
        model._eager_build(data[0])
        trainable_variables = [v.value for v in model.trainable_variables]
        non_trainable_variables = [
            v.value for v in model.non_trainable_variables
        ]
        optimizer_variables = [v.value for v in model.optimizer.variables]
        metrics_variables = [v.value for v in model.metrics_variables]
        state = (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
                )
        model.make_train_function()
        model.train_function(state, data)
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            # Run the operations to be profiled
            for _ in range(100):
                model.train_function(state, data)


    with open("tinyshakespeare.txt", "r", encoding="utf-8") as input_f:
        text = input_f.read()
    vocab = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}

    def encode(text: str):
        return [char_to_idx[ch] for ch in text]

    def decode(ids):
        return "".join(vocab[i] for i in ids)

    text = ops.array(encode(text), dtype="int8")

    def generate_dataset():
        def get_batch(text, seq_len, i):
            x = text[i : i + seq_len]
            y = text[i + 1 : i + seq_len + 1]
            return x, y

        if os.path.exists("batches.npy"):
            batches = np.load("batches.npy")
        else:
            batches = ops.array([get_batch(text, 128, i) for i in range(100_000)])
            np.save("batches.npy", batches)
        return batches

    dataset = generate_dataset()

    class TextEval(keras.callbacks.Callback):
        def __init__(self, dataset):
            self.dataset = dataset
            self.table = wandb.Table(columns=["batch", "generation"])

        def generate(self, prompt):
            tokens = prompt[None, :]
            print()
            progbar = keras.utils.Progbar(30, unit_name="token")
            current_step = 0
            for _ in range(30):
                output = self.model(tokens)
                progbar.update(current_step)
                current_step += 1
                new_token = ops.argmax(output[:, -1, :])
                tokens = ops.append(tokens, new_token)[None, :]
            return tokens

        def on_train_batch_end(self, batch, logs=None):
            if not batch % 1000:
                prompt = self.dataset[-34:]
                generation = self.generate(prompt=prompt)
                print()
                generation = decode(generation[0])
                print(generation)
                generation = generation.replace("\n", "<br>")
                self.table.add_data(
                    batch,
                    wandb.Html(
                        data=f"<strong>{generation[:34]}</strong>{generation[34:]}"
                    ),
                )

        def on_train_end(self, logs=None):
            wandb.run.log({"predictions": self.table})

    config = dict(
        seq_len=128,
        n_layers=8,
        dim=512,
        n_heads=8,
        norm_eps=1e-7,
        batch_size=16,
    )
    config["vocab_size"] = len(vocab)

    wandb.init(
        mode="disabled",
        project="my-gpt",
        config=config,
        name="base",
        notes="Baseline",
        tags=["baseline"],
        anonymous="allow",
        dir="logs",
    )

    model = get_functional_llama(
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        n_layers=config["n_layers"],
        dim=config["dim"],
        n_heads=config["n_heads"],
        norm_eps=config["norm_eps"],
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.build((config["batch_size"], config["seq_len"]))
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)

    xs, ys = ops.split(dataset, 2, axis=1)
    xs = xs[:, 0, :]
    ys = ys[:, 0, :]
    train_xs = xs[: int(xs.shape[0] * 0.8)]
    train_ys = ys[: int(ys.shape[0] * 0.8)]
    val_xs = xs[int(xs.shape[0] * 0.8) :]
    val_ys = ys[int(ys.shape[0] * 0.8) :]

    #profile_jax_model(model)

    model.fit(
        x=xs,
        y=ys,
        batch_size=config["batch_size"],
        epochs=1,
        shuffle=True,
        verbose=1,
        callbacks=[
            WandbCallback(
                log_batch_frequency=10,
                labels=vocab,
                save_graph=False,
                save_model=False,
            ),
            WandbMetricsLogger(),
            keras.callbacks.ModelCheckpoint(
                filepath="logs/llama-{epoch:02d}-{val_loss:.2f}.keras"
            ),
            TextEval(text),
        ],
        validation_data=(val_xs, val_ys),
    )
