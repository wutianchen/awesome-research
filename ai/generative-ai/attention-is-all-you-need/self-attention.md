# Self-attention

**Self-attention** is how a transformer lets every token in a sequence look at every
other token and pull in the information it needs. The output for each token is a
context-aware blend of the whole sequence. It was introduced in
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Self-attention is the **mechanism**. What it produces, stacked over many layers, is a
**contextual embedding**: a vector for each token that depends on the sentence around
it. See [Embedding → Contextual Embedding](../embedding.md#contextual-embedding).

## The intuition

In *"The animal didn't cross the street because **it** was too tired"*, the model has to
work out that **it** means *animal*, not *street*. Self-attention does this by letting the
vector for "it" ask: which other words are relevant to me? It then mixes those words'
information into its own representation.

## The mechanism

Each token's vector $x_i$ is projected into three vectors by learned weight matrices. In
the first layer $x_i$ is the token embedding. In every later layer it is the previous
layer's output, which is already contextual.

| Vector | Formula | Role |
|---|---|---|
| **Query** $q_i$ | $x_i W^Q$ | "What am I looking for?" |
| **Key** $k_j$ | $x_j W^K$ | "What do I contain?" (what queries are matched against) |
| **Value** $v_j$ | $x_j W^V$ | "What I pass on if selected" |

Then, for all tokens at once:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Step by step, for one token $i$:

```
1. score   s_ij = q_i · k_j          how well token j matches what token i is looking for
2. scale   s_ij / √d_k               keeps the scores from growing with the dimension
3. softmax a_ij = softmax_j(s_ij)    turns scores into weights that sum to 1
4. mix     out_i = Σ_j a_ij · v_j    a weighted average of all the value vectors
```

Without the $\sqrt{d_k}$ scaling, large dot products push the softmax into saturation,
where gradients vanish.

## A minimal example

```python
import numpy as np

def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv            # (n, d_k) each
    scores = Q @ K.T / np.sqrt(K.shape[-1])    # (n, n): every token vs every token
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)  # softmax over each row
    return weights @ V                         # (n, d_k): context-mixed tokens

n, d = 4, 8                                    # 4 tokens, 8-dim embeddings
X = np.random.randn(n, d)
W = [np.random.randn(d, d) for _ in range(3)]
print(self_attention(X, *W).shape)             # (4, 8)
```

The `(n, n)` weight matrix is the attention map: row $i$ says how much token $i$ draws
from each token.

## The pieces around it

| Concept | What it adds |
|---|---|
| **Multi-head attention** | Runs $h$ attention operations in parallel, each with its own $W^Q, W^K, W^V$, then concatenates and projects the results. Different heads can specialise, for example one in syntax, another in coreference. |
| **Causal mask** | In decoders such as GPT-style models, scores for future positions ($j > i$) are set to $-\infty$ before the softmax, so a token can't see what comes after it. That's what makes left-to-right generation possible. |
| **Positional encoding** | Attention alone ignores order: shuffle the tokens and you get the same outputs, shuffled. Position information (sinusoidal, learned, or rotary/RoPE) is added so order matters. |
| **"Self" vs cross-attention** | In *self*-attention, Q, K and V all come from the same sequence. In *cross*-attention (encoder–decoder models), Q comes from the decoder and K, V from the encoder output. |

## What it builds: contextual embeddings

Self-attention's output is not the token's final vector. It is *added* to the token's
existing vector (the residual connection), then an MLP processes each token on its own,
and the block repeats layer after layer. The hidden state that comes out is a
**contextual embedding**: `bank` ends up with different vectors in *"river bank"* and
*"bank account"*. Self-attention is the only part of the block that brings in context.

The full picture (static vs contextual vectors, parameter vs activation, the transformer
block, and turning token vectors into a sentence vector) is in
[Embedding → Contextual Embedding](../embedding.md#contextual-embedding).

## Why it replaced RNNs

| | RNN / LSTM | Self-attention |
|---|---|---|
| Path between distant tokens | $O(n)$ steps, so information fades | $O(1)$: a direct connection |
| Parallelism | Sequential, one token at a time | All tokens at once |
| Cost in sequence length | $O(n)$ | $O(n^2)$ time and memory for the $n \times n$ matrix |

The $O(n^2)$ cost is the main weakness, and it's why long context windows are expensive.
Techniques such as FlashAttention (computing attention exactly, but with far less memory
traffic), sparse or sliding-window attention, and linear-attention variants all target it.
