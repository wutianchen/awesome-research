# Self-attention

> Follows MIT 15.773 *Hands-on Deep Learning*,
> [Lecture 8: Transformers II](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/)
> (Ramakrishnan, Spring 2024), and uses its notation. Introduced in
> [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

**Self-attention** is the layer that turns each word's **stand-alone embedding** into a
**contextual embedding**. It replaces every word's vector with a **weighted average of
all the words' vectors in the same sentence** ("self": the sentence attends to itself).
The weights say how related each word is to the word being updated, and they come from
**learned key, query and value matrices**.

$n$ vectors go in and $n$ vectors come out, one per word, each now informed by the whole
sentence.

## The intuition

In *"The train slowly left the station"*, `station` could be a train station, a radio
station or a military posting. To find out which, `station` should **pay attention** to
`train` a lot, to `slowly` and `left` a little, and to `The` hardly at all. Self-attention
does this by pulling `station`'s vector toward `train`'s. The same idea resolves
*"The animal didn't cross the street because **it** was too tired"*: `it` should draw
mostly from `animal`.

## Step 1: a plain weighted average (Lecture 7, reviewed)

Write the positional input embeddings of the six words as $w_1 \dots w_6$. The contextual
embedding of `station` ($w_6$) is

$$
\hat{w}_6 = s_{1,6}\,w_1 + s_{2,6}\,w_2 + s_{3,6}\,w_3 + s_{4,6}\,w_4 + s_{5,6}\,w_5 + s_{6,6}\,w_6
$$

where $s_{i,6}$ is how much weight word $i$ gets. Related words have large dot products,
and a softmax turns the dot products into weights that are non-negative and sum to 1:

$$
s_{i,6} = \frac{\exp\langle w_i, w_6\rangle}{\sum_{k=1}^{6} \exp\langle w_k, w_6\rangle}
$$

**The problem:** this layer has **no internal weights**. Nothing in it can be learned,
and in multi-head attention every head would produce exactly the same output. The layer
has to be made **"tunable"**.

## Step 2: making it tunable

Lecture 8 adds learnable matrices one at a time. Multiplying a vector by a matrix is the
same as a dense layer with a linear activation, and the numbers in the matrix are weights
optimised by backpropagation. That is what "tunable" means.

| Version | Weight $s_{1,6}$ | Contextual embedding $\hat{w}_6$ | Learned |
|---|---|---|---|
| Plain | $\dfrac{\exp\langle w_1, w_6\rangle}{\sum_i \exp\langle w_i, w_6\rangle}$ | $\sum_i s_{i,6}\, w_i$ | Nothing |
| One matrix | $\dfrac{\exp\langle A w_1, A w_6\rangle}{\sum_i \exp\langle A w_i, A w_6\rangle}$ | $\sum_i s_{i,6}\, w_i$ | $A$ |
| Key + query | $\dfrac{\exp\langle A^K w_1, A^Q w_6\rangle}{\sum_i \exp\langle A^K w_i, A^Q w_6\rangle}$ | $\sum_i s_{i,6}\, w_i$ | $A^K, A^Q$: "twice as tunable" |
| Key + query + value | same as above | $\sum_i s_{i,6}\, A^V w_i$ | $A^K, A^Q, A^V$ |

The three matrices give each word three roles:

| Vector | Formula | Role |
|---|---|---|
| **Query** | $A^Q w_j$ | What word $j$ (here `station`) is looking for |
| **Key** | $A^K w_i$ | What word $i$ is matched on |
| **Value** | $A^V w_i$ | What word $i$ passes on if it gets weight |

Why separate matrices instead of one $A$? With one $A$, the score
$\langle A w_i, A w_j\rangle$ is symmetric: `it` would weigh `animal` exactly as much as
`animal` weighs `it`. Separate key and query matrices let relevance be one-directional,
and a separate value matrix lets *what is matched on* differ from *what is passed on*.

## Step 3: all words at once

Stack the queries, keys and values for all $n$ words as the rows of $Q$, $K$ and $V$.
Then every weight and every contextual embedding comes from one matrix equation, the
paper's:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

| Part | Shape | Meaning |
|---|---|---|
| $QK^\top$ | $n \times n$ | Every query against every key: all the $\langle A^K w_i, A^Q w_j\rangle$ |
| $\div \sqrt{d_k}$ | | Keeps the scores from growing with the key dimension $d_k$; without it, large dot products push the softmax into all-or-nothing weights where gradients vanish. The lecture's formulas leave this out |
| softmax per row | $n \times n$ | Row $j$ holds the weights $s_{1,j} \dots s_{n,j}$, summing to 1 |
| $\times V$ | $n \times d_v$ | Row $j$ is $\hat{w}_j = \sum_i s_{i,j}\, A^V w_i$ |

## A minimal example

```python
import numpy as np

def self_attention(W, Aq, Ak, Av, causal=False):
    # W: (n, d) positional input embeddings, one word per row.
    # A vector-times-matrix "A w" for every row at once is W @ A.T.
    Q, K, V = W @ Aq.T, W @ Ak.T, W @ Av.T       # (n, d_k) each
    scores = Q @ K.T / np.sqrt(K.shape[-1])      # (n, n): every word vs every word
    if causal:                                   # see "Causal (masked) self-attention"
        n = len(W)
        scores[np.triu(np.ones((n, n), dtype=bool), k=1)] = -np.inf   # hide later words
    S = np.exp(scores - scores.max(-1, keepdims=True))
    S /= S.sum(-1, keepdims=True)                # softmax per row: weights sum to 1
    return S @ V                                 # (n, d_k): one contextual embedding per word

n, d = 6, 8                                      # "The train slowly left the station"
W = np.random.randn(n, d)
A = [np.random.randn(d, d) for _ in range(3)]    # A^Q, A^K, A^V: learned in a real model
print(self_attention(W, *A).shape)               # (6, 8)
```

The `(n, n)` matrix `S` is the attention map: row $j$ says how much word $j$ draws from
each word.

## Multi-head attention

One set of $A^K, A^Q, A^V$ learns one pattern of relatedness. Multi-head attention runs
several **attention heads** in parallel on the same input, **each with its own**
$A^K, A^Q, A^V$, then **concatenates and projects** their outputs back to one vector per
word. Because each head's matrices are learned separately, heads learn different
patterns (tense, tone, relationships between entities). This is the reason the layer had
to be made tunable: without its own matrices, every head would compute the same thing.

## Causal (masked) self-attention

Everything above lets each word attend to **every** word, before and after it. That is
**bidirectional** self-attention, used in encoders such as BERT. Models that **generate**
text, such as GPT and the transformer decoder, use **causal self-attention**, also
called **masked self-attention**: each word may attend only to **itself and the words
before it**.

### Why it's needed

A generator predicts the next word from the words so far. If position $j$ could attend
to position $j+1$ during training, it would simply **copy the answer** it is supposed to
predict, and at generation time that word doesn't exist yet. The mask makes training
match generation: what a word can see is exactly what would exist when it is generated.

### How it works

Give every later word **zero weight**. In the lecture's notation, where $s_{i,j}$ is the
weight of word $i$ in the contextual embedding of word $j$:

$$
s_{i,j} = 0 \quad \text{for every } i > j,
\qquad
s_{i,j} = \frac{\exp\langle A^K w_i, A^Q w_j\rangle}{\sum_{k \le j} \exp\langle A^K w_k, A^Q w_j\rangle} \quad \text{for } i \le j
$$

The softmax now sums only over the allowed words, so each row's weights still add up to
1. In practice you don't zero the weights after the softmax. You add a **mask** to the
scores **before** it: $0$ where attention is allowed, $-\infty$ where it isn't, because
$e^{-\infty} = 0$:

$$
\text{CausalAttention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad
M_{j,i} = \begin{cases} 0 & i \le j \\ -\infty & i > j \end{cases}
$$

For *"The train slowly left the station"*, the allowed weights form a **lower
triangle** (rows: the word being updated; columns: the word attended to):

```
            The  train  slowly  left  the  station
The          ✓     ·      ·      ·     ·     ·
train        ✓     ✓      ·      ·     ·     ·
slowly       ✓     ✓      ✓      ·     ·     ·
left         ✓     ✓      ✓      ✓     ·     ·
the          ✓     ✓      ✓      ✓     ✓     ·
station      ✓     ✓      ✓      ✓     ✓     ✓      ← sees the whole sentence
```

- `station` is last, so nothing is hidden from it. It can still use the lecture's
  weights (0.05, 0.3, 0.12, 0.08, 0.05, 0.4).
- `train` can use only *The* and itself. It can't learn from *station* that this is a
  railway context. Only words **after** `station` could combine both.
- In the code above, `causal=True` sets the upper triangle of the scores to $-\infty$, so
  the upper triangle of `S` is exactly 0.

### What it makes possible

- **One sentence = many training examples, in parallel.** Every position predicts its
  next word from its prefix. *"The → train"*, *"The train → slowly"*, … are all trained
  in one pass, with no leakage. This is what LLM pretraining does (next-token
  prediction).
- **Generation one word at a time.** At inference, append the predicted word and run
  again. Earlier words never attend to later ones, so their vectors don't change when a
  word is added. That is why their keys and values can be cached and reused (the **KV
  cache**).

### Causal vs bidirectional

| | Bidirectional | Causal (masked) |
|---|---|---|
| Each word sees | All words | Itself and earlier words |
| Weight matrix | Full $n \times n$ | Lower triangle |
| Used in | Encoders: BERT; the lectures' Transformer Encoder | Decoders: GPT, most LLMs; the lectures' "Transformer **Causal** Encoder" |
| Good for | Understanding: classification, labeling, embeddings | Generation: next-token prediction |
| Training objective | Masked language modelling (hide words, predict them) | Next-token prediction |

Note the two different "masks": BERT masks **input tokens** (replaces words with
`[MASK]`), while causal self-attention masks **attention weights** (hides later
positions).

## The pieces around it

| Concept | What it adds |
|---|---|
| **Causal mask** | Hides later words from each word; see [Causal (masked) self-attention](#causal-masked-self-attention). |
| **Positional encoding** | A weighted average ignores order: shuffle the words and you get the same outputs, shuffled. So $w_i$ is the stand-alone embedding **plus** a position embedding (learned in the lectures; sinusoidal or rotary/RoPE in other models). |
| **"Self" vs cross-attention** | In *self*-attention, queries, keys and values all come from the same sentence. In *cross*-attention (encoder–decoder models), queries come from the decoder and keys and values from the encoder output. |

## Where it sits in the transformer

In a transformer encoder block, self-attention (multi-head) is followed by a
feed-forward layer, and each of the two has a **residual connection** (its input is
added to its output, which helps gradients flow) and **layer normalisation** (each
embedding is standardised, which keeps values small). Blocks are stacked $N$ times, each
starting from vectors that are already contextual. See
[Transformer → The encoder](../models/transformer.md#the-encoder) for the whole block,
and [Embedding → Contextual Embedding](../embedding.md#contextual-embedding) for what the
resulting vectors are.

## Why it replaced RNNs

| | RNN / LSTM | Self-attention |
|---|---|---|
| Path between distant tokens | $O(n)$ steps, so information fades | $O(1)$: a direct connection |
| Parallelism | Sequential, one token at a time | All tokens at once |
| Cost in sequence length | $O(n)$ | $O(n^2)$ time and memory for the $n \times n$ matrix |

The $O(n^2)$ cost is the main weakness, and it's why long context windows are expensive.
Techniques such as FlashAttention (computing attention exactly, but with far less memory
traffic), sparse or sliding-window attention, and linear-attention variants all target it.
