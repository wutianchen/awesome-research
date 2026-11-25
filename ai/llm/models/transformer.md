# Transformer

> Based on MIT 15.773 *Hands-on Deep Learning* (Ramakrishnan, Spring 2024):
> [Lecture 7: Transformers I](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)
> and [Lecture 8: Transformers II](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/).
> The lectures cover the **encoder**. The **decoder** section follows the original paper,
> [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
> because the lectures leave generation to Lecture 9.

## What a transformer is

A **transformer** is a neural network that turns a sequence of tokens into a sequence of
**contextual embeddings**: one vector per token, each informed by every other token in
the sequence. It was introduced in 2017 for machine translation and is now used for
search, speech, vision, reinforcement learning, LLMs, text-to-image models and
special-purpose systems such as AlphaFold.

The lectures motivate it with **slot filling**: label every word of *"I want to fly from
boston at 7 am…"* with one of 123 slot types. That needs a model that:

| Requirement | How the transformer meets it |
|---|---|
| Takes the **surrounding context** of each word into account | Attention: each word's vector becomes a weighted average of all the words' vectors |
| Takes the **order** of the words into account | Positional embeddings added to the input |
| Produces an **output of the same length** as the input | Every layer maps $n$ vectors to $n$ vectors |

The architecture itself is generic. **The transformer block is used as-is across
applications; what changes is how the input is turned into embeddings** (words, image
patches in the Vision Transformer, table columns in the TabTransformer). Once the input
is in the "common language" of embeddings, the same blocks process it, which also makes
transformers natural for **multimodal** data.

### The two high-level components

```
                 input sequence                         output so far (shifted right)
                       │                                          │
            embedding + position                       embedding + position
                       │                                          │
          ┌────────────▼────────────┐              ┌──────────────▼──────────────┐
          │ ENCODER  (N× blocks)    │              │ DECODER  (N× blocks)        │
          │  multi-head attention   │              │  masked multi-head attention│
          │  feed-forward           │   keys,      │  cross-attention ◄──────────┼── encoder output
          │                         ├──values─────►│  feed-forward               │
          └────────────┬────────────┘              └──────────────┬──────────────┘
                       │                                          │
        contextual embeddings of the input            linear + softmax over vocabulary
                                                                  │
                                                          next output token
```

| | **Encoder** | **Decoder** |
|---|---|---|
| Job | **Understand** the input: turn it into contextual embeddings | **Generate** an output sequence, one token at a time |
| Attention | Every token sees **every** token (bidirectional) | Each token sees only **earlier** output tokens (masked), plus the whole encoder output (cross-attention) |
| Output | $n$ vectors for $n$ input tokens | A probability distribution over the vocabulary for the next token |
| Runs | Once, over the whole input in parallel | Once per generated token |

The original transformer uses both. Most modern models keep only one half:

| Family | Uses | Examples | Typical tasks |
|---|---|---|---|
| **Encoder-only** | Encoder | [BERT](https://arxiv.org/abs/1810.04805), sentence-transformers | Sequence classification, sequence labeling, embeddings, search |
| **Decoder-only** | Decoder without cross-attention (the lectures call it a **causal encoder**) | GPT, most LLMs | Text generation, chat |
| **Encoder–decoder** | Both | The original transformer, T5 | Translation, summarisation |


## The encoder

One **encoder block** is a multi-head attention layer and a feed-forward layer, each
wrapped in a residual connection and layer normalisation. Blocks are stacked $N$ times.

```
w₁…wₙ ──┬──► Multi-head attention ──► (+) ──► LayerNorm ──┬──► Feed-forward ──► (+) ──► LayerNorm ──► ŵ₁…ŵₙ
        │                              ▲                  │                     ▲
        └────────── residual ──────────┘                  └───── residual ──────┘

positional input embeddings                                                     contextual embeddings
```

### 1. Input: positional input embeddings

Each token's input vector, of embedding dimension $D$, is the sum of:

- a **stand-alone embedding** for the word: random at first, or pretrained (e.g. GloVe),
  and
- a **positional embedding** for its position in the sentence, also random at first and
  learned.

Without the positional part, attention ignores order: scramble the sentence and you get
the same contextual embeddings, just shuffled. Worked example: *"cat sat mat"* in
[Embedding → Word order](../embedding.md#word-order-positional-encoding).

### 2. Attention: from stand-alone to contextual

The contextual embedding of word $j$ is a weighted average of all the words' vectors,
with weights from a softmax over dot products. In the lectures' notation, for
*"The train slowly left the station"*:

$$
s_{i,6} = \frac{\exp\langle w_i, w_6\rangle}{\sum_{k=1}^{6}\exp\langle w_k, w_6\rangle},
\qquad
\hat{w}_6 = \sum_{i=1}^{6} s_{i,6}\, w_i
$$

`station` is pulled toward `train` because their dot product is large. The full
derivation is in [Embedding → Contextual Embedding](../embedding.md#contextual-embedding).

### 3. Making attention "tunable": keys, queries and values

As written above, attention has **no internal weights**. Every head would produce the
same output, and nothing could be learned. Lecture 8 fixes this in three steps:

| Step | Weight $s_{1,6}$ / output $\hat{w}_6$ | What's learned |
|---|---|---|
| Plain | $\dfrac{\exp\langle w_1, w_6\rangle}{\sum_i \exp\langle w_i, w_6\rangle}$ | Nothing |
| One matrix | $\dfrac{\exp\langle A w_1, A w_6\rangle}{\sum_i \exp\langle A w_i, A w_6\rangle}$ | $A$ (a dense layer with linear activation) |
| Key and query | $\dfrac{\exp\langle A^K w_1, A^Q w_6\rangle}{\sum_i \exp\langle A^K w_i, A^Q w_6\rangle}$ | $A^K$ (**key** matrix), $A^Q$ (**query** matrix): "twice as tunable" |
| Value | $\hat{w}_6 = \sum_i s_{i,6}\, A^V w_i$ | $A^V$ (**value** matrix) |

- The **query** $A^Q w_6$ is what `station` is looking for.
- The **key** $A^K w_i$ is what word $i$ is matched on.
- The **value** $A^V w_i$ is what word $i$ passes on if chosen.

$A^K, A^Q, A^V$ are learned by gradient descent. With all words at once, this is the
paper's compact form, where $d_k$ is the key dimension:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

### 4. Multi-head attention

Several attention **heads** run in parallel on the same input, **each with its own**
$A^K, A^Q, A^V$, and their outputs are **concatenated and projected** back to dimension
$D$. Because the matrices differ and are learned, heads learn **different patterns**
(tense, tone, relationships between entities), much like multiple filters in a
convolutional layer.

### 5. Feed-forward layer

After attention, each token's vector goes through the same small dense network on its
own. This adds **non-linearity**; attention itself is a weighted average.

### 6. Residual connections and layer normalisation

- **Residual connection:** add each sublayer's **input** to its **output**
  ($x + \text{sublayer}(x)$). This helps gradients flow during backpropagation.
- **Layer normalisation:** after each sublayer, **standardise each embedding** (subtract
  its mean, divide by its standard deviation), then translate and rescale each dimension
  with learned parameters. This keeps the values small and training stable.

Together these are the "Add & Norm" boxes in the paper's diagram. The original
transformer normalises *after* the addition (post-norm). Most modern models normalise
the input to each sublayer instead (pre-norm), which trains more stably when stacks get
deep.

### 7. Stacking and shape

An encoder block takes $n$ vectors of size $D$ and returns $n$ vectors of size $D$:

- **Length $n$ is kept** because attention produces one output per query (the $n \times n$
  weight matrix times $n$ values), and the feed-forward layer works on each position
  separately.
- **Width $D$ is kept** because the heads' concatenation is projected back to $D$, the
  feed-forward layer maps $D \to$ (wider) $\to D$, and the residual addition only works
  if the shapes match.

So blocks are **stackable** ($N\times$), and each block starts from vectors that are
already contextual.

### 8. What gets trained

In the slot-filling model (encoder → `Dense` (ReLU) → 123-way softmax per word),
backpropagation learns:

1. positional embeddings
2. stand-alone embeddings (unless pretrained and frozen)
3. $A^K, A^Q, A^V$ for **each** attention head
4. layer-norm scale and bias
5. the feed-forward weights inside each block
6. the `Dense` layer outside the encoder
7. the final softmax layer

### Using an encoder

| Task | Example | How |
|---|---|---|
| **Sequence labeling** | *"fly from Boston to Denver"* → `O O B-fromloc.city_name O B-toloc.city_name` | A classifier (`Dense` + softmax) on **each** output vector |
| **Sequence classification** | *"I loved the movie"* → Positive | Summarise the $n$ output vectors into **one** sentence vector, then `Dense` (ReLU) + sigmoid/softmax |

For classification, averaging the output vectors works but treats every word the same.
The better approach: add a special **`<CLS>`** token at the start of every sentence and
use **its** output vector. Over training, it comes to represent the sentence as a whole.

**Pretraining with masking.** Labeled data is scarce, but text is everywhere.
**Self-supervised learning** creates labels from the input itself: hide some words with
`<MASK>` and train the encoder to fill them in. *"The `<MASK>` of the MIT Sloan School of
`<MASK>` is to…"* → *mission*, *Management*. This is just sequence labeling, and to fill
in the blanks well the encoder has to learn how words relate. Doing this on a large
amount of English text gives **BERT**. Then, as with transfer learning on images, attach
new output layers, label a few hundred examples, and fine-tune (or freeze the encoder
and train only the new layers). For standard NLP tasks, a ready-made model on the
Hugging Face Hub may need no fine-tuning at all.


## The decoder

> Not covered in Lectures 7–8. This follows *Attention Is All You Need*; the lectures'
> "causal encoder" (Lecture 9) is the decoder-only version.

The decoder **generates** an output sequence **one token at a time**. Each step feeds in
everything generated so far and predicts the next token. This is called
**autoregressive** generation.

### The decoder block

A decoder block has **three** sublayers, each with a residual connection and layer
normalisation, stacked $N$ times:

| Sublayer | Queries from | Keys and values from | Purpose |
|---|---|---|---|
| 1. **Masked** multi-head attention | Output tokens so far | Output tokens so far | Context from what has been generated, but only from **earlier** positions |
| 2. **Cross-attention** (encoder–decoder attention) | Output tokens so far | The **encoder's output** | Look up the relevant parts of the input |
| 3. Feed-forward | n/a | n/a | Non-linearity, per token |

On top of the last block, a linear layer and a **softmax over the whole vocabulary** turn
the last position's vector into probabilities for the next token.

### The causal mask

During generation, token $j$ can't depend on tokens that come after it, because they
don't exist yet. So the decoder's attention gives them **zero weight**. In the lectures'
notation:

$$
s_{i,j} = 0 \quad \text{for every } i > j
$$

In practice, the scores for later positions are set to $-\infty$ before the softmax. This
mask is what makes the lectures' "Transformer **Causal** Encoder" causal. A decoder-only
model like GPT is exactly this: an encoder-style stack with the mask, and no
cross-attention.

### Generating a translation

*"I loved the movie"* → *"Ich habe den Film geliebt"*. The encoder runs once; the
decoder runs once per output token:

| Step | Decoder input | Predicted next token |
|---|---|---|
| 1 | `<start>` | Ich |
| 2 | `<start>` Ich | habe |
| 3 | `<start>` Ich habe | den |
| 4 | `<start>` Ich habe den | Film |
| 5 | `<start>` Ich habe den Film | geliebt |
| 6 | `<start>` Ich habe den Film geliebt | `<end>`: stop |

- **Output length is free.** The encoder maps $n$ tokens to $n$ vectors. The decoder
  stops when it predicts `<end>`, so input and output lengths can differ. Cross-attention
  produces one vector per **query**, meaning per output token, whatever the input length.
- **Training is parallel, generation is not.** In training the whole correct output is
  known, so all positions are computed at once, and the mask stops each one from seeing
  its answer (*teacher forcing*). At generation time each token depends on the previous
  one, so steps run one after another.
- **The KV cache** stores past tokens' keys and values, so each new step only computes
  the new token's attention instead of redoing the whole prefix.


# Reference

* Ramakrishnan. *15.773 Hands-on Deep Learning, Lecture 7: Transformers I*. MIT OpenCourseWare, Spring 2024. [Slides (PDF)](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)
* Ramakrishnan. *15.773 Hands-on Deep Learning, Lecture 8: Transformers II*. MIT OpenCourseWare, Spring 2024. [Slides](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/)
* Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
* Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Vision Transformer). 2020. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
* Huang, Khetan, Cvitkovic, Karnin. *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*. 2020. [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
