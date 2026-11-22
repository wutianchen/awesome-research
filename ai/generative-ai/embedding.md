# Embedding

Model the tokens with number (usually vector) and map the semantic relationship in the world of token to the geometric relationship in the world of its vector representation.

> *"You shall know a word by the company it keeps."*
> — J. R. Firth, *A Synopsis of Linguistic Theory 1930–1955* (1957)

This is the **distributional hypothesis**: words that appear in similar contexts have
similar meanings. Every embedding method below learns from it. The vectors are trained
so that words sharing neighbours end up close together.


## Standalone Embedding

Also called **static** embeddings. Each word gets **one fixed vector**, learned from
co-occurrence statistics over a large corpus. Once training is done, the embedding is a
lookup table: the same word always maps to the same vector, whatever sentence it is in.

| Tool | Year | How it learns "the company a word keeps" |
|---|---|---|
| **LSA** (Latent Semantic Analysis) | 1990 | Builds a word × document count matrix and compresses it with SVD. It came before neural methods but used the same distributional idea |
| **word2vec** ([Mikolov et al.](https://arxiv.org/abs/1301.3781)) | 2013 | A shallow network trained to predict context from a word (**skip-gram**) or a word from its context (**CBOW**). [Negative sampling](https://arxiv.org/abs/1310.4546) made it cheap enough to train on billions of words |
| **GloVe** ([Pennington et al.](https://nlp.stanford.edu/projects/glove/)) | 2014 | Factorises the global word–word co-occurrence matrix directly, so that the dot product of two vectors fits the log of how often the words co-occur |
| **fastText** ([Bojanowski et al.](https://arxiv.org/abs/1607.04606)) | 2016 | word2vec over **character n-grams**: a word's vector is the sum of its subword vectors, so it can build vectors for rare and unseen words (`unhappiness` ← `un`, `happi`, `ness`, …) |

What they get right: the geometry captures relations. The well-known example is
$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$
([Mikolov et al., 2013](https://aclanthology.org/N13-1090/)).

What they can't do: **one vector per word means one vector per spelling.** `bank` (river)
and `bank` (money) share one point, roughly an average of both senses. Fixing that needs
a vector that depends on the sentence, which is a [contextual embedding](#contextual-embedding).


## Contextual Embedding

> Based on MIT 15.773 *Hands-on Deep Learning*, Lecture 7: *Transformers I*
> (Ramakrishnan, Spring 2024).
> [Slides](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)

A **contextual embedding** is a vector for a word that depends on the sentence it is in.
It starts from the word's stand-alone embedding and moves it toward the other words in
the sentence, in proportion to how related they are.

### Why stand-alone embeddings aren't enough

The lecture's running task is **slot filling** on the ATIS airline dataset: label every
word of a query such as *"I want to fly from boston at 7 am and arrive in denver at 11 in
the morning"* with one of **123 slot types** (`O`, `B-fromloc.city_name`,
`B-depart_time.time`, …). So the network must produce one output per input word (18
here) and put a **123-way softmax** on each. That needs a model that:

1. takes the **surrounding context** of each word into account,
2. takes the **order** of the words into account,
3. produces an **output of the same length** as the input.

A single embedding per word, as GloVe gives, fails the first requirement, because
the meaning of a word can change completely with its context:

| Word | Example contexts |
|---|---|
| see | *I will **see** you soon* · *I will **see** this project to its end* · *I **see** what you mean* |
| bank | *I went to the **bank** to apply for a loan* · *I am **banking** on the job offer* · *I am standing on the left **bank*** |
| it | *The animal didn't cross the street because **it** was too tired* (it = animal) · *…because **it** was too wide* (it = street) |
| station | *The train left the **station** on time* · *The radio **station** was playing 60s hits* · *I was **stationed** on a remote island* |

### From stand-alone to contextual: a weighted average

Take *"The train slowly left the station"* with stand-alone embeddings $w_1 \dots w_6$.
How can `station`'s embedding $w_6$ take in the other words?

Suppose we knew how much **attention** `station` should pay to each word, that is, how
much weight to give it. Intuitively a lot to *train*, a little to *slowly* and *left*,
and hardly anything to *the*:

| | The | train | slowly | left | the | station |
|---|---|---|---|---|---|---|
| weight $s_{i,6}$ | 0.05 | **0.3** | 0.12 | 0.08 | 0.05 | 0.4 |

The contextual version of $w_6$ is the **weighted average of the stand-alone embeddings
of all the words in the sentence**, including itself:

$$
\hat{w}_6 = s_{1,6}\,w_1 + s_{2,6}\,w_2 + s_{3,6}\,w_3 + s_{4,6}\,w_4 + s_{5,6}\,w_5 + s_{6,6}\,w_6
$$

and the same for every word $j$: $\hat{w}_j = \sum_i s_{i,j}\, w_i$.

### How the weights are chosen: dot product, then softmax

**Intuition:** the weight of a word should be proportional to how *related* it is to
`station`. One way to measure how related two words are is the **dot product** of their
stand-alone embeddings: large when they point the same way, near zero when unrelated,
negative when opposed.

Proper weights must be non-negative and sum to 1. Dot products can be negative, so
exponentiate them and normalise, which is a **softmax**:

$$
s_{i,6} = \frac{\exp\langle w_i, w_6 \rangle}{\sum_{k=1}^{6} \exp\langle w_k, w_6 \rangle}
$$

**What this does geometrically.** `station` has many contexts (trains, radio, military
postings). In this sentence `train` is closely related to `station`, so it exerts a
strong **pull** on it. `radio` is also related to `station`, but it isn't in the
sentence, so it automatically gets zero weight. Moving `station` toward `train`, that is,
paying more *attention* to `train`, contextualises `station`'s embedding to the world of
trains, platforms and departures.

```
train ●  ◄─────────────  ● station                       ● radio
         pulled toward train;  radio absent from the sentence → weight 0
```

**A tiny worked example.** Use 2-D vectors whose dimensions mean *[finance, geography]*:
`bank` $= [0.5, 0.5]$ (both senses mixed), `river` $= [0, 1]$, `account` $= [1, 0]$. In
*"river bank"*, $\langle\text{bank},\text{river}\rangle = \langle\text{bank},\text{bank}\rangle = 0.5$,
so the softmax gives equal weights and

$$
\hat{w}_{\text{bank}} = 0.5\,[0, 1] + 0.5\,[0.5, 0.5] = [0.25,\ 0.75] \quad \text{(toward geography)}
$$

In *"bank account"* the same steps give $[0.75,\ 0.25]$ (toward finance). One
stand-alone vector, two contextual ones.

**As matrix operations.** Stack the stand-alone embeddings as the rows of $W$ ($n$ words
× $d$ dimensions). All the dot products are $WW^\top$, so all contextual embeddings come
from one expression, with the softmax applied to each row:

$$
\hat{W} = \text{softmax}\!\left(W W^\top\right) W
$$

```python
import numpy as np

def contextualize(W):                         # W: (n, d) stand-alone embeddings
    scores = W @ W.T                          # (n, n): every word vs every word
    S = np.exp(scores - scores.max(-1, keepdims=True))
    S /= S.sum(-1, keepdims=True)             # softmax per row: weights sum to 1
    return S @ W                              # (n, d): one contextual vector per word

W = np.array([[0, 1], [0.5, 0.5]])            # "river bank"
print(contextualize(W)[1])                    # [0.25 0.75]
```

This layer, stand-alone embeddings in and contextual embeddings out, one per word, is an
**attention layer**.

### Multi-head attention

One attention layer captures one pattern of relatedness. A sentence holds several at
once, for example patterns of **tense**, of **tone**, and of the **relationships between
entities**. So the transformer runs **several attention layers ("heads") in parallel**
over the same input, then **concatenates and projects** their outputs back to one vector
per word:

```
w₁ … w₆ ──► head 1 ──► ŵ₁ … ŵ₆ ─┐
                                 ├─► concatenate and project ──► ŵ₁ … ŵ₆
w₁ … w₆ ──► head 2 ──► ŵ₁ … ŵ₆ ─┘
```

$$
\text{MultiHead}(W) = [\,\text{head}_1 ;\ \text{head}_2 ;\ \dots ;\ \text{head}_h\,]\; W^O
$$

- It is like having **multiple filters in a convolutional layer**: each filter detects a
  different feature, and each head attends to a different pattern.
- The attention maps from *Attention Is All You Need* show this: one head links each word
  to distant related words, while another links mostly neighbouring words.
- **What makes the heads differ.** The bare weighted average above has no parameters, so
  two copies would compute the same thing. In the real transformer each head first
  applies its **own learned linear projections** to the incoming embeddings, mapping each
  one into three different spaces (a **query**, a **key** and a **value**). Weights come
  from query·key dot products, and the average is taken over the values:
  $\text{head}_m = \text{softmax}\!\big((WW^Q_m)(WW^K_m)^\top / \sqrt{d_k}\big)\, WW^V_m$.
  The lecture leaves this for the next lecture; it's what lets each head learn its own
  notion of "related".

### Feed-forward layers: adding non-linearity

A weighted average is linear. After multi-head attention, the transformer passes each
word's vector through **feed-forward (dense) layers** to inject some non-linearity. The
story so far: *start with random embeddings, end with contextual embeddings*.

### Word order: positional encoding

Context and same-length output are now covered, but **not word order**. A weighted
average doesn't care about order: scramble the words of a sentence and you get exactly
the same contextual embeddings, just shuffled.

**The fix: positional encoding.** Each word's input embedding becomes the sum of two
things:

- its usual **stand-alone embedding** (depends on *which* word), plus
- a **position embedding** (depends on *where* the word is in the sentence).

The lecture's example, *"cat sat mat"*, with 2-D embeddings:

| Word | Stand-alone | Position | Position embedding | Input embedding (sum) |
|---|---|---|---|---|
| cat | $[0.5,\ 7.1]$ | 0 | $[1.3,\ 3.9]$ | $[1.8,\ 11.0]$ |
| sat | $[1.2,\ 5.3]$ | 1 | $[6.3,\ 3.7]$ | $[7.5,\ 9.0]$ |
| mat | $[-2.0,\ -3.1]$ | 2 | $[0.6,\ 8.1]$ | $[-1.4,\ 5.0]$ |

Now `cat` at position 2, as in *"mat sat cat"*, would enter as
$[0.5+0.6,\ 7.1+8.1] = [1.1,\ 15.2]$, a different vector, so the dot products, the
weights and the contextual embeddings all change with order. Both tables are learned,
just like the stand-alone embeddings.

### Putting it together: the transformer encoder

The whole stack is a **transformer encoder**:

```
words ──► stand-alone embedding (random or pretrained)
      ──► + position embedding
      ──► multi-head attention           ← context
      ──► feed-forward layers            ← non-linearity
      ──► contextual embeddings ŵ₁ … ŵₙ  (same length as the input)
```

1. The input embedding can be **random** or **pretrained** (for example GloVe).
2. **Add** a position-dependent embedding for each word's position.
3. Pass through **multi-head attention** to get a context-dependent representation.
4. Pass that through a simple **feed-forward network**.
5. An encoder's input and output have the **same size**, so encoders can be
   **daisy-chained (stacked)** for more modelling capacity. Each layer starts from
   vectors that are already contextual.

The real encoder also has **residual connections** and **layer normalisation** around
each step, and the query/key/value projections described above.

**Back to slot filling.** Feed *"fly from Boston to Denver"* through the encoder, put a
`Dense` (ReLU) layer and then a **123-way softmax** on each output vector, and read off
one slot per word (`O`, `O`, `B-fromloc.city_name`, `O`, `B-toloc.city_name`). All the
weights, including the stand-alone and position embeddings, are learned together by
backpropagation.

### Related facts

- **Context grows with depth.** [Ethayarajh (2019)](https://arxiv.org/abs/1909.00512)
  found that in upper layers, less than 5% of the variance in a word's representations
  can be explained by one static vector.
- **Causal (decoder) models** such as GPT give zero weight to every word to the right, so
  a word only uses the words before it. **Bidirectional (encoder) models** such as
  [BERT](https://arxiv.org/abs/1810.04805) use both directions, which is why they became
  the standard for contextual word embeddings.
- **Embedding APIs** (sentence-transformers, hosted embedding endpoints) return
  contextual vectors, usually from the last layer, either mean-pooled or taken from a
  special token. They don't return the stand-alone embeddings.
- **The KV cache** is the one place contextual information is kept for a while. During
  generation, past tokens' key and value vectors are cached so they aren't recomputed for
  each new token. The cache belongs to one conversation and is then discarded.


# Reference

* Ramakrishnan. *15.773 Hands-on Deep Learning, Lecture 7: Transformers I*. MIT OpenCourseWare, Spring 2024. [Slides (PDF)](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)
* [Deep Learning for Natural Language – Embeddings](https://www.youtube.com/watch?v=LqFc0z-pQTg)
* Firth. *A Synopsis of Linguistic Theory 1930–1955*. In *Studies in Linguistic Analysis*, 1957.
* Mikolov, Chen, Corrado, Dean. *Efficient Estimation of Word Representations in Vector Space*. 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
* Mikolov, Sutskever, Chen, Corrado, Dean. *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS 2013. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
* Pennington, Socher, Manning. *GloVe: Global Vectors for Word Representation*. EMNLP 2014. [Project page](https://nlp.stanford.edu/projects/glove/)
* Bojanowski, Grave, Joulin, Mikolov. *Enriching Word Vectors with Subword Information*. TACL 2017. [arXiv:1607.04606](https://arxiv.org/abs/1607.04606)
* Hemphill, Godfrey, Doddington. *The ATIS Spoken Language Systems Pilot Corpus*. 1990. [ACL Anthology H90-1021](https://aclanthology.org/H90-1021/)
* Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
* Ethayarajh. *How Contextual are Contextualized Word Representations?* EMNLP 2019. [arXiv:1909.00512](https://arxiv.org/abs/1909.00512)
