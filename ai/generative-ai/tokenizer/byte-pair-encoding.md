# Byte Pair Encoding (BPE)

> Follows MIT 15.773 *Hands-on Deep Learning*,
> [Lecture 9: Large Language Models](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec09_pdf/)
> (Ramakrishnan, Spring 2024), slides 58–67.

**Byte Pair Encoding** is the tokenizer used by the GPT family. It splits text into
**sub-word tokens**: a vocabulary *learned from a corpus* by repeatedly merging the pair
of adjacent tokens that occurs most often. The final vocabulary holds:

| Kind of token | Example | Why it's there |
|---|---|---|
| Single characters (bytes in GPT) | `r`, `é` | Anything can still be spelled out, so there is no "unknown word" |
| Frequent full words | `the`, ` station` | Common words cost one token |
| Frequent sub-words | `un`, `ably`, `berry` | Rare words are built from reusable pieces |

BPE is a "middle ground" between characters and complete words (lecture). BERT uses a
close relative, [WordPiece](wordpiece.md).

## Why not just split on words?

The earlier lectures' Keras pipeline was *standardize → tokenize (split on spaces) →
index → encode*. Word-level tokens have problems that sub-words fix:

| Problem | Word-level | Character-level | BPE |
|---|---|---|---|
| Unseen word (*"Feldera"*, a typo) | Out-of-vocabulary → `[UNK]` | Fine | Fine: falls back to pieces |
| Vocabulary size | Huge (every word form) | Tiny (~100) | Chosen: e.g. 50k or 100k |
| Sequence length | Short | Very long, and attention costs $O(n^2)$ | Short for common text |
| Shared meaning (`play`, `playing`, `played`) | Three unrelated IDs | Lost in letters | Shared piece `play` |

## How BPE works

**Key intuition (lecture):** start with each character as a token, and **merge** the
tokens that most frequently occur next to each other. **Don't merge across
whitespace.** Stop when the vocabulary reaches a user-defined size.

### Training: learn the merges

The lecture's corpus *"The cat sat on the mat"*, standardised to lower case, with a
target vocabulary of 12 (whitespace ignored for simplicity):

| Step | Most frequent adjacent pair | Vocabulary | Corpus |
|---|---|---|---|
| Start | | `t h e c a s o n m` (9) | `t,h,e _ c,a,t _ s,a,t _ o,n _ t,h,e _ m,a,t` |
| Merge 1 | `a,t` – 3 | + `at` (10) | `t,h,e _ c,at _ s,at _ o,n _ t,h,e _ m,at` |
| Merge 2 | `t,h` – 2 (tied with `h,e`) | + `th` (11) | `th,e _ c,at _ s,at _ o,n _ th,e _ m,at` |
| Merge 3 | `th,e` – 2 | + `the` (12) | `the _ c,at _ s,at _ o,n _ the _ m,at` |

Training produces two things: the **vocabulary** and the **ordered list of merges**
(`a,t → at`, `t,h → th`, `th,e → the`).

### Encoding: replay the merges

A new piece of text is split into characters, and the merges are applied **in the same
order** they were learned:

```
the rat   →  t,h,e _ r,a,t
merge 1   →  t,h,e _ r,at
merge 2   →  th,e  _ r,at
merge 3   →  the   _ r,at      tokens: [the] [r] [at]
```

`rat` never appeared in training, yet it is encoded without an unknown token. Each token
is then mapped to its integer ID, its position in the vocabulary.

### Minimal implementation

```python
from collections import Counter

def train(corpus, vocab_size):
    words = [list(w) for w in corpus.split()]          # don't merge across whitespace
    vocab = sorted({c for w in words for c in w})       # start: every character
    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter(p for w in words for p in zip(w, w[1:]))
        if not pairs:
            break
        best = max(pairs, key=pairs.get)                # most frequent adjacent pair
        merges.append(best)
        vocab.append(best[0] + best[1])
        words = [apply(w, best) for w in words]
    return vocab, merges

def apply(w, pair):                                     # merge every occurrence of pair in w
    out, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
            out.append(w[i] + w[i + 1]); i += 2
        else:
            out.append(w[i]); i += 1
    return out

def encode(text, merges):
    words = [list(w) for w in text.split()]
    for pair in merges:                                 # replay merges in training order
        words = [apply(w, pair) for w in words]
    return words

vocab, merges = train("the cat sat on the mat", vocab_size=12)
print(merges)                     # [('a', 't'), ('t', 'h'), ('th', 'e')]
print(encode("the rat", merges))  # [['the'], ['r', 'at']]
```

The vocabulary size is the one knob: **vocabulary = base symbols + number of merges**.

## Origin: from compression to NLP

| Year | Work | Contribution |
|---|---|---|
| 1994 | Gage, *A New Algorithm for Data Compression* | The original BPE, a compression algorithm: repeatedly replace the most frequent pair of **bytes** with an unused byte |
| 2016 | Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units* | Used it to **learn a sub-word vocabulary** for translation, handling rare and unseen words. Merges stay inside words (end-of-word marker `</w>`) |
| 2019 | Radford et al., *GPT-2* | **Byte-level BPE**: start from the 256 byte values of UTF-8 instead of characters |

## How it relates to GPT

### Byte-level BPE

From GPT-2 onwards, BPE runs over **UTF-8 bytes**, not Unicode characters:

- **The base vocabulary is the 256 byte values**, so any string (any language, emoji,
  code) can be encoded and nothing is ever unknown.
- **A regex pre-splits the text** into chunks (words, numbers, punctuation, whitespace),
  and merges never cross a chunk boundary. This is the production version of the
  lecture's "don't merge across whitespace". It stops variants like `dog.`, `dog!` and
  `dog?` each taking a vocabulary slot.
- **The leading space belongs to the word:** ` station` (with a space) and `station` are
  different tokens.

| Encoding ([tiktoken](https://github.com/openai/tiktoken)) | Used by | Vocabulary size |
|---|---|---|
| `gpt2` | GPT-2, GPT-3 | 50,257 = 256 bytes + 50,000 merges + `<\|endoftext\|>` |
| `cl100k_base` | GPT-3.5, GPT-4 | ~100k |
| `o200k_base` | GPT-4o | ~200k |

The lecture gives ~50,000 for GPT-2/3 and ~100,000 for GPT-3.5/4. Real output from
`cl100k_base`:

| Text | Tokens |
|---|---|
| `The train slowly left the station` | `The` ` train` ` slowly` ` left` ` the` ` station` (6) |
| `unbelievably` | `un` `belie` `vably` |
| `strawberry` | `str` `aw` `berry` |
| `12345678` | `123` `456` `78` |

### Where tokens enter and leave the model

The tokenizer fixes the **vocabulary size $V$**, and $V$ sets both ends of a GPT model,
the lecture's "Transformer Causal Encoder" (see [Transformer](../models/transformer.md)
and [causal self-attention](../attention-is-all-you-need/self-attention.md#causal-masked-self-attention)):

```
text ──BPE──► token IDs ──► embedding table (V × d) ──► causal transformer stack
                                                                 │
next token ◄── sample ◄── softmax over V tokens ◄── dense ◄──────┘
```

- **Input:** each token ID selects a row of a $V \times d$ embedding matrix.
- **Output:** the last position's softmax has **$V$ categories**, one per token
  (lecture: "# of softmax categories = vocab size"). Generation samples one token,
  appends it and runs again, so GPT writes **one BPE token at a time**, not one word.
- **Context windows and API pricing are counted in tokens.** In English, a token is
  roughly ¾ of a word.

### The trade-off in choosing $V$

| Larger vocabulary | Smaller vocabulary |
|---|---|
| Shorter sequences: more text fits in the context, and attention ($O(n^2)$) is cheaper | Longer sequences |
| Bigger embedding and softmax matrices | Smaller matrices |
| Rare tokens get few training examples | Each token is well trained |

### Quirks that come from BPE

| Behaviour | Cause |
|---|---|
| Counting letters (*"how many r's in strawberry?"*) is hard | The model sees `str` `aw` `berry`, not letters |
| Arithmetic is awkward | Numbers are split into arbitrary chunks (`123` `456` `78`) |
| Non-English text costs more tokens | The merges were learned mostly from English, so other scripts break into more, shorter pieces |
| ` hello` ≠ `hello` | The leading space is part of the token, so a trailing space in a prompt changes the result |

## BPE vs other sub-word tokenizers

| | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| Builds the vocabulary by | Merging the **most frequent** pair | Merging the pair that most **increases the likelihood** of the training data | Starting large and **pruning** the tokens that matter least |
| Used by | GPT family, RoBERTa, Llama | BERT, DistilBERT | T5, ALBERT, XLNet (via SentencePiece) |
| Marks word pieces | Leading space inside the token (GPT) | `##` on continuations (`play`, `##ing`) | `▁` for a leading space |

[SentencePiece](https://github.com/google/sentencepiece) is a library that implements
both BPE and Unigram directly on raw text, treating the space as an ordinary symbol.

## Try it

- [GPT tokenizer visualiser](https://observablehq.com/@simonw/gpt-tokenizer) (Simon
  Willison), the lecture's demo.
- [tiktoken](https://github.com/openai/tiktoken), OpenAI's BPE tokenizer:
  `tiktoken.get_encoding("cl100k_base").encode("hello")`.
- [minbpe](https://github.com/karpathy/minbpe) and the video
  [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE)
  (Karpathy), the lecture's "code BPE from scratch" sidebar.


# Reference

* Ramakrishnan. *15.773 Hands-on Deep Learning, Lecture 9: Large Language Models*. MIT OpenCourseWare, Spring 2024. [Slides](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec09_pdf/)
* Gage. *A New Algorithm for Data Compression*. The C Users Journal, 12(2), 1994.
* Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
* Radford, Wu, Child, Luan, Amodei, Sutskever. *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI, 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* Schuster, Nakajima. *Japanese and Korean Voice Search* (WordPiece). ICASSP 2012.
* Wu et al. *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*. 2016. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
* Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates* (Unigram LM). ACL 2018. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959)
* Kudo, Richardson. *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. EMNLP 2018. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
* Karpathy. [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE), 2024, and [minbpe](https://github.com/karpathy/minbpe).
* OpenAI. [tiktoken](https://github.com/openai/tiktoken).
