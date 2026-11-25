# Self-supervised learning

**Self-supervised learning** trains a model on **unlabeled data** by making the labels
from the data itself. You hide part of the input and train the model to predict it from
the rest.

> **Predict a subset of the input data using the rest of the input.**
> — MIT 15.773 *Hands-on Deep Learning*,
> [Lecture 8](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/)

## Where it fits

| | Labels come from | Example |
|---|---|---|
| **Supervised** | Humans | ImageNet: 1.2M photos labeled with 1000 categories |
| **Unsupervised** | None; find structure | Clustering, PCA |
| **Self-supervised** | **The data itself**: a hidden part of the input becomes the label | Hide a word, predict it |

It is technically supervised training, with a loss, a target and backprop. The
difference is that no human writes the labels, so it can use all the text, images or
audio on the internet.

## How it works: masking

Lecture 8's example, in two steps:

1. **Make "fake" (input, label) pairs** by masking part of the input:
   - **Input:** *"The `<MASK>` of the MIT Sloan School of `<MASK>` is to develop
     principled, innovative `<MASK>` who…"*
   - **Labels:** *mission*, *Management*, *leaders*
2. **Train a network to fill in the blanks.** For text, this is ordinary sequence
   labeling with a [transformer encoder](transformer.md#the-encoder).

The useful result isn't the blank-filling. To fill in blanks well, the model has to learn
how words relate (grammar, meaning, facts), so **it learns a good representation of the
input along the way**. You then keep that encoder and **fine-tune** it on a real task
with a few hundred labeled examples. That is transfer learning, like reusing a
"headless" ResNet for images:

```
pretrain (self-supervised)              fine-tune (supervised, little data)
masked text ──► encoder ──► fill blanks     task text ──► same encoder ──► new output layer ──► label
                   └──────────── keep ─────────────────────────┘
```

## Common pretext tasks

The made-up prediction task is called the **pretext task**. The task you actually care
about afterwards is the **downstream task**.

| Pretext task | What is hidden | Model |
|---|---|---|
| Predict nearby words from a word | Its context | **word2vec** skip-gram (Firth's "company it keeps"; see [Embedding](../embedding.md)) |
| **Masked language modelling** | Random words in a sentence | **BERT** |
| **Next-token prediction** | Everything after position $t$ | **GPT** and most LLMs: every prefix of a text is a training example |
| Contrastive learning | Which two augmented views (crop, colour change) come from the same image | SimCLR (images; end of Lecture 8) |
| Masked image modelling | Image patches | MAE |
| Masked audio | Stretches of a sound recording | wav2vec 2.0 |

## Why it matters

- **Labels are the bottleneck; raw data isn't.** Labeling costs money and time, while
  text and images are plentiful.
- **Pretrain once, fine-tune many times.** One self-supervised model (BERT, GPT) can be
  adapted to classification, labeling, search and more with little labeled data.
- **It's how LLMs are trained.** Pretraining an LLM is self-supervised next-token
  prediction on trillions of tokens. Later stages (instruction tuning, RLHF) then add
  human-labeled data.


# Reference

* Ramakrishnan. *15.773 Hands-on Deep Learning, Lecture 8: Transformers II*. MIT OpenCourseWare, Spring 2024. [Slides](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/)
* Mikolov, Chen, Corrado, Dean. *Efficient Estimation of Word Representations in Vector Space*. 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
* Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
* Brown et al. *Language Models are Few-Shot Learners* (GPT-3). 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
* Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
* He et al. *Masked Autoencoders Are Scalable Vision Learners* (MAE). 2021. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
* Baevski, Zhou, Mohamed, Auli. *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
