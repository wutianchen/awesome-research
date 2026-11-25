# BERT

**BERT** (*Bidirectional Encoder Representations from Transformers*; Devlin et al.,
Google, 2018) is a [transformer encoder](transformer.md#the-encoder) pretrained on a
large amount of English text with
[self-supervised learning](../learning-algorithms/self-supervised-learning.md). It turns a sentence into
contextual embeddings, one per token, that you **fine-tune** for your own task.

> Sources: MIT 15.773
> [Lecture 8](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/)
> and the [BERT paper](https://arxiv.org/abs/1810.04805).

## What it is

| | |
|---|---|
| Architecture | Encoder only: a stack of transformer encoder blocks, almost identical to Vaswani et al. (2017). No decoder, so it doesn't generate text |
| Key word: **bidirectional** | Every token attends to tokens on **both** sides. GPT's attention is constrained to the left |
| Input | WordPiece tokens (30k vocabulary), up to 512 per sequence. Each token's input = token embedding + **position** embedding + **segment** embedding (sentence A or B) |
| Special tokens | **`[CLS]`** at the start (its output represents the whole sequence), **`[SEP]`** between and after sentences |

| Size | Layers $L$ | Hidden size $H$ | Heads $A$ | Feed-forward | Parameters |
|---|---|---|---|---|---|
| BERT-Base | 12 | 768 | 12 | 3072 | 110M |
| BERT-Large | 24 | 1024 | 16 | 4096 | 340M |

## How it is pretrained

On BooksCorpus (800M words) and English Wikipedia (2,500M words), with two
self-supervised tasks:

1. **Masked language model (MLM).** Pick 15% of the tokens and predict the originals.
   Of those, 80% become `[MASK]`, 10% a random token and 10% stay unchanged, so the model
   can't count on seeing `[MASK]`, which never appears in fine-tuning. As the lecture
   puts it, this is just sequence labeling: *"The `[MASK]` of the MIT Sloan School of
   `[MASK]`…"* → *mission*, *Management*. Masking is what lets BERT use both sides of a
   word. A left-to-right objective would let each word only see what came before it.
2. **Next sentence prediction (NSP).** Given sentences A and B, predict from `[CLS]`
   whether B really follows A (50% of pairs) or is a random sentence. Later work
   (RoBERTa) found NSP unnecessary.

To fill in the blanks, BERT has to learn grammar, word meaning and facts, so its
contextual embeddings become a general-purpose representation of English.

## How it is used

**Fine-tuning:** add a small output layer and train **all** the weights for a few epochs
on a few hundred to a few thousand labeled examples.

| Task | Which output | Example |
|---|---|---|
| **Sequence classification** | The `[CLS]` vector → `Dense` → sigmoid/softmax | *"I loved the movie"* → Positive |
| **Sequence labeling** | **Every** token's vector → classifier per token | Named entities: `O`, `B-PER`, … ; slots: `B-fromloc.city_name` |
| Sentence-pair tasks | `[CLS]` over *A `[SEP]` B* | Entailment, paraphrase |
| Question answering | Per-token scores for the answer's start and end | SQuAD |

The lecture notes that BERT was trained with `[CLS]`, so it can be used for
classification "out of the box". It is also an excellent starting point for labeling.

When the paper came out, fine-tuned BERT set new state-of-the-art results on 11 NLP
tasks, including GLUE, MultiNLI and SQuAD. In 2019 Google started using it in Search.

## Variants and practice

| Model | Change |
|---|---|
| [RoBERTa](https://arxiv.org/abs/1907.11692) (2019) | More data, longer training, dynamic masking, no NSP |
| [DistilBERT](https://arxiv.org/abs/1910.01108) (2019) | Distilled: about 40% smaller and 60% faster, keeping about 97% of BERT's performance |
| [ALBERT](https://arxiv.org/abs/1909.11942) (2019) | Shares parameters across layers to cut model size |
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) (2019) | Fine-tuned so that one pooled vector per sentence is good for similarity search; the [Sentence Transformers](https://www.sbert.net/) library |

The lecture's practical advice: for classification or labeling on text, use a model like
BERT as a pretrained encoder, label "a few hundred" examples, attach the right final
layers and fine-tune. For a "standard" NLP problem, a ready-made model on the
[Hugging Face Hub](https://huggingface.co/models) may need no fine-tuning at all.


# Reference

* Ramakrishnan. *15.773 Hands-on Deep Learning, Lecture 8: Transformers II*. MIT OpenCourseWare, Spring 2024. [Slides](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/mit15_773_s24_lec08_pdf/)
* Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
* Liu et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. 2019. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
* Sanh, Debut, Chaumond, Wolf. *DistilBERT, a distilled version of BERT*. 2019. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
* Lan et al. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. 2019. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
* Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
* Alammar. [*The Illustrated BERT, ELMo, and co.*](https://jalammar.github.io/illustrated-bert/)
