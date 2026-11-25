# Training

How an LLM is made: the stages a model goes through, from raw text to a chat or
reasoning assistant, and the vocabulary around them.

```
                    ┌──────────── base model ─────────────┐
raw text  ──►  pre-training  ──►  mid-training  ──►  post-training  ──►  assistant
              trillions of       billions of        SFT → preference tuning → RL
              raw tokens         curated tokens     (thousands–millions of examples)
                                                                         │
                                       adapting a released model ◄───────┘
                                       fine-tuning, LoRA, distillation, continued pre-training
```

## Phases

| Phase | Goal | Objective | Data | Output |
|---|---|---|---|---|
| **Pre-training** | General language, facts, code | Next-token prediction (self-supervised) | Web, books, code: trillions of tokens | Base model |
| **Mid-training** | Sharpen chosen skills: math, code, reasoning, long context, a domain | **Still** next-token prediction | Smaller, curated, high quality; synthetic Q&A and reasoning traces | Better base model |
| **Post-training: SFT** | Follow instructions, a chat format | Next-token loss on the **response** only | Instruction–response pairs | Instruct model |
| **Post-training: preference tuning** | Helpful, harmless, preferred style | Reward model + RL (RLHF), or a direct loss (DPO) | Pairs of responses ranked by humans or AI | Aligned chat model |
| **Post-training: RL for reasoning** | Solve problems correctly, think longer | RL with **verifiable** rewards | Math, code, puzzles with checkable answers | Reasoning model |

### Pre-training

Train a randomly initialised transformer to **predict the next token** on as much text as
possible (see [self-supervised learning](../learning-algorithms/self-supervised-learning.md)).
Every prefix of every document is a training example, so no labels are needed.

| | |
|---|---|
| Scale | Llama 3: over 15T tokens. Almost all of a model's training compute |
| What it learns | Grammar, facts, reasoning patterns, code, many languages |
| What it doesn't do | Follow instructions. A base model **continues** text: ask it a question and it may write more questions |
| How much data per parameter | **Scaling laws**: loss falls predictably with parameters, data and compute (Kaplan 2020). Chinchilla (Hoffmann 2022): compute-optimal is roughly **20 tokens per parameter**. Today's models train far past that, because a smaller model is cheaper to serve |

### Mid-training

A shorter stage at the end of pre-training. It keeps next-token prediction but switches to
a **curated data mix**, usually while the learning rate decays (**annealing**).

| Why | |
|---|---|
| Data quality matters most at the end | What the model sees as the learning rate falls shapes the final model, so labs save their best data for it |
| Cheap to iterate | Try several mixes from one pre-trained checkpoint without paying for pre-training again |
| Prepares for post-training | Q&A and reasoning-trace text make SFT and RL work better, especially for [reasoning models](../techniques/chain-of-thought/README.md) |
| Adds capabilities | Long-context extension (long documents), new languages, domains (code, math, law, medicine) |

Llama 3 calls this **annealing**; OLMo 2 calls it **mid-training** (the "Dolmino" mix). The
term dates from about 2024 and labs use it loosely.

### Post-training

Turns a base model into an assistant. Usually several rounds of these steps:

| Step | How it works | Notes |
|---|---|---|
| **Supervised fine-tuning (SFT)**, also **instruction tuning** | Train on (instruction, ideal response) pairs; the loss counts only the response tokens | FLAN showed it improves zero-shot ability. LIMA: ~1,000 carefully chosen examples go a long way |
| **Reward model** | Humans compare two responses; a model learns to score which is preferred | Used by RLHF |
| **RLHF** (RL from human feedback) | Optimise the model with RL (PPO) to maximise the reward model's score, with a penalty (KL) for drifting too far from the SFT model | InstructGPT, the recipe behind ChatGPT |
| **DPO** (Direct Preference Optimisation) | Learn from the preference pairs directly with a classification-style loss, no reward model and no RL loop | Simpler and more stable; widely used in open models |
| **RLAIF / Constitutional AI** | An AI, guided by written principles, produces the preference labels instead of humans | Scales feedback beyond human labelling |
| **Rejection sampling fine-tuning** | Sample many responses, keep the best (by reward or correctness), run SFT on them | Llama 2; also the idea behind STaR |
| **RLVR** (RL with verifiable rewards) | Reward = is the answer correct? (math checker, unit tests), not a learned model | GRPO (DeepSeekMath) is the common algorithm. DeepSeek-R1 used it to produce long chain-of-thought reasoning |

**Alignment** is the umbrella term for making a model's behaviour match human intent and
values. In practice it means the preference-tuning and safety parts of post-training.

## Adapting a released model

What you can do with someone else's base or instruct model:

| Method | What changes | When to use |
|---|---|---|
| **Fine-tuning** (full) | All weights, on your labeled task data | Enough data and GPUs; maximum quality |
| **PEFT** (parameter-efficient fine-tuning) | A small number of new or selected parameters; the original weights stay frozen | Limited hardware; many tasks sharing one base model |
| · **LoRA** | Adds a low-rank update $W + BA$ to weight matrices; only $A$ and $B$ are trained | The default PEFT method |
| · **QLoRA** | LoRA on top of a 4-bit quantised base model | Fine-tune large models on one GPU |
| · **Adapters**, **prompt / prefix tuning** | Small inserted layers, or learned "virtual tokens" | Earlier PEFT methods |
| **Continued pre-training** (domain-adaptive pre-training) | Next-token training on unlabeled domain text | Your domain's language differs (legal, biomedical, a codebase) |
| **Distillation** | A small **student** is trained to match a large **teacher**'s outputs (soft probabilities, or generated text) | Smaller, cheaper models; many small reasoning models are distilled from large ones |
| **Transfer learning** | The general idea behind all of the above: reuse what was learned on one task for another | See [transfer learning](../learning-algorithms/transfer-learning.md) |

Not training, often confused with it: **quantisation** (store weights in fewer bits after
training) and **pruning** (remove weights) make a model smaller without teaching it
anything new.

## Learning paradigms

Where the learning signal comes from. Each is carried out by training.

| Paradigm | Signal | In the LLM pipeline |
|---|---|---|
| **Supervised** | Human-provided labels | SFT, task fine-tuning |
| **Self-supervised** | Labels made from the data itself (hide part, predict it) | Pre-training, mid-training ([note](../learning-algorithms/self-supervised-learning.md)) |
| **Unsupervised** | None: find structure | Clustering embeddings; not a training stage of LLMs |
| **Semi-supervised** | A few labels plus a lot of unlabeled data | Pseudo-labelling, self-training |
| **Reinforcement** | A reward for actions or whole outputs | RLHF, RLVR ([note](../learning-algorithms/reinforcement-learning.md)) |
| **Transfer** | Knowledge from a previous task | Pre-train once, fine-tune many times |

## Learning without training

At inference time the weights are frozen, yet behaviour can still improve.

| Technique | What happens |
|---|---|
| **In-context learning**: zero-shot, few-shot | The model infers the task from instructions or examples in the prompt (GPT-3) |
| **Chain of thought** | The model writes out reasoning before answering ([note](../techniques/chain-of-thought/README.md)) |
| **Retrieval-augmented generation (RAG)** | Relevant documents are fetched and put in the prompt, so knowledge is added without retraining |
| **Test-time compute** | Spend more inference on a question: longer reasoning, sampling many answers and voting or verifying |

## Core vocabulary

| Term | Meaning |
|---|---|
| **Parameters / weights** | The numbers that training changes |
| **Loss** | One number for how wrong the model is. For LLMs: **cross-entropy** of the next token |
| **Backpropagation** | Computes the gradient of the loss with respect to every weight |
| **Optimiser** | Uses the gradients to update the weights: SGD, Adam, **AdamW** (the LLM default) |
| **Learning rate** and **schedule** | Step size. Typical: **warmup**, then **cosine decay**, or **warmup–stable–decay (WSD)**, whose decay phase is where mid-training data goes |
| **Batch, step, epoch** | Examples per update; one update; one pass over the data. LLM pre-training usually does about **one epoch**, so budgets are counted in tokens |
| **Overfitting / regularisation** | Memorising the training data instead of generalising; countered with weight decay, dropout, more data |
| **Train / validation / test split** | Data for fitting / for choosing settings / for the final honest score |
| **Checkpoint** | A saved copy of the weights during training |
| **Mixed precision** | Train in 16-bit (bf16) to save memory and time |
| **Parallelism** | Splitting training across GPUs: **data** (copies of the model), **tensor** (split each layer), **pipeline** (split the layers) |
| **Base vs instruct vs chat vs reasoning model** | After pre-/mid-training; after SFT; after preference tuning; after RL for reasoning |

## Easily confused

| Pair | Difference |
|---|---|
| Pre-training vs continued pre-training | The same objective; continued pre-training starts from an existing model, on new (often domain) data |
| Mid-training vs continued pre-training | Mid-training is a planned stage inside a lab's pipeline; continued pre-training is usually done afterwards, by someone adapting the model |
| Fine-tuning vs instruction tuning | Instruction tuning is fine-tuning on many tasks phrased as instructions, to teach instruction following in general |
| RLHF vs DPO | Both learn from preference pairs. RLHF trains a reward model and runs RL; DPO optimises one loss directly |
| RLHF vs RLVR | Learned reward from human preferences vs a programmatic check of correctness |
| Fine-tuning vs in-context learning | Changes the weights vs only changes the prompt |
| Distillation vs quantisation | A new, smaller model trained to imitate vs the same model stored with fewer bits |


# Reference

* Mitchell. *Machine Learning*. McGraw-Hill, 1997.
* Kaplan et al. *Scaling Laws for Neural Language Models*. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
* Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla). 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
* Brown et al. *Language Models are Few-Shot Learners* (GPT-3). 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
* Grattafiori et al. (Meta). *The Llama 3 Herd of Models*. 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
* OLMo Team. *2 OLMo 2 Furious*. 2024. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)
* Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD schedule). 2024. [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
* Wei et al. *Finetuned Language Models Are Zero-Shot Learners* (FLAN). 2021. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
* Zhou et al. *LIMA: Less Is More for Alignment*. 2023. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
* Christiano et al. *Deep Reinforcement Learning from Human Preferences*. 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
* Schulman et al. *Proximal Policy Optimization Algorithms*. 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
* Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT). 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
* Bai et al. *Constitutional AI: Harmlessness from AI Feedback*. 2022. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
* Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
* Rafailov et al. *Direct Preference Optimization*. 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
* Shao et al. *DeepSeekMath* (GRPO). 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
* Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* (RLVR). 2024. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
* DeepSeek-AI. *DeepSeek-R1*. 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
* Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
* Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
* Houlsby et al. *Parameter-Efficient Transfer Learning for NLP* (adapters). 2019. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
* Gururangan et al. *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*. 2020. [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)
* Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*. 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
* Loshchilov, Hutter. *Decoupled Weight Decay Regularization* (AdamW). 2017. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
