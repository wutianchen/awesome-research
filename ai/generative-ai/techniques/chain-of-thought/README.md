# Chain of thought

**Chain-of-thought (CoT)** prompting makes an LLM write out its intermediate reasoning
steps before the final answer. Since each generated token is another pass through the
network (see [causal self-attention](../attention-is-all-you-need/self-attention.md#causal-masked-self-attention)),
a longer chain gives the model more computation before it commits to an answer.

Papers marked ⭐ are the ones to read first.

## 1. Origins: getting models to reason step by step

| Paper | Contribution |
|---|---|
| Ling et al., *Program Induction by Rationale Generation* (ACL 2017), [1705.04146](https://arxiv.org/abs/1705.04146) | An early version: train a model to write a rationale before the answer (the AQuA dataset) |
| Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021), [2110.14168](https://arxiv.org/abs/2110.14168) | The **GSM8K** benchmark, and verifiers that rank sampled solutions |
| Nye et al., *Show Your Work: Scratchpads for Intermediate Computation* (2021), [2112.00114](https://arxiv.org/abs/2112.00114) | A "scratchpad" for intermediate steps; the direct precursor of CoT |
| ⭐ **Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*** (NeurIPS 2022), [2201.11903](https://arxiv.org/abs/2201.11903) | **The CoT paper.** Few-shot examples with worked reasoning. It helps only once models are large (~100B parameters) |
| ⭐ **Kojima et al., *Large Language Models are Zero-Shot Reasoners*** (NeurIPS 2022), [2205.11916](https://arxiv.org/abs/2205.11916) | Zero-shot CoT: just add *"Let's think step by step"* |

## 2. Improving the prompting

| Paper | Idea |
|---|---|
| ⭐ **Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*** (ICLR 2023), [2203.11171](https://arxiv.org/abs/2203.11171) | Sample many reasoning paths and take a majority vote on the answer |
| Zhou et al., *Least-to-Most Prompting* (ICLR 2023), [2205.10625](https://arxiv.org/abs/2205.10625) | Split a problem into subproblems and solve them in order |
| Zhang et al., *Automatic Chain of Thought Prompting* (Auto-CoT, ICLR 2023), [2210.03493](https://arxiv.org/abs/2210.03493) | Generate the example reasoning chains automatically |
| Gao et al., *PAL: Program-aided Language Models* (ICML 2023), [2211.10435](https://arxiv.org/abs/2211.10435) | Reason in code and let an interpreter compute the answer |
| Chen et al., *Program of Thoughts Prompting* (2022), [2211.12588](https://arxiv.org/abs/2211.12588) | The same idea, developed independently |
| ⭐ **Yao et al., *ReAct: Synergizing Reasoning and Acting*** (ICLR 2023), [2210.03629](https://arxiv.org/abs/2210.03629) | Interleave reasoning with tool calls; the basis of LLM agents |
| ⭐ **Yao et al., *Tree of Thoughts*** (NeurIPS 2023), [2305.10601](https://arxiv.org/abs/2305.10601) | Search over a tree of partial reasoning steps, with backtracking |
| Besta et al., *Graph of Thoughts* (AAAI 2024), [2308.09687](https://arxiv.org/abs/2308.09687) | Generalises the tree to a graph |

## 3. Training models to reason

| Paper | Idea |
|---|---|
| ⭐ **Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*** (NeurIPS 2022), [2203.14465](https://arxiv.org/abs/2203.14465) | Fine-tune on the model's own reasoning chains that reached correct answers, and iterate |
| Uesato et al., *Solving Math Word Problems with Process- and Outcome-based Feedback* (2022), [2211.14275](https://arxiv.org/abs/2211.14275) | Rewarding each step (process) vs rewarding only the final answer (outcome) |
| ⭐ **Lightman et al., *Let's Verify Step by Step*** (2023), [2305.20050](https://arxiv.org/abs/2305.20050) | Process reward models; the PRM800K dataset |
| Wang & Zhou, *Chain-of-Thought Reasoning Without Prompting* (2024), [2402.10200](https://arxiv.org/abs/2402.10200) | CoT paths already exist in a model's outputs; changing how tokens are picked (decoding) reveals them |

## 4. Reasoning models and test-time compute (2024–2025)

| Paper | Idea |
|---|---|
| OpenAI, [*Learning to Reason with LLMs*](https://openai.com/index/learning-to-reason-with-llms/) (o1, 2024) | Reinforcement learning (RL) trains long hidden CoT; accuracy improves with more thinking time |
| ⭐ **Snell et al., *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters*** (2024), [2408.03314](https://arxiv.org/abs/2408.03314) | When spending more compute at inference beats using a bigger model |
| ⭐ **DeepSeek-AI, *DeepSeek-R1*** (2025), [2501.12948](https://arxiv.org/abs/2501.12948) | RL with only answer-correctness rewards produces long CoT, self-checking and "aha moments"; open weights |
| Muennighoff et al., *s1: Simple Test-Time Scaling* (2025), [2501.19393](https://arxiv.org/abs/2501.19393) | 1,000 examples plus "budget forcing" (appending "Wait") to make the model think longer |
| Hao et al., *Coconut: Training LLMs to Reason in a Continuous Latent Space* (2024), [2412.06769](https://arxiv.org/abs/2412.06769) | Reason in hidden vectors instead of written tokens |

## 5. Why CoT works (theory and analysis)

| Paper | Finding |
|---|---|
| Wei et al., *Emergent Abilities of Large Language Models* (TMLR 2022), [2206.07682](https://arxiv.org/abs/2206.07682) | CoT as an ability that appears only at scale |
| Suzgun et al., *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them* (2022), [2210.09261](https://arxiv.org/abs/2210.09261) | BIG-Bench Hard (BBH), a benchmark of hard tasks where CoT helps |
| Feng et al., *Towards Revealing the Mystery behind Chain of Thought* (NeurIPS 2023), [2305.15408](https://arxiv.org/abs/2305.15408) | Transformers that can't solve some problems directly can with CoT |
| ⭐ **Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought*** (ICLR 2024), [2310.07923](https://arxiv.org/abs/2310.07923) | More CoT steps increase the class of problems a transformer can solve |
| Li et al., *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems* (ICLR 2024), [2402.12875](https://arxiv.org/abs/2402.12875) | CoT adds serial computation steps that a fixed-depth transformer lacks |
| Pfau et al., *Let's Think Dot by Dot* (2024), [2404.15758](https://arxiv.org/abs/2404.15758) | Meaningless filler tokens can help too, so the extra computation, not the words, can be doing the work |
| Sprague et al., *To CoT or Not to CoT?* (2024), [2409.12183](https://arxiv.org/abs/2409.12183) | CoT helps mainly on math and symbolic reasoning, and little elsewhere |

## 6. Faithfulness and safety: does the CoT reflect what the model actually did?

| Paper | Finding |
|---|---|
| ⭐ **Turpin et al., *Language Models Don't Always Say What They Think*** (NeurIPS 2023), [2305.04388](https://arxiv.org/abs/2305.04388) | Biased prompts change answers, and the CoT never mentions the bias |
| Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning* (2023), [2307.13702](https://arxiv.org/abs/2307.13702) | Edit or cut the CoT and see whether the answer changes |
| Chen et al., *Reasoning Models Don't Always Say What They Think* (2025), [2505.05410](https://arxiv.org/abs/2505.05410) | Reasoning models often don't mention hints they used |
| Baker et al., *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* (2025), [2503.11926](https://arxiv.org/abs/2503.11926) | Penalising "bad thoughts" during training teaches models to hide them |
| Korbak et al., *Chain of Thought Monitorability* (2025), [2507.11473](https://arxiv.org/abs/2507.11473) | Reading the CoT is a useful but fragile safety tool, worth preserving |

## A reading path

1. **Wei 2022** and **Kojima 2022**: what CoT is.
2. **Self-Consistency**, **ReAct** and **Tree of Thoughts**: how prompting developed.
3. **STaR** and **Let's Verify Step by Step**: how models are trained to reason.
4. **Snell 2024** and **DeepSeek-R1**: today's reasoning models.
5. **Merrill & Sabharwal** for why it works, then **Turpin 2023** for its limits.
