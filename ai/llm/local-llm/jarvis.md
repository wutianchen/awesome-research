# OpenJarvis

Notes as of 2026-11-09. Descriptions come from each project's GitHub page; none of these tools have been installed or run for these notes.

## What is it

[OpenJarvis](https://github.com/open-jarvis/OpenJarvis) — "Personal AI, On Personal Devices". A framework for local-first personal AI agents.

* **Origin:** comes out of Stanford — [project site](https://openjarvis.stanford.edu/), [paper page](https://scalingintelligence.stanford.edu/pubs/openjarvis/), [arXiv](https://arxiv.org/abs/2605.17172). The code lives under the separate `open-jarvis` GitHub organization
* **License / stack:** Apache-2.0, Python ≥ 3.10, about 14k stars
* **Why it exists:** most personal AI agents still route intelligence through cloud APIs; OpenJarvis aims to run them on your own device

Three core ideas:

* shared primitives for building on-device agents
* evaluations that treat energy, FLOPs, latency, and dollar cost as first-class constraints alongside accuracy
* a learning loop that improves models using local trace data

Usage: the installer sets up uv, a Python venv, Ollama, and a starter model; then run `jarvis` (CLI) or `jarvis gui`, or use the desktop app.

| Preset | What it does |
|---|---|
| `morning-digest-*` | spoken daily briefing from email, calendar, health, news |
| `deep-research` | multi-hop research across indexed docs with citations |
| `code-assistant` | agent with code execution, file I/O, and shell access |
| `scheduled-monitor` | stateful agent on a schedule with memory |
| `chat-simple` | lightweight conversation, no tools |

Where it fits in the [layers](README.md#layers-of-using-llm): layer 2 (framework), mainly for non-code agents. It also ships a finished CLI and desktop app, which is closer to layer 4.

## Similar tools

Star counts as of 2026-11-09.

### Closest: personal agents that run on your machine

The OpenJarvis paper names OpenClaw and Hermes Agent as the cloud-routed personal AI stacks it reacts to.

| Tool | Stars | License | What it is |
|---|---|---|---|
| [OpenClaw](https://github.com/openclaw/openclaw) | 394k | MIT | assistant reached through Slack, WhatsApp, iMessage, and 20+ other channels; state and credentials stay on your hardware; local or cloud models are swappable plugins |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | 262k | MIT | creates skills from experience and builds memory of you over time; nearest match to the learning loop, but works with any model provider rather than focusing on local ones |
| [Khoj](https://github.com/khoj-ai/khoj) | 38k | AGPL-3.0 | self-hostable "second brain" with custom agents, scheduled automations, and deep research on online or local models; last push August 2026 |

### Partial overlap: local AI stacks and apps

| Tool | Stars | License | What it is |
|---|---|---|---|
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | 68k | MIT | all-in-one, local-first agent app |
| [LocalAI](https://github.com/mudler/LocalAI) | 50k | MIT | engine for running models on your own hardware; fits below OpenJarvis as the inference layer, though it also has agent features |
| [Open WebUI](https://github.com/open-webui/open-webui) | 158k | custom | chat interface for Ollama and OpenAI-compatible APIs; more a front end than an agent framework |

### Adjacent: for building rather than using

| Tool | Stars | License | What it is |
|---|---|---|---|
| [Letta](https://github.com/letta-ai/letta) | 25k | Apache-2.0 | platform for stateful agents with long-term memory |
| [goose](https://github.com/aaif-goose/goose) | 56k | Apache-2.0 | extensible agent for any model; leans toward coding |

### What sets OpenJarvis apart

Treating energy, latency, and dollar cost as first-class evaluation metrics. None of the projects above mention this in their descriptions, so none is a direct replacement on that front. For a capable personal assistant, start with OpenClaw, or Hermes Agent if learning over time matters more.
