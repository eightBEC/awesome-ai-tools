# Awesome AI Tools

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)](#contributing)

> An opinionated, curated, topic-oriented collection of AI tools, frameworks, and resources.
>
> If you find this useful, ⭐ star the repo, share and contribute!

---

## Table of Contents

- [Blogs / Newsletters](#blogs--newsletters)
- [Best Practices](#best-practices)
- [Prompt Engineering](#prompt-engineering)
- [LLM - Providers & Runtimes](#llm---providers--runtimes)
- [LLM - Training & Fine-Tuning Tools](#llm---training--fine-tuning-tools)
- [LLM - Serving & Inference Tools](#llm---serving--inference-tools)
- [Agents - General](#agents---general)
- [Agents - Coding](#agents---coding)
- [Agents - Protocols](#agents---standards--protocols)
- [Agents - Skills / Tools / MCP Servers](#agents---skills--tools--mcp-servers)
- [Agents - Ops / MLOps / Evaluation / Monitoring](#agents---ops--mlops--evaluation--monitoring)
- [Vector Databases](#vector-databases)
- [Misc - Popular AI Frameworks and Libraries](#misc---popular-ai-frameworks-and-libraries)
- [Misc - Starter Templates](#misc---starter-templates)
- [Courses & Learning](#courses--learning)
- [TODOs](#todos)
- [Contributing](#contributing)
- [License](#license)

---

## Blogs / Newsletters

Stay up-to-date with the latest AI news, research, and insights through curated newsletters and blogs.

- **Phil Schmid** - Latest in AI, practical hands-on tutorials <https://www.philschmid.de/>
- **The Batch (DeepLearning.AI)** — Weekly AI news and insights: <https://www.deeplearning.ai/the-batch/>
- **Import AI** — Weekly AI newsletter by Jack Clark: <https://importai.substack.com>
- **AlphaSignal** — 5-min email summary of latest news: <https://alphasignal.ai/>
- **Papers with Code** — Latest ML research with code: <https://paperswithcode.com>
- **Towards Data Science** — Medium publication on data science and AI: <https://towardsdatascience.com>
- **Niklas Hediloff's Blog** - Hands-on AI blog: <https://heidloff.net/>

---

## Best Practices

Comprehensive guides and best practices covering multiple AI topics, from fine-tuning to deployment strategies.

- **Fine-Tuning LLMs with Huggingface** - Transformers provides the Trainer API, which offers a comprehensive set of training features, for fine-tuning any of the models on the Hub:<https://huggingface.co/docs/transformers/training>
- **Reinforcement Learning** — A great video workshop about Reinforcement Learning: <https://www.youtube.com/watch?v=OkEGJ5G3foU>
- **Unsloth** - Fine-tuning LLMs Guide:<https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>

## Prompt Engineering

Tools and frameworks for crafting, testing, and optimizing prompts to get the best results from LLMs.

- **DSPy** — Programmatic prompting framework: <https://github.com/stanfordnlp/dspy>
- **LMQL** — Structured prompting language: <https://lmql.ai>
- **OpenAI Cookbook** — Patterns & examples for ChatGPT: <https://github.com/openai/openai-cookbook>
- **Claude Cookbook** - Patterns & examples for Claude: <https://github.com/anthropics/claude-cookbooks>
- **Promptfoo** — Prompt testing and evaluation: <https://promptfoo.dev>
- **IBM's Guide to Prompt Engineering** — Comprehensive guide with a curated collection of tools, tutorials and real-world examples: <https://www.ibm.com/think/prompt-engineering>

---

## LLM - Providers & Runtimes

Commercial and open-source platforms for accessing and running large language models, from cloud APIs to local deployment.

- **Anthropic Claude** — LLM platform: <https://www.anthropic.com>
- **Azure OpenAI Service** — Enterprise-grade GPT on Azure: <https://learn.microsoft.com/azure/ai-services/openai/>
- **Cohere** — LLM platform: <https://cohere.com>
- **ComfyUI** — Stable Diffusion pipelines/UI: <https://github.com/comfyanonymous/ComfyUI>
- **LM Studio** — Desktop LLM runner: <https://lmstudio.ai>
- **Meta Llama** — Open foundation models: <https://ai.meta.com/llama/>
- **Mistral AI** — Open-weight models: <https://mistral.ai>
- **Ollama** — Run LLMs locally: <https://ollama.com>
- **OpenAI Platform** — GPT APIs and models: <https://platform.openai.com>
- **Stable Diffusion** — Image generation models: <https://stability.ai>
- **watsonx API & SDKs** — Programmatic access and client libraries: <https://www.ibm.com/docs/en/watsonx>
- **watsonx.ai** — Build, tune, and deploy foundation models: <https://www.ibm.com/watsonx/ai>

---

## LLM - Training & Fine-Tuning Tools

Frameworks and libraries for training foundation models from scratch or fine-tuning pre-trained models for specific tasks.

- **axolotl** - A Free and Open Source LLM Fine-tuning Framework: <https://github.com/axolotl-ai-cloud/axolotl>
- **Huggingface PEFT** - PEFT: State-of-the-art Parameter-Efficient Fine-Tuning: <https://github.com/huggingface/peft>
- **Huggingface TRL** - Train transformer language models with reinforcement learning: <https://github.com/huggingface/trl>
- **LlamaFactory** - Unified Efficient Fine-Tuning of 100+ LLMs & VLMs: <https://github.com/hiyouga/LlamaFactory>
- **Ludwig** - Low-code framework for building custom LLMs, neural networks, and other AI models: <https://github.com/ludwig-ai/ludwig>
- **Unsloth** - Fine-tuning & Reinforcement Learning for LLMs. Train OpenAI gpt-oss, DeepSeek, Qwen, Llama, Gemma, TTS 2x faster with 70% less VRAM: <https://unsloth.ai/>
- **watsonx.ai Tuning Studio** - Tune a foundation model with the Tuning Studio to customize the model for your needs: <https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-tuning-studio.html?context=wx&locale=en>

## LLM - Serving & Inference Tools

High-performance inference engines and serving frameworks for deploying LLMs in production environments.

- **airllm** - AirLLM 70B inference with single 4GB GPU: <https://github.com/lyogavin/airllm>
- **CTranslate2** - Fast inference engine for Transformer models: <https://github.com/OpenNMT/CTranslate2>
- **mistral.rs** - Fast LLM inference framework: <https://github.com/EricLBuehler/mistral.rs>
- **Ray** — Distributed compute & model serving: <https://www.ray.io>
- **Text Embeddings Inference (TEI)** - A blazing fast inference solution for text embeddings models: <https://github.com/huggingface/text-embeddings-inference>
- **Text Generation Inference (TGI)** - Large Language Model Text Generation Inference: <https://github.com/huggingface/text-generation-inference>
- **vllm** - A high-throughput and memory-efficient inference and serving engine for LLMs: <https://github.com/vllm-project/vllm>
- **IBM watsonx.ai Bring Your Own Model - Software Hub (on-prem)** - You can upload and deploy a custom foundation model for use with watsonx.ai inferencing capabilities: <https://www.ibm.com/docs/en/software-hub/5.3.x?topic=setup-deploying-custom-foundation-models>

---

## Agents - General

Frameworks for building autonomous AI agents that can plan, reason, and execute complex multi-step tasks.

- **Agent Framework (Microsoft)** — A framework for building, orchestrating and deploying AI agents and multi-agent workflows with support for Python and .NET.: <https://github.com/microsoft/agent-framework>
- **CrewAI** — Agent collaboration framework: <https://www.crewai.com>
- **deepagents** - Deep Agents is an agent harness built on langchain and langgraph: <https://github.com/langchain-ai/deepagents>
- **Google Vertex Agent Builder** - Google's Agent Builder: <https://cloud.google.com/blog/products/ai-machine-learning/more-ways-to-build-and-scale-ai-agents-with-vertex-ai-agent-builder?hl=en>
- **LangGraph** — Agent graphs for LLMs: <https://langchain-ai.github.io/langgraph/>
- **OpenAI Agents** — Agent Framework: <https://platform.openai.com/docs/guides/agents>
- **watsonx Orchestrate** — Orchestration with watsonx.ai + enterprise tools: <https://www.ibm.com/products/watsonx-orchestrate>

---

## Agents - Coding

AI-powered coding assistants and agents that help with code generation, completion, debugging, and refactoring.

- **Aider** — AI pair programming in terminal: <https://aider.chat>
- **Claude Code** - AI code assistant: <https://www.claude.com/product/claude-code>
- **Continue** — Open-source AI code assistant: <https://continue.dev>
- **Cursor** — AI-first code editor: <https://cursor.sh>
- **GitHub Copilot** — AI pair programmer: <https://github.com/features/copilot>
- **OpenAI Codex** — AI code assistant: <https://openai.com/codex/>
- **OpenHands (formerly OpenDevin)** — AI-driven development: <https://github.com/All-Hands-AI/OpenHands>
- **Replit Agent** — AI agent for building apps: <https://replit.com/ai>
- **Tabnine** — AI code completion: <https://www.tabnine.com>
- **IBM Bob** — Enterprise AI software development partner: <https://www.ibm.com/products/watsonx-code-assistant>

---

## Agents - Standards & Protocols

Open standards and protocols enabling communication and interoperability between different AI agents and systems.

- **Agent Skills** - A simple, open format for giving agents new capabilities and expertise: <https://agentskills.io/home>
- **Agent2Agent (A2A)** - An open protocol enabling communication and interoperability between opaque agentic applications: <https://a2a-protocol.org/latest/>
- **Model Context Protocol (MCP)** - MCP is an open-source standard for connecting AI applications to external systems: <https://modelcontextprotocol.io/docs/getting-started/intro>
- **MCP for Beginners** - This open-source curriculum introduces the fundamentals of Model Context Protocol (MCP) through real-world, cross-language examples: <https://github.com/microsoft/mcp-for-beginners>
- **Agent Communication Protocol (ACP)** - The ACP is an open protocol for agent interoperability that solves the growing challenge of connecting AI agents, applications, and humans: <https://agentcommunicationprotocol.dev/introduction/welcome>

---

## Agents - Skills / Tools / MCP Servers

Reusable skills, tools, and Model Context Protocol (MCP) servers that extend agent capabilities with specialized functions and integrations.

- **Awesome MCP Servers** - A curated list of MCP servers: <https://github.com/appcypher/awesome-mcp-servers>
- **Anthropic Skills** - Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks: <https://github.com/anthropics/skills>
- **FastMCP** - Pythonic way to build MCP servers and clients: <https://github.com/jlowin/fastmcp>
- **Model Context Protocol (MCP)** - MCP is an open-source standard for connecting AI applications to external systems: <https://modelcontextprotocol.io/docs/getting-started/intro>
- **Context Forge MCP Gateway** - A Model Context Protocol (MCP) Gateway & Registry. Serves as a central management point for tools, resources, and prompts that can be accessed by MCP-compatible LLM applications: <https://github.com/IBM/mcp-context-forge>

---

## Agents - Ops / MLOps / Evaluation / Monitoring

Tools for observing, evaluating, testing, and monitoring AI agents and ML models in production environments.

- **Adversarial Robustness Toolbox (ART)** — Machine Learning Robustness testing (open source): <https://github.com/IBM/adversarial-robustness-toolbox>
- **Arize AI** — ML observability platform: <https://arize.com>
- **Deepchecks** — Testing ML models and data: <https://deepchecks.com>
- **DeepEval** - LLM Evaluation Framework: <https://github.com/confident-ai/deepeval>
- **Evidently AI** — Monitoring & data drift detection: <https://www.evidentlyai.com>
- **Giskard** — LLM & ML testing framework: <https://giskard.ai>
- **Kubeflow** — Kubernetes ML pipelines: <https://www.kubeflow.org>
- **Langfuse** - LLM Observability, metrics, evals, prompt management, playground, datasets: <https://github.com/langfuse/langfuse>
- **MLflow** — Experiment tracking & model registry: <https://mlflow.org>
- **Opik** - Debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows: <https://github.com/comet-ml/opik>
- **Phoenix (Arize)** — Open-source eval & tracing: <https://github.com/Arize-ai/phoenix>
- **Weights & Biases (W&B)** — Experiment tracking and collaboration: <https://wandb.ai>

---

## Vector Databases

Specialized databases optimized for storing and querying high-dimensional vector embeddings for semantic search and RAG applications.

- **Chroma** — Lightweight vector store: <https://www.trychroma.com>
- **Elasticsearch** — Distributed search & analytics with vector support: <https://www.elastic.co/elasticsearch>
- **FAISS** — Vector search library by Meta: <https://github.com/facebookresearch/faiss>
- **OpenSearch** — Open-source search & analytics: <https://opensearch.org>
- **Pinecone** — Managed vector database: <https://www.pinecone.io>
- **Qdrant** — Open-source vector database: <https://qdrant.tech>
- **Weaviate** — Open-source/managed vector database: <https://weaviate.io>

---

## Misc - Popular AI Frameworks and Libraries

Essential AI frameworks and libraries commonly used across the industry for building ML and AI applications.

- **LangChain** — LLM application framework: <https://python.langchain.com>
- **Hugging Face - Datasets** - The largest data library: <https://huggingface.co/datasets>
- **Hugging Face - Models** - The largest model library: <https://huggingface.co/models>
- **Hugging Face - Transformers** — LLMs & NLP models: <https://huggingface.co/docs/transformers>
- **Streamlit** — Data apps made easy: <https://streamlit.io>
- **Gradio** — Web UIs for ML demos: <https://gradio.app>

---

## Misc - Starter Templates

Production-ready project templates and boilerplates to kickstart your AI application development.

- **Full Stack FastAPI Template** — Production-ready FastAPI app template: <https://github.com/fastapi/full-stack-fastapi-template>
- **RAG Starters (LangChain)** — Retrieval-augmented generation templates: <https://python.langchain.com/docs/use_cases/question_answering/>
- **OpenRAG** - IBM's open-source RAG distribution, powered by OpenSearch, Langflow, and Docling: <https://www.openr.ag/>

---

## Courses & Learning

Educational resources, courses, and tutorials for learning AI, machine learning, and deep learning fundamentals.

- **Andrew Ng — Machine Learning (Coursera)** — Foundational ML course: <https://www.coursera.org/learn/machine-learning>
- **DeepLearning.AI — GenAI & ML courses** — Specialized AI courses: <https://www.deeplearning.ai>
- **fast.ai — Practical Deep Learning** — Hands-on deep learning: <https://course.fast.ai>
- **Stanford CS224N (NLP) incl. Slides** — Natural language processing course: <http://web.stanford.edu/class/cs224n/>
- **Coursera: IBM - AI Developer** - Coursera IBM AI Course: <https://www.coursera.org/professional-certificates/applied-artifical-intelligence-ibm-watson-ai>
- **Coursera: IBM - Building AI Agents** - Building AI Agents & Workflows: <https://www.coursera.org/specializations/building-ai-agents-and-agentic-workflows>
- **IBM Developer** — Tutorials & code patterns: <https://developer.ibm.com>

---

## TODOs

[ ] - Add `Memory` section
[ ] - Add `Graph` section

## Contributing

We welcome contributions of tools, tutorials, and examples! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

### Quick Guidelines

- Add items under the appropriate **topic** subsection.
- Use the format: `**Name** — concise one-line description: URL`.
- Prefer links to **official docs** or **canonical GitHub**.
- Keep descriptions neutral (no marketing language) and avoid duplicates.
- Add reference to table of contents if adding a new section.

---

## License

Distributed under the **Apache Version 2.0**. See [`LICENSE`](LICENSE) for details.
