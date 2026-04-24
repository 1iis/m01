![alt text](img/2616.6.2235.png)

| Full version | [📰 𝕏](https://x.com/1i__is/status/2045790740584735038 "X.com Article link") | [📚 **SolveIT**](https://share.solve.it.com/d/ec8018951af13d01bc4dc8b03abb6663 "solve.it.com dialog (article.ipynb notebook)") | [Ⓜ️ **Markdown**](article.md "article.md rendered, with images") | [🗒️ **`raw`**](https://github.com/1iis/m01/raw/refs/heads/main/article.md "raw Markdown, best for AI LLM agents: GET, wget, curl, read_url()…") |
| --- | --- | --- | --- | --- |
| **Abridged** | [📰 **𝕏**](https://x.com/1i__is/status/2046429247644791168 "X.com Abridged link") | [📚 **SolveIT**](https://share.solve.it.com/d/ab643991b1d68a22268ceeee6f4aa7d5 "solve.it.com dialog (abridged.ipynb notebook)") | [Ⓜ️ **Markdown**](abridged.md "abridged.md rendered, with images") | [🗒️ **`raw`**](https://github.com/1iis/m01/raw/refs/heads/main/abridged.md "raw Markdown, best for AI LLM agents: GET, wget, curl, read_url()…")

# Dockerizing<br> SGLang + vLLM<br> on local RTX 3090

> **Mission 1: Foundations**  
> *Let's discover the basics of running fast local inference jobs!*

In this article, we implement a template to deploy two major AI GPU inference engines: [**SGLang**](https://www.sglang.io/) and [**vLLM**](https://vllm.ai/).

Read the **[full](article.md) version** (beginner-friendly),  
or (faster!) the **[abridged](abridged.md)** **version**.  
Or just read below if you need no explaining.  

> [!TIP]
> Use `raw` versions of Markdown to feed LLM/agents (links at the top).

## Feeback & contribution

> Don't hesitate! 👋

- [Create](https://github.com/1iis/m01/issues/new) a new [issue](https://github.com/1iis/m01/issues) (even just to talk)
- Reply to the [article on X](https://x.com/1i__is/status/2046429247644791168)
- Feed free to [DM me! `@1i__is`](https://x.com/1i__is)

---

## Overview

This is a typical server-client architecture:
- 🅰️ Server: [**`docker-compose.yml`**](docker-compose.yml) configures containers for SGLang and vLLM to run [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- 🅱️ Client:  
[**`test_stream.py`**](test_stream.py) queries the server with an image and text prompt.  
[**`long_ctx.py`**](long_ctx.py) stress-tests context length (KV cache) with one of the full [**`books`**](books) from Gutenberg.  

## Usage

After cloning this repo, you:
1. Build the docker container (SGLang or vLLM)
2. Set environment variables
3. Prompt the model through `curl` then Python scripts (you may need to install `openai` package)

### 1. Docker

Pick one:
```bash
docker compose --profile sglang up -d   # SGLang
docker compose --profile vllm up -d     # vLLM
```
Those settings are good for a 24 GB local GPU. Modify them if needed (*how to* in the article).

> [!TIP]
> Alternatively, same thing as above:
> ```bash
> export COMPOSE_PROFILES=sglang     # or =vllm
> docker compose up -d               # --profile sglang not needed!
> docker compose down
> ```

### 2. Env vars

Define environment variables for the openai-compatible API.
```bash
export OPENAI_API_KEY="EMPTY"

export OPENAI_BASE_URL="http://localhost:8001/v1"   # SGLang
export OPENAI_BASE_URL="http://localhost:8002/v1"   # vLLM
```

### 3. Prompt

First let's `curl` for a quick spin:
```
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "What is the meaning of life, the universe, and everything?"}],
    "temperature": 0.7,
    "max_tokens": 4096
  }' | jq .
```

Then more advanced tests:
```bash
python test_stream.py                    # vision + test
python long_ctx.py books/dracula.txt     # pushing context length
```

> [!TIP]
> If you get this error:
> 
> ```
> Traceback (most recent call last):
>   File "/home/kit/data/1/org/1iis/ai/Lab0/art/m01/test_stream.py", line 1, in <module>
>     from openai import OpenAI
> ModuleNotFoundError: No module named 'openai'
> ```
> 
> Then proceed to install that package.  
> Optionally, first make a virtual environment to keep your host clean.
> ```bash
> # venv creation & activation
> python -m venv .venv && source .venv/bin/activate
> 
> # Pick one:
> pip install -e .        # pip: install deps listed in pyproject.toml
> uv pip install -e .     # uv: same
> uv pip install openai   # install openai package directly
> ```
