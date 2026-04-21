![alt text](img/2616.6.2235.png)

| Full version | [📰 𝕏](# "X.com Article link") | [📚 **SolveIT**](https://share.solve.it.com/d/ec8018951af13d01bc4dc8b03abb6663) | [Ⓜ️ **Markdown**](article.md "LLM-friendly input") | [🗒️ **Raw**](https://github.com/1iis/m01/raw/refs/heads/main/article.md "best with GET, wget, curl") |
| --- | --- | --- | --- | --- |
| **Abridged** | [📰 **𝕏**](https://x.com/1i__is/status/2046429247644791168 "X.com Abridged version") | [📚 **SolveIT**](https://share.solve.it.com/d/ab643991b1d68a22268ceeee6f4aa7d5) | [Ⓜ️ **Markdown**](abridged.md "LLM-friendly input") | [🗒️ **Raw**](https://github.com/1iis/m01/raw/refs/heads/main/abridged.md "best with GET, wget, curl")

# Dockerizing<br> SGLang + vLLM<br> on local RTX 3090

> **Mission 1: Foundations**  
> *Let's discover the basics of running fast local inference jobs!*

In this article, we implement a template to deploy two major AI GPU inference engines: [**SGLang**](https://www.sglang.io/) and [**vLLM**](https://vllm.ai/).

Read the [full](https://github.com/1iis/m01/blob/main/article.md) version (beginner-friendly),  
or the [abridged](https://github.com/1iis/m01/blob/main/abridged.md) (faster!) version.

---

## Overview

This is a typical server-client architecture:
- 🅰️ Server: [`docker-compose.yml`](https://github.com/1iis/m01/blob/main/docker-compose.yml) configures containers for SGLang and vLLM to run [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- 🅱️ Client:  
[`test_stream.py`](https://github.com/1iis/m01/blob/main/test_stream.py) queries the server with an image and text prompt.  
[`long_ctx.py`](https://github.com/1iis/m01/blob/main/long_ctx.py) stress-tests context length (KV cache) with one of the full [`books`](books) from Gutenberg.  

## Usage

Clone this repo.

1. Pick one:
   ```bash
   docker compose --profile sglang up -d   # SGLang
   docker compose --profile vllm up -d     # vLLM
   ```
   Those settings are good for a 24 GB local GPU. Modify them if needed (*how to* in the article).

2. Run inference:
   ```bash
   python test_stream.py                    # quick test
   python long_ctx.py books/dracula.txt     # pushing context
   ```
