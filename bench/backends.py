"""Model backends. All return the same record shape:

{content, reasoning, finish_reason, input_tokens, output_tokens, latency_s, error}

`content` is the FINAL answer text only; reasoning is kept separately for error analysis.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time


def _empty(error=None, **kw):
    rec = {"content": None, "reasoning": None, "finish_reason": None, "input_tokens": 0,
           "output_tokens": 0, "latency_s": 0.0, "error": error}
    rec.update(kw)
    return rec


async def _retry(fn, attempts=6, base=4.0):
    for a in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            retryable = any(s in msg.lower() for s in (
                "rate", "429", "overloaded", "529", "500", "502", "503", "504", "timeout",
                "connection", "temporarily"))
            if not retryable or a == attempts - 1:
                raise
            await asyncio.sleep(base * 2 ** a + random.random() * 2)


class OpenAICompatBackend:
    """vLLM / SGLang / LM Studio / any OpenAI-compatible /v1/chat/completions server."""

    def __init__(self, cfg):
        from openai import AsyncOpenAI
        self.cfg = cfg
        self.client = AsyncOpenAI(base_url=cfg["base_url"],
                                  api_key=os.environ.get(cfg.get("api_key_env", ""), "EMPTY"),
                                  timeout=cfg.get("timeout_s", 7200), max_retries=0)
        self.concurrency = cfg.get("concurrency", 64)

    async def generate(self, item, sample_idx):
        p = self.cfg.get("params", {})
        messages = []
        if self.cfg.get("system_prompt"):
            messages.append({"role": "system", "content": self.cfg["system_prompt"]})
        messages.append({"role": "user", "content": item["prompt"]})
        seed = int(hashlib.sha256(f"{item['id']}-{sample_idx}".encode()).hexdigest()[:8], 16)
        kwargs = dict(model=self.cfg["model_id"], messages=messages,
                      temperature=p.get("temperature", 0.6), top_p=p.get("top_p", 0.95),
                      max_tokens=p.get("max_tokens", 32768), seed=seed)
        extra = dict(p.get("extra_body", {}))
        t0 = time.time()
        try:
            r = await _retry(lambda: self.client.chat.completions.create(**kwargs, extra_body=extra or None))
        except Exception as e:  # noqa: BLE001
            return _empty(error=f"{type(e).__name__}: {e}"[:500], latency_s=time.time() - t0)
        ch = r.choices[0]
        msg = ch.message
        extra_fields = getattr(msg, "model_extra", None) or {}
        reasoning = extra_fields.get("reasoning_content") or extra_fields.get("reasoning")
        return {"content": msg.content, "reasoning": reasoning, "finish_reason": ch.finish_reason,
                "input_tokens": r.usage.prompt_tokens if r.usage else 0,
                "output_tokens": r.usage.completion_tokens if r.usage else 0,
                "latency_s": time.time() - t0, "error": None}


class MockBackend:
    """Offline backend for dry runs: answers correctly with probability `accuracy`."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.concurrency = cfg.get("concurrency", 16)
        self.acc = cfg.get("accuracy", 0.7)

    async def generate(self, item, sample_idx):
        rng = random.Random(f"{self.cfg['name']}-{item['id']}-{sample_idx}")
        await asyncio.sleep(0.001)
        good = rng.random() < self.acc
        g, gold = item["grader"], item["gold"]
        finish = "stop"
        if rng.random() < 0.03:
            return _empty(content="Let me keep thinking", finish_reason="length",
                          output_tokens=self.cfg.get("max_tokens", 32000), latency_s=1.0)
        if g in ("math", "exact"):
            content = f"So the answer is $\\boxed{{{gold if good else '12345'}}}$."
        elif g == "mcq":
            content = f"Reasoning...\nAnswer: {gold if good else 'ABCD'[('ABCD'.index(gold) + 1) % 4]}"
        elif g == "zebra":
            import json
            hdr, rows = gold["header"], gold["rows"]
            sol = {f"House {i + 1}": {hdr[j]: (rows[i][j] if good or j > 1 else "??")
                                     for j in range(1, len(hdr))} for i in range(len(rows))}
            content = "```json\n" + json.dumps({"solution": sol}) + "\n```"
        else:  # code
            ref = item.get("meta", {}).get("mock_solution", "print('hi')")
            content = "```python\n" + (ref if good else "print(0)") + "\n```"
        n_out = rng.randint(800, 9000) * (2 if not good else 1)
        return {"content": content, "reasoning": "<mock reasoning>", "finish_reason": finish,
                "input_tokens": len(item["prompt"]) // 4, "output_tokens": n_out,
                "latency_s": n_out / self.cfg.get("tok_per_s", 60), "error": None}


def make_backend(cfg):
    kind = cfg["backend"]
    return {"openai": OpenAICompatBackend, "mock": MockBackend}[kind](cfg)
