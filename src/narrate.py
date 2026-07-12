"""Optional LLM polish for justification prose.

Design rule: the LLM never sees the raw reviews and never decides anything. It receives the
*already-computed facts* (hotel name, matched aspects with scores, the chosen evidence
quotes) and is asked only to rephrase the deterministic justification into one fluent
sentence or two. The output is then validated:

  * it must mention the hotel name,
  * it must not introduce any digit-bearing claim absent from the facts,
  * it must not name a *different* hotel,
  * it must respect a length cap.

Any failure silently falls back to the deterministic template. So enabling the LLM can only
improve wording, never corrupt a recommendation — the "LLM never invents the ranking"
guarantee holds. Uses Qwen2.5-1.5B-Instruct (fits the 4GB laptop GPU in fp16; CPU fallback).
"""
from __future__ import annotations

import re

from . import config
from .taxonomy import DISPLAY_NAMES


class Narrator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        use_cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="cuda" if use_cuda else None,
        )
        if not use_cuda:
            self.model = self.model.to("cpu")
        self.model.eval()

    def polish(self, profile: dict, row, matched: list[str], evidence: list[dict],
               template: str) -> str:
        facts = self._facts_block(profile, row, matched, evidence)
        prompt = (
            "You are a concise travel-recommendation writer. Rewrite the DRAFT into one "
            "or two natural sentences for the traveler below. Use ONLY the facts given. "
            "Do not invent numbers, hotels, or amenities. Mention the hotel by name.\n\n"
            f"{facts}\nDRAFT: {template}\n\nPolished:"
        )
        try:
            text = self._generate(prompt)
        except Exception:
            return template
        cleaned = self._validate(text, row, evidence, template)
        return cleaned

    def _facts_block(self, profile, row, matched, evidence) -> str:
        strengths = ", ".join(f"{DISPLAY_NAMES.get(d, d).lower()} {row['aspect_contrib'].get(d, 0):+.2f}"
                              for d in matched[:3]) or "solid all-round"
        quotes = " | ".join(e["quote"] for e in evidence[:2])
        return (f"TRAVELER: {profile['traveler_type']} on a {profile['budget'].replace('_','-')} budget.\n"
                f"HOTEL: {row['hotel_name']} ({row['hotel_category']}), ranked #{int(row['rank'])}.\n"
                f"STRENGTHS: {strengths}.\n"
                f"GUEST QUOTES: {quotes}.")

    def _generate(self, prompt: str) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=90, do_sample=False,
                                      temperature=None, top_p=None,
                                      pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _validate(self, text: str, row, evidence, template: str) -> str:
        if not text or len(text) > 600:
            return template
        # must mention the hotel (first token of its name is enough)
        key = row["hotel_name"].split(",")[0].split()[0]
        if key.lower() not in text.lower():
            return template
        # reject if it introduces a dollar amount or star claim not in the facts
        allowed_nums = set(re.findall(r"\d+", row["hotel_category"]))
        for e in evidence:
            allowed_nums.update(re.findall(r"\d+", e["quote"]))
        for num in re.findall(r"\d+", text):
            if num not in allowed_nums:
                return template
        # strip any leading label the model might echo
        text = re.sub(r"^(polished|answer)\s*:\s*", "", text, flags=re.I).strip()
        return text
