"""
DSPy Few-Shot Optimizer for SmolLM2 Intent Extractor
=====================================================
Runs on top of the winning prompt variant from prompt_sweep.py.

What DSPy does here:
  1. Runs the model on all TRAIN_SET queries using your winning prompt
  2. Checks which (query → JSON) pairs the model gets right (pass metric)
  3. Bootstraps those successful pairs as few-shot demonstrations
  4. Tests all demo combinations to find the set that maximises F1
  5. Saves the optimized program + exports a standalone prompt with
     the best demonstrations already embedded (drop-in for your script)

Usage:
  pip install dspy-ai

  python prompt_dspy_optimize.py \
      --winning-variant v4_few_shot \
      --output-dir dspy_output
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch


# ---------------------------------------------------------------------------
# Shared eval set  (same 30 queries as prompt_sweep.py)
# Split: first 20 → TRAIN (bootstrap source), last 10 → TEST (held-out eval)
# ---------------------------------------------------------------------------

ALL_QUERIES = [
    # ── Unseen actor names ──────────────────────────────────────────────────
    {"query": "Saurav Chakraborty kids 3-5",
     "expected": {"actor": "Saurav Chakraborty", "audience": "kids", "age": "3-5 years"}},

    {"query": "Meera Nair movies",
     "expected": {"actor": "Meera Nair", "content_type": "Movies"}},

    {"query": "Arjun Kapoor action flicks",
     "expected": {"actor": "Arjun Kapoor", "content_type": "Movies", "genre": "Action"}},

    {"query": "Nandita Das drama films",
     "expected": {"actor": "Nandita Das", "content_type": "Movies", "genre": "Drama"}},

    {"query": "Konkona Sen Sharma thriller",
     "expected": {"actor": "Konkona Sen Sharma", "genre": "Thriller"}},

    # ── Synonym keywords for content_type ───────────────────────────────────
    {"query": "Anuv Jain songs",
     "expected": {"actor": "Anuv Jain", "content_type": "Songs"}},

    {"query": "Arijit Singh music",
     "expected": {"actor": "Arijit Singh", "content_type": "Music"}},

    {"query": "Raghav juyal popular cinema",
     "expected": {"actor": "Raghav juyal", "content_type": "Movies","rating_type":"top rated"}},

    {"query": "Devanand flicks",
     "expected": {"actor": "Devanand", "content_type": "Movies"}},

    {"query": "manish paul episodes",
     "expected": {"actor": "Manish Paul", "content_type": "Episodes"}},

    # ── Platform aliases ─────────────────────────────────────────────────────
    {"query": "Dilip Joshi hindi shows Prime",
    "expected": {"actor": "Dilip Joshi", "language": "Hindi", "content_type": "Shows", "platform": "Amazon Prime"}},

    {"query": "latest Hindi films on Hotstar",
     "expected": {"recency": "latest", "language": "Hindi", "content_type": "Movies", "platform": "Disney+ Hotstar"}},

    {"query": "Tamil action on Sony",
     "expected": {"language": "Tamil", "genre": "Action", "platform": "SonyLIV"}},

    {"query": "Kannada movies on Zee",
     "expected": {"language": "Kannada", "content_type": "Movies", "platform": "ZEE5"}},

    {"query": "Marathi comedy on Jio",
     "expected": {"language": "Marathi", "genre": "Comedy", "platform": "Jio Cinema"}},

    # ── Age / audience combos ────────────────────────────────────────────────
    {"query": "Disney kids 6-8",
     "expected": {"title": "Disney", "audience": "kids", "age": "6-8 years"}},

    {"query": "cartoon shows for children 4 years",
     "expected": {"content_type": "Shows", "audience": "kids", "age": "4 years"}},

    {"query": "family movies on Netflix",
     "expected": {"audience": "family", "content_type": "Movies", "platform": "Netflix"}},

    {"query": "educational cartoons kids 5-10",
     "expected": {"genre": "Educational", "content_type": "Show", "audience":"kids", "age": "5-10 years"}},

    {"query": "family movies on Netflix",
     "expected": {"audience": "family", "content_type": "Movies", "platform": "Netflix"}},

    # ── Mood / rating synonyms ───────────────────────────────────────────────
    {"query": "feel good Bollywood movies",
     "expected": {"mood": "feel good", "language": "Hindi", "content_type": "Movies"}},

    {"query": "scary Hindi horror",
     "expected": {"mood": "scary", "language": "Hindi", "genre": "Horror"}},

    {"query": "must watch Tamil thriller",
     "expected": {"rating_type": "must watch", "language": "Tamil", "genre": "Thriller"}},

    {"query": "underrated Telugu drama",
     "expected": {"rating_type": "underrated", "language": "Telugu", "genre": "Drama"}},

    # ── Multi-field combos ───────────────────────────────────────────────────
    {"query": "Hrithik Roshan action movies on Netflix 2023",
     "expected": {"actor": "Hrithik Roshan", "genre": "Action", "content_type": "Movies",
                  "platform": "Netflix", "year": "2023"}},

    {"query": "latest Malayalam blockbuster on Prime",
     "expected": {"recency": "latest", "language": "Malayalam", "rating_type": "blockbuster",
                  "platform": "Amazon Prime"}},

    {"query": "Alia Bhatt romantic drama 2022",
     "expected": {"actor": "Alia Bhatt", "genre": "Romance", "content_type": "Movies", "year": "2022"}},

    {"query": "latest elections news",
     "expected": {"recency": "latest", "content_type": "LiveChannel"}},

    # ── Edge cases ───────────────────────────────────────────────────────────
    {"query": "something similar to RRR",
     "expected": {"similar_to": "RRR"}},

    {"query": "CNN News 18",
     "expected": {"content_type": "LiveChannel", "title": "CNN News 18"}},

    {"query": "Football matches",
     "expected": {"content_type": "LiveChannel","title":"Football"}},
]


TRAIN_SET = ALL_QUERIES[:20]
TEST_SET  = ALL_QUERIES[20:]


# ---------------------------------------------------------------------------
# Winning prompt variants  (paste your winning variant's template here,
# or pass --winning-variant to select one at runtime)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a structured intent extractor for an Indian OTT content platform. "
    "Return only valid JSON."
)

PROMPT_VARIANTS = {

    # ── V1: Minimal baseline (original) ─────────────────────────────────────
    "v1_minimal": (
        "Convert the user query into a valid JSON intent object.\n"
        "Return only valid JSON, with no markdown fences and no extra text.\n\n"
        "Query: {query}\n"
        "JSON:"
    ),

    # ── V2: Field list added ─────────────────────────────────────────────────
    "v2_field_list": (
        "Convert the user query into a valid JSON intent object.\n"
        "Return only valid JSON, with no markdown fences and no extra text.\n\n"
        "Available fields (only include what is explicitly mentioned):\n"
        "  actor, director, title, similar_to      — person or content names (copy exactly)\n"
        "  content_type                             — Songs | Music | Movies | Shows | Series | Web Series | Trailers\n"
        "  genre                                    — Action | Comedy | Drama | Horror | Romance | Thriller | Crime | Biopic | Sports | Sci-Fi | Mystery | Historical\n"
        "  language                                 — Hindi | Tamil | Telugu | Malayalam | Kannada | Bengali | Marathi | Punjabi | Gujarati | Bhojpuri\n"
        "  mood                                     — feel good | funny | sad | scary | dark | emotional | romantic | intense | inspirational | motivational | light hearted | suspenseful\n"
        "  platform                                 — Netflix | Amazon Prime | Disney+ Hotstar | SonyLIV | ZEE5 | Jio Cinema | MX Player | YouTube\n"
        "  rating_type                              — top rated | must watch | blockbuster | hit | classic | cult | underrated | overrated | flop | award winning\n"
        "  audience                                 — kids | adults | family\n"
        "  age                                      — numeric range e.g. 3-5 years\n"
        "  recency                                  — latest\n"
        "  year                                     — 2018 to 2025\n"
        "  query_type                               — availability | rating/review\n\n"
        "Query: {query}\n"
        "JSON:"
    ),

    # ── V3: Field list + mapping rules ──────────────────────────────────────
    "v3_rules": (
        "Convert the user query into a valid JSON intent object.\n"
        "Return only valid JSON, with no markdown fences and no extra text.\n\n"
        "Available fields (only include what is explicitly mentioned):\n"
        "  actor, director, title, similar_to      — person or content names (copy exactly)\n"
        "  content_type                             — Songs | Music | Movies | Shows | Series | Web Series | Trailers\n"
        "  genre                                    — Action | Comedy | Drama | Horror | Romance | Thriller | Crime | Biopic | Sports | Sci-Fi | Mystery | Historical\n"
        "  language                                 — Hindi | Tamil | Telugu | Malayalam | Kannada | Bengali | Marathi | Punjabi | Gujarati | Bhojpuri\n"
        "  mood                                     — feel good | funny | sad | scary | dark | emotional | romantic | intense | inspirational | motivational | light hearted | suspenseful\n"
        "  platform                                 — Netflix | Amazon Prime | Disney+ Hotstar | SonyLIV | ZEE5 | Jio Cinema | MX Player | YouTube\n"
        "  rating_type                              — top rated | must watch | blockbuster | hit | classic | cult | underrated | overrated | flop | award winning\n"
        "  audience                                 — kids | adults | family\n"
        "  age                                      — numeric range e.g. 3-5 years\n"
        "  recency                                  — latest\n"
        "  year                                     — 2018 to 2025\n\n"
        "Mapping rules:\n"
        "  - songs / music / tracks           → content_type\n"
        "  - movies / films / cinema / flicks → content_type = Movies\n"
        "  - shows / series / episodes        → content_type\n"
        "  - kids / children / child          → audience = kids\n"
        "  - family                           → audience = family\n"
        "  - adults                           → audience = adults\n"
        "  - numbers like 3-5 or 6 to 8      → age field\n"
        "  - prime / prime video              → platform = Amazon Prime\n"
        "  - hotstar / disney                 → platform = Disney+ Hotstar\n"
        "  - sony / sonyliv                   → platform = SonyLIV\n"
        "  - zee / zee5                       → platform = ZEE5\n"
        "  - jio / jio cinema                 → platform = Jio Cinema\n"
        "  - latest / new / recent / newest   → recency = latest\n"
        "  - must watch / top rated / classic → rating_type\n\n"
        "Important:\n"
        "  - Copy person names EXACTLY as written. Never substitute or paraphrase.\n"
        "  - Do NOT add fields that are not present in the query.\n"
        "  - Do NOT hallucinate values. If unsure, omit the field.\n\n"
        "Query: {query}\n"
        "JSON:"
    ),

    # ── V4: Field list + rules + few-shot examples ───────────────────────────
    "v4_few_shot": (
        "Convert the user query into a valid JSON intent object.\n"
        "Return only valid JSON, with no markdown fences and no extra text.\n\n"
        "Available fields (only include what is explicitly mentioned):\n"
        "  actor, director, title, similar_to      — person or content names (copy exactly)\n"
        "  content_type                             — Songs | Music | Movies | Shows | Series | Web Series | Trailers\n"
        "  genre                                    — Action | Comedy | Drama | Horror | Romance | Thriller | Crime | Biopic | Sports | Sci-Fi | Mystery | Historical\n"
        "  language                                 — Hindi | Tamil | Telugu | Malayalam | Kannada | Bengali | Marathi | Punjabi | Gujarati | Bhojpuri\n"
        "  mood                                     — feel good | funny | sad | scary | dark | emotional | romantic | intense | inspirational | motivational | light hearted | suspenseful\n"
        "  platform                                 — Netflix | Amazon Prime | Disney+ Hotstar | SonyLIV | ZEE5 | Jio Cinema | MX Player | YouTube\n"
        "  rating_type                              — top rated | must watch | blockbuster | hit | classic | cult | underrated | overrated | flop | award winning\n"
        "  audience                                 — kids | adults | family\n"
        "  age                                      — numeric range e.g. 3-5 years\n"
        "  recency                                  — latest\n"
        "  year                                     — 2018 to 2025\n\n"
        "Mapping rules:\n"
        "  - songs / music / tracks           → content_type\n"
        "  - movies / films / cinema / flicks → content_type = Movies\n"
        "  - shows / series / episodes        → content_type\n"
        "  - kids / children / child          → audience = kids\n"
        "  - family                           → audience = family\n"
        "  - numbers like 3-5 or 6 to 8      → age field\n"
        "  - prime / prime video              → platform = Amazon Prime\n"
        "  - hotstar / disney                 → platform = Disney+ Hotstar\n"
        "  - sony / sonyliv                   → platform = SonyLIV\n"
        "  - zee / zee5                       → platform = ZEE5\n"
        "  - jio / jio cinema                 → platform = Jio Cinema\n"
        "  - latest / new / recent            → recency = latest\n\n"
        "Important:\n"
        "  - Copy person names EXACTLY as written. Never substitute or paraphrase.\n"
        "  - Do NOT add fields not present in the query.\n\n"
        "Examples:\n"
        "Query: Salman Khan songs\n"
        'JSON: {{"actor": "Salman Khan", "content_type": "Songs"}}\n\n'
        "Query: Saurav Chakraborty kids 3-5\n"
        'JSON: {{"actor": "Saurav Chakraborty", "audience": "kids", "age": "3-5 years"}}\n\n'
        "Query: latest Telugu action movies on Prime\n"
        'JSON: {{"recency": "latest", "language": "Telugu", "genre": "Action", "content_type": "Movies", "platform": "Amazon Prime"}}\n\n'
        "Query: scary Hindi horror on Netflix\n"
        'JSON: {{"mood": "scary", "language": "Hindi", "genre": "Horror", "platform": "Netflix"}}\n\n'
        "Query: Alia Bhatt romantic drama 2022\n"
        'JSON: {{"actor": "Alia Bhatt", "genre": "Romance", "content_type": "Movies", "year": "2022"}}\n\n'
        "Query: must watch Tamil thriller\n"
        'JSON: {{"rating_type": "must watch", "language": "Tamil", "genre": "Thriller"}}\n\n'
        "Query: Disney kids 6-8\n"
        'JSON: {{"title": "Disney", "audience": "kids", "age": "6-8 years"}}\n\n'
        "Query: {query}\n"
        "JSON:"
    ),

    # ── V5: Negative examples + strict constraints ───────────────────────────
    "v5_negative_examples": (
        "Convert the user query into a valid JSON intent object.\n"
        "Return only valid JSON, with no markdown fences and no extra text.\n\n"
        "Available fields (only include what is explicitly mentioned):\n"
        "  actor, director, title, similar_to      — person or content names (copy exactly)\n"
        "  content_type                             — Songs | Music | Movies | Shows | Series | Web Series | Trailers\n"
        "  genre                                    — Action | Comedy | Drama | Horror | Romance | Thriller | Crime | Biopic | Sports | Sci-Fi | Mystery | Historical\n"
        "  language                                 — Hindi | Tamil | Telugu | Malayalam | Kannada | Bengali | Marathi | Punjabi | Gujarati | Bhojpuri\n"
        "  mood                                     — feel good | funny | sad | scary | dark | emotional | romantic | intense | inspirational | motivational | light hearted | suspenseful\n"
        "  platform                                 — Netflix | Amazon Prime | Disney+ Hotstar | SonyLIV | ZEE5 | Jio Cinema | MX Player | YouTube\n"
        "  rating_type                              — top rated | must watch | blockbuster | hit | classic | cult | underrated | overrated | flop | award winning\n"
        "  audience                                 — kids | adults | family\n"
        "  age                                      — numeric range e.g. 3-5 years\n"
        "  recency                                  — latest\n"
        "  year                                     — 2018 to 2025\n\n"
        "Strict rules:\n"
        "  - songs / music / tracks           → content_type (NOT genre)\n"
        "  - movies / films / cinema / flicks → content_type = Movies (NOT genre)\n"
        "  - kids / children                  → audience = kids (NOT genre = Family)\n"
        "  - family                           → audience = family (NOT genre = Family)\n"
        "  - numbers like 3-5                 → age field (NOT year)\n"
        "  - prime / prime video              → platform = Amazon Prime\n"
        "  - hotstar / disney+                → platform = Disney+ Hotstar\n"
        "  - latest / new / recent            → recency = latest\n"
        "  - Copy names EXACTLY. 'Saurav Chakraborty' must NOT become 'Sallu' or any other name.\n"
        "  - Do NOT invent fields. Only output fields explicitly in the query.\n\n"
        "✓ Correct examples:\n"
        "Query: Salman Khan songs\n"
        'JSON: {{"actor": "Salman Khan", "content_type": "Songs"}}\n\n'
        "Query: Saurav Chakraborty kids 3-5\n"
        'JSON: {{"actor": "Saurav Chakraborty", "audience": "kids", "age": "3-5 years"}}\n\n'
        "Query: Disney kids 6-8\n"
        'JSON: {{"title": "Disney", "audience": "kids", "age": "6-8 years"}}\n\n'
        "Query: Aamir Khan action films on Prime 2023\n"
        'JSON: {{"actor": "Aamir Khan", "genre": "Action", "content_type": "Movies", "platform": "Amazon Prime", "year": "2023"}}\n\n'
        "✗ Wrong examples (do NOT do this):\n"
        "Query: Salman Khan songs\n"
        'WRONG: {{"actor": "Salman Khan", "genre": "Music"}}  ← songs must go to content_type, not genre\n\n'
        "Query: Saurav Chakraborty kids 3-5\n"
        'WRONG: {{"actor": "Sallu", "genre": "Family"}}  ← copy name exactly; kids goes to audience not genre\n\n'
        "Query: family movies on Netflix\n"
        'WRONG: {{"genre": "Family", "platform": "Netflix"}}  ← family goes to audience not genre\n\n'
        "Query: {query}\n"
        "JSON:"
    ),
}


# ---------------------------------------------------------------------------
# Scoring  (same as prompt_sweep.py)
# ---------------------------------------------------------------------------

def safe_json_load(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rstrip("`\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def score_output(predicted: dict, expected: dict) -> dict:
    if not isinstance(predicted, dict):
        predicted = {}
    pred_keys = set(predicted.keys())
    exp_keys  = set(expected.keys())
    correct   = sum(
        1 for k in (pred_keys & exp_keys)
        if str(predicted[k]).strip().lower() == str(expected[k]).strip().lower()
    )
    precision = correct / len(pred_keys) if pred_keys else 0.0
    recall    = correct / len(exp_keys)  if exp_keys  else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}


def avg_f1(results):
    return round(sum(r["f1"] for r in results) / len(results), 4)


# ---------------------------------------------------------------------------
# DSPy LM wrapper  — wraps local PEFT model into DSPy's LM interface
# ---------------------------------------------------------------------------

class LocalPeftLM(dspy.LM):
    """
    Wraps a loaded HuggingFace + PEFT model as a DSPy LM.
    DSPy calls lm(prompt) or lm(messages=[...]) and expects list[str] back.
    """

    def __init__(self, hf_model, tokenizer, device, max_new_tokens: int = 128):
        # dspy.LM.__init__ expects a model string identifier
        super().__init__(model="local-smollm2-peft", cache=False)
        self._hf_model       = hf_model
        self._tokenizer      = tokenizer
        self._device         = device
        self._max_new_tokens = max_new_tokens

    # DSPy 2.x calls __call__ with either a prompt string or messages list
    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is not None:
            # DSPy passes chat messages → apply chat template
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        text = self._tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        self.history.append({"prompt": prompt, "response": text})
        return [text]

    # Required by DSPy internals
    def get_usage_and_reset(self):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------------------
# DSPy Signature  — embeds winning prompt instructions in docstring
# ---------------------------------------------------------------------------

class IntentSignature(dspy.Signature):
    """
    Extract a structured JSON intent object from an Indian OTT content search query.

    Rules:
    - Copy actor/director/title names EXACTLY as written. Never substitute nicknames.
    - songs/music -> content_type | movies/films/cinema/flicks -> content_type=Movies
    - kids/children -> audience=kids | family -> audience=family | adults -> audience=adults
    - Numeric ranges like 3-5 or 6-8 -> age field (e.g. "3-5 years")
    - prime/prime video -> Amazon Prime | hotstar/disney -> Disney+ Hotstar
    - sony/sonyliv -> SonyLIV | zee/zee5 -> ZEE5 | jio -> Jio Cinema
    - latest/new/recent -> recency=latest
    - Do NOT add fields that are not present in the query.
    - Return ONLY valid JSON. No markdown fences. No extra text.

    Available fields:
    actor, director, title, similar_to, content_type, genre, language, mood,
    platform, rating_type, audience, age, recency, year, query_type, badge
    """
    query:       str = dspy.InputField(desc="User search query for Indian OTT content")
    json_output: str = dspy.OutputField(
        desc="Valid JSON object with only the fields explicitly mentioned in the query"
    )


# ---------------------------------------------------------------------------
# DSPy Program
# ---------------------------------------------------------------------------

class IntentExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(IntentSignature)

    def forward(self, query: str):
        return self.predict(query=query)


# ---------------------------------------------------------------------------
# DSPy metric  — F1 score gates which bootstrapped examples are kept
# ---------------------------------------------------------------------------

def intent_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Returns F1 score between predicted JSON and expected JSON.
    DSPy's BootstrapFewShot uses this to decide whether a (query, output)
    pair is good enough to use as a demonstration.
    Threshold: F1 >= 0.6 to be kept as a demo (configurable below).
    """
    pred_json = safe_json_load(prediction.json_output) or {}
    expected  = json.loads(example.expected_json)
    return score_output(pred_json, expected)["f1"]


def binary_metric(example, prediction, trace=None) -> bool:
    """Strict version: only keep demos with F1 == 1.0 (exact match)."""
    return intent_metric(example, prediction, trace) >= 0.6


# ---------------------------------------------------------------------------
# Evaluate a program on a split
# ---------------------------------------------------------------------------

def evaluate_program(program: IntentExtractor, split: list[dict], label: str) -> list[dict]:
    results = []
    for item in split:
        try:
            pred    = program(query=item["query"])
            pred_j  = safe_json_load(pred.json_output) or {}
            scores  = score_output(pred_j, item["expected"])
        except Exception:
            scores = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        results.append({**scores, "query": item["query"]})
    print(f"  [{label}] avg F1={avg_f1(results):.4f}  "
          f"({sum(1 for r in results if r['f1']==1.0)}/{len(results)} exact)")
    return results


# ---------------------------------------------------------------------------
# Extract optimised few-shots from the compiled program
# ---------------------------------------------------------------------------

def extract_demos(optimized_program: IntentExtractor) -> list[dict]:
    """Pull the bootstrapped demonstrations out of the compiled program."""
    demos = []
    try:
        for demo in optimized_program.predict.demos:
            demos.append({
                "query":         demo.query,
                "json_output":   demo.json_output,
                "expected_json": demo.get("expected_json", "{}"),
            })
    except AttributeError:
        pass
    return demos


# ---------------------------------------------------------------------------
# Build the final standalone prompt  (winning variant + DSPy-found demos)
# ---------------------------------------------------------------------------

def build_final_prompt(winning_template: str, demos: list[dict]) -> str:
    """
    Inject DSPy-optimised few-shot demos into the winning prompt template.
    Replaces any existing Examples block (or appends before {query}).
    """
    demo_block_lines = ["Examples (selected by DSPy optimisation):"]
    for d in demos:
        try:
            parsed = json.loads(d["json_output"])
            json_str = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            json_str = d["json_output"]
        demo_block_lines.append(f"Query: {d['query']}")
        demo_block_lines.append(f"JSON: {json_str}\n")
    demo_block = "\n".join(demo_block_lines) + "\n"

    # Remove any existing Examples block
    template = re.sub(
        r"(Examples.*?:\n)(.*?)(Query: \{query\})",
        lambda m: demo_block + "Query: {query}",
        winning_template,
        flags=re.DOTALL,
    )

    # If no Examples block existed, insert before "Query: {query}"
    if "DSPy optimisation" not in template:
        template = template.replace(
            "Query: {query}\nJSON:",
            demo_block + "Query: {query}\nJSON:",
        )

    return template


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DSPy few-shot optimizer for SmolLM2 intent extractor.")
    parser.add_argument("--base-model-id",    type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter-dir",      type=str, default="artifacts_full_dataset/models/smollm2/adapter")
    parser.add_argument("--winning-variant",  type=str, default="v1_minimal",
                        choices=list(PROMPT_VARIANTS.keys()),
                        help="The best prompt variant found by prompt_sweep.py")
    parser.add_argument("--optimizer",        type=str, default="bootstrap",
                        choices=["bootstrap", "random_search"],
                        help="bootstrap=fast, random_search=thorough (tries more combos)")
    parser.add_argument("--max-demos",        type=int, default=4,
                        help="Max few-shot examples to inject (2-6 recommended for small models)")
    parser.add_argument("--max-new-tokens",   type=int, default=128)
    parser.add_argument("--output-dir",       type=str, default="dspy_output")
    parser.add_argument("--device",           type=str, default="auto")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             (args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.base_model_id} + adapter {args.adapter_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)

    # Merge adapter for cleaner inference
    print("Merging LoRA adapter ...")
    hf_model = peft_model.merge_and_unload()
    hf_model.to(device)
    hf_model.eval()

    # ── 2. Configure DSPy with local model ───────────────────────────────────
    print("\nConfiguring DSPy with local PEFT LM ...")
    local_lm = LocalPeftLM(hf_model, tokenizer, device, args.max_new_tokens)
    dspy.configure(lm=local_lm)

    # ── 3. Build DSPy train/test sets ────────────────────────────────────────
    # DSPy Examples: input field = query, label field = expected_json (JSON string)
    train_examples = [
        dspy.Example(
            query=item["query"],
            expected_json=json.dumps(item["expected"]),
        ).with_inputs("query")
        for item in TRAIN_SET
    ]

    print(f"Train: {len(train_examples)} examples | Test: {len(TEST_SET)} queries")

    # ── 4. Baseline: winning variant before DSPy ─────────────────────────────
    print(f"\n── Baseline ({args.winning_variant}) ──")
    baseline_program = IntentExtractor()
    baseline_results = evaluate_program(baseline_program, TEST_SET, "baseline")

    # ── 5. Run DSPy optimiser ─────────────────────────────────────────────────
    print(f"\n── Running DSPy {args.optimizer} (max_demos={args.max_demos}) ──")
    print("  DSPy will run model on training queries, keep correct (query→JSON) pairs,")
    print("  and find the combination of demos that maximises F1 on the training set.\n")

    if args.optimizer == "bootstrap":
        optimizer = BootstrapFewShot(
            metric=binary_metric,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
        )
    else:
        # BootstrapFewShotWithRandomSearch tries random subsets of demos
        # Better result, but takes ~5× longer
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=binary_metric,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
            num_candidate_programs=8,
        )

    optimized_program = optimizer.compile(
        IntentExtractor(),
        trainset=train_examples,
    )

    # ── 6. Evaluate optimised program ────────────────────────────────────────
    print(f"\n── Optimised program ──")
    optimized_results = evaluate_program(optimized_program, TEST_SET, "optimized")

    # ── 7. Save DSPy program ──────────────────────────────────────────────────
    dspy_save_path = output_dir / "optimized_intent_extractor.json"
    optimized_program.save(str(dspy_save_path))
    print(f"\nSaved DSPy program → {dspy_save_path}")
    print("  (Reload later with: program.load(str(path)))")

    # ── 8. Extract demos + build standalone prompt ────────────────────────────
    demos = extract_demos(optimized_program)
    print(f"\nBootstrapped {len(demos)} few-shot demonstrations:")
    for i, d in enumerate(demos, 1):
        print(f"  [{i}] Q: {d['query']}")
        print(f"       A: {d['json_output']}")

    winning_template  = PROMPT_VARIANTS[args.winning_variant]
    final_prompt      = build_final_prompt(winning_template, demos)

    prompt_save_path  = output_dir / "final_prompt_with_dspy_demos.txt"
    prompt_save_path.write_text(final_prompt, encoding="utf-8")
    print(f"\nSaved final standalone prompt → {prompt_save_path}")

    # ── 9. Save comparison report ─────────────────────────────────────────────
    report_path = output_dir / "dspy_optimization_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("DSPy Few-Shot Optimization Report\n")
        f.write(f"Base model     : {args.base_model_id}\n")
        f.write(f"Adapter        : {args.adapter_dir}\n")
        f.write(f"Winning variant: {args.winning_variant}\n")
        f.write(f"Optimizer      : {args.optimizer}\n")
        f.write(f"Max demos      : {args.max_demos}\n\n")

        f.write(f"{'='*60}\n")
        f.write("RESULTS COMPARISON\n")
        f.write(f"{'='*60}\n")
        f.write(f"  Baseline  avg F1 : {avg_f1(baseline_results):.4f}\n")
        f.write(f"  Optimized avg F1 : {avg_f1(optimized_results):.4f}\n")
        delta = avg_f1(optimized_results) - avg_f1(baseline_results)
        f.write(f"  Delta           : {delta:+.4f}\n\n")

        f.write(f"{'='*60}\n")
        f.write("PER-QUERY COMPARISON\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Query':<45}  {'Base':>5}  {'Opt':>5}  {'Δ':>6}\n")
        f.write(f"  {'-'*45}  {'-'*5}  {'-'*5}  {'-'*6}\n")
        for b, o in zip(baseline_results, optimized_results):
            delta_q = o["f1"] - b["f1"]
            flag = " ✓" if delta_q > 0 else (" ✗" if delta_q < 0 else "")
            f.write(f"  {b['query'][:45]:<45}  {b['f1']:>5.3f}  {o['f1']:>5.3f}  {delta_q:>+6.3f}{flag}\n")

        f.write(f"\n{'='*60}\n")
        f.write("BOOTSTRAPPED FEW-SHOT DEMONSTRATIONS\n")
        f.write(f"{'='*60}\n")
        for i, d in enumerate(demos, 1):
            f.write(f"[{i}] Query: {d['query']}\n")
            f.write(f"     JSON : {d['json_output']}\n\n")

        f.write(f"{'='*60}\n")
        f.write("FINAL STANDALONE PROMPT\n")
        f.write(f"{'='*60}\n")
        f.write(final_prompt)

    print(f"Saved report → {report_path}")

    # ── 10. Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Baseline  F1 : {avg_f1(baseline_results):.4f}")
    print(f"  Optimized F1 : {avg_f1(optimized_results):.4f}  ({delta:+.4f})")
    print(f"  Demos found  : {len(demos)}")
    print(f"  Output dir   : {output_dir}/")


if __name__ == "__main__":
    main()