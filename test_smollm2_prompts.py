import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Evaluation Query Set  (30 queries — unseen names, synonyms, edge cases)
# ---------------------------------------------------------------------------
# Each entry: query the model sees, expected JSON it should produce.
# Covers: unseen actor names, synonym keywords, multi-field combos,
#         age/audience, platform aliases, mood, edge cases.

EVAL_QUERIES = [
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


# ---------------------------------------------------------------------------
# Prompt Variants  (5 templates, progressively more detailed)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a structured intent extractor for an Indian OTT content platform. Return only valid JSON."

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
# Scoring
# ---------------------------------------------------------------------------

def score_output(predicted: dict, expected: dict) -> dict:
    """Field-level precision, recall, F1 against expected output."""
    if not isinstance(predicted, dict):
        predicted = {}

    pred_keys = set(predicted.keys())
    exp_keys  = set(expected.keys())

    correct_values = sum(
        1 for k in (pred_keys & exp_keys)
        if str(predicted[k]).strip().lower() == str(expected[k]).strip().lower()
    )

    precision = correct_values / len(pred_keys) if pred_keys else 0.0
    recall    = correct_values / len(exp_keys)  if exp_keys  else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "f1":           round(f1, 4),
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "correct":      correct_values,
        "missing_keys": sorted(exp_keys - pred_keys),
        "extra_keys":   sorted(pred_keys - exp_keys),
        "wrong_values": {
            k: {"predicted": predicted[k], "expected": expected[k]}
            for k in (pred_keys & exp_keys)
            if str(predicted[k]).strip().lower() != str(expected[k]).strip().lower()
        },
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    return {
        "f1":        round(sum(r["f1"]        for r in results) / n, 4),
        "precision": round(sum(r["precision"] for r in results) / n, 4),
        "recall":    round(sum(r["recall"]    for r in results) / n, 4),
        "exact_match_pct": round(
            100 * sum(1 for r in results if r["f1"] == 1.0) / n, 1
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_json_load(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rstrip("`\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def generate_one(model, tokenizer, system_prompt, user_prompt, device, args) -> str:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, variant_name, prompt_template, device, args, out_f):
    out_f.write(f"\n{'='*70}\n")
    out_f.write(f"VARIANT: {variant_name}\n")
    out_f.write(f"{'='*70}\n\n")

    per_query_results = []

    for idx, item in enumerate(EVAL_QUERIES, 1):
        query    = item["query"]
        expected = item["expected"]

        user_prompt = prompt_template.format(query=query)
        generated   = generate_one(model, tokenizer, SYSTEM_PROMPT, user_prompt, device, args)
        parsed      = safe_json_load(generated) or {}
        scores      = score_output(parsed, expected)

        per_query_results.append(scores)

        out_f.write(f"  [{idx:02d}] Query   : {query}\n")
        out_f.write(f"       Expected: {json.dumps(expected, ensure_ascii=False)}\n")
        out_f.write(f"       Got     : {json.dumps(parsed,   ensure_ascii=False)}\n")
        out_f.write(f"       F1={scores['f1']:.3f}  P={scores['precision']:.3f}  R={scores['recall']:.3f}")
        if scores["missing_keys"]:
            out_f.write(f"  missing={scores['missing_keys']}")
        if scores["extra_keys"]:
            out_f.write(f"  extra={scores['extra_keys']}")
        if scores["wrong_values"]:
            out_f.write(f"  wrong={scores['wrong_values']}")
        out_f.write("\n\n")

    summary = summarize(per_query_results)
    out_f.write(f"  SUMMARY — F1={summary['f1']:.4f}  "
                f"P={summary['precision']:.4f}  "
                f"R={summary['recall']:.4f}  "
                f"ExactMatch={summary['exact_match_pct']}%\n")

    return summary, per_query_results


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def print_leaderboard(leaderboard: dict, out_f):
    out_f.write(f"\n\n{'='*70}\n")
    out_f.write("LEADERBOARD (sorted by F1)\n")
    out_f.write(f"{'='*70}\n")
    out_f.write(f"  {'Variant':<26}  {'F1':>6}  {'Precision':>9}  {'Recall':>6}  {'Exact%':>7}\n")
    out_f.write(f"  {'-'*26}  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*7}\n")
    for name, s in sorted(leaderboard.items(), key=lambda x: -x[1]["f1"]):
        out_f.write(
            f"  {name:<26}  {s['f1']:>6.4f}  {s['precision']:>9.4f}"
            f"  {s['recall']:>6.4f}  {s['exact_match_pct']:>6.1f}%\n"
        )
    winner = max(leaderboard, key=lambda x: leaderboard[x]["f1"])
    out_f.write(f"\n  ✓ Best prompt: {winner}  (F1={leaderboard[winner]['f1']:.4f})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt variant sweep for SmolLM2 intent extractor.")
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter-dir",   type=str, default="artifacts_full_dataset/models/smollm2/adapter")
    parser.add_argument("--output-file",   type=str, default="prompt_sweep_results.txt")
    parser.add_argument("--max-new-tokens",type=int, default=128)
    parser.add_argument("--variants",      type=str, default="all",
                        help="Comma-separated variant names to run, or 'all'.")
    parser.add_argument("--device",        type=str, default="auto")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
              args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    variants_to_run = (
        PROMPT_VARIANTS if args.variants == "all"
        else {k: PROMPT_VARIANTS[k] for k in args.variants.split(",") if k in PROMPT_VARIANTS}
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard      = {}
    per_query_scores = {}

    with output_path.open("w", encoding="utf-8") as out_f:
        out_f.write("PROMPT VARIANT SWEEP — SmolLM2 Intent Extractor\n")
        out_f.write(f"Base model : {args.base_model_id}\n")
        out_f.write(f"Adapter    : {args.adapter_dir}\n")
        out_f.write(f"Eval size  : {len(EVAL_QUERIES)} queries\n")

        for name, template in variants_to_run.items():
            print(f"\nRunning variant: {name} ...")
            summary, per_q = run_eval(model, tokenizer, name, template, device, args, out_f)
            leaderboard[name]      = summary
            per_query_scores[name] = per_q
            print(f"  F1={summary['f1']:.4f}  P={summary['precision']:.4f}"
                  f"  R={summary['recall']:.4f}  Exact={summary['exact_match_pct']}%")

        print_leaderboard(leaderboard, out_f)
        # Per-query F1 comparison table
        names = list(per_query_scores.keys())
        out_f.write(f"\n\n{'='*70}\n")
        out_f.write("PER-QUERY F1 COMPARISON\n")
        out_f.write(f"{'='*70}\n")
        out_f.write(f"  {'Query':<45} " + "  ".join(f"{n[:10]:>10}" for n in names) + "\n")
        for idx, item in enumerate(EVAL_QUERIES):
            row = f"  {item['query'][:45]:<45} "
            row += "  ".join(f"{per_query_scores[n][idx]['f1']:>10.3f}" for n in names)
            out_f.write(row + "\n")

    print(f"\nResults saved to: {output_path}")
    winner = max(leaderboard, key=lambda x: leaderboard[x]["f1"])
    print(f"Best prompt: {winner}  (F1={leaderboard[winner]['f1']:.4f})")


if __name__ == "__main__":
    main()