import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "Convert the user query into a valid JSON intent object.\n"
    "Return only valid JSON, with no markdown fences and no extra text.\n\n"
    "Query: {query}\n"
    "JSON:"
)

SYSTEM_PROMPT = "You are a structured intent extractor. Return only valid JSON."


# ---------------------------------------------------------------------------
# Rule-Based Value Mapper
# ---------------------------------------------------------------------------
# Closed-vocabulary fields: deterministic keyword -> canonical value maps.
# These directly fix "unseen synonym -> wrong/missing value" errors
# (e.g. "songs"/"music" -> content_type, "kids"/"children" -> audience).
# Open-vocabulary fields (actor, director, title, similar_to) are left to the
# model's output, with a simple sanity check against the original query.

CONTENT_TYPE_MAP = {
    "web series": "Web Series",
    "songs": "Songs", "song": "Songs", "music": "Music",
    "movies": "Movies", "movie": "Movies", "film": "Movies", "films": "Movies",
    "shows": "Shows", "show": "Shows", "series": "Series",
    "trailers": "Trailers", "trailer": "Trailers",
    "episodes": "Episodes", "shorts": "Shorts",
}

AUDIENCE_MAP = {
    "kids": "kids", "children": "kids", "child": "kids","kid'": "kid",
    "adults": "adults", "adult": "adults",
    "family": "family",
}

GENRE_MAP = {
    "action": "Action", "comedy": "Comedy", "drama": "Drama",
    "horror": "Horror", "romance": "Romance", "thriller": "Thriller",
    "crime": "Crime", "biopic": "Biopic", "sports": "Sports",
    "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "mystery": "Mystery",
    "historical": "Historical", "supernatural": "Supernatural",
    "dark comedy": "Dark comedy", "psychological thriller": "Psychological thriller",
}

MOOD_MAP = {
    "action packed": "action packed", "dark": "dark", "emotional": "emotional",
    "feel good": "feel good", "funny": "funny", "happy": "happy",
    "inspirational": "inspirational", "intense": "intense",
    "light hearted": "light hearted", "motivational": "motivational",
    "romantic": "romantic", "sad": "sad", "scary": "scary",
    "suspenseful": "suspenseful", "tearjerker": "tearjerker",
    "thriller wala": "thriller wala",
}

PLATFORM_MAP = {
    "amazon prime": "Amazon Prime", "prime video": "Amazon Prime", "prime": "Amazon Prime",
    "disney+ hotstar": "Disney+ Hotstar", "disney hotstar": "Disney+ Hotstar",
    "hotstar": "Disney+ Hotstar", "disney": "Disney+ Hotstar",
    "jio cinema": "Jio Cinema", "jio": "Jio Cinema",
    "mx player": "MX Player", "mx": "MX Player",
    "netflix": "Netflix",
    "sonyliv": "SonyLIV", "sony liv": "SonyLIV", "sony": "SonyLIV",
    "youtube": "YouTube",
    "zee5": "ZEE5", "zee": "ZEE5",
}

RATING_TYPE_MAP = {
    "award winning": "award winning", "blockbuster": "blockbuster",
    "classic": "classic", "cult": "cult", "flop": "flop",
    "hit": "hit", "must watch": "must watch", "overrated": "overrated",
    "top rated": "top rated", "underrated": "underrated",
}

RECENCY_MAP = {
    "latest": "latest", "newest": "latest", "new": "latest", "recent": "latest",
}

LANGUAGE_MAP = {
    "bengali": "Bengali", "bhojpuri": "Bhojpuri", "gujarati": "Gujarati",
    "hindi": "Hindi", "kannada": "Kannada", "malayalam": "Malayalam",
    "marathi": "Marathi", "punjabi": "Punjabi", "tamil": "Tamil", "telugu": "Telugu",
}

QUERY_TYPE_MAP = {
    "available on": "availability", "availability": "availability",
    "where to watch": "availability", "streaming on": "availability",
    "rating": "rating/review", "review": "rating/review",
    "reviews": "rating/review", "ratings": "rating/review",
}

# Fields whose VALUES come purely from the maps above (closed vocabulary)
CLOSED_VOCAB_FIELD_MAPS = {
    "content_type": CONTENT_TYPE_MAP,
    "audience": AUDIENCE_MAP,
    "genre": GENRE_MAP,
    "mood": MOOD_MAP,
    "platform": PLATFORM_MAP,
    "rating_type": RATING_TYPE_MAP,
    "recency": RECENCY_MAP,
    "language": LANGUAGE_MAP,
    "query_type": QUERY_TYPE_MAP,
}

# Open-vocabulary fields: values are copied from the query, not from a fixed list
OPEN_VOCAB_FIELDS = {"actor", "director", "title", "similar_to"}


def match_from_map(query_lower: str, value_map: dict):
    """Return the canonical value for the longest matching trigger phrase, or None."""
    for trigger in sorted(value_map.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(trigger)}\b", query_lower):
            return value_map[trigger]
    return None


def rule_based_extract(query: str) -> dict:
    """
    Deterministically extract all closed-vocabulary fields directly from
    the query text. Covers content_type, audience, genre, mood, platform,
    rating_type, recency, language, query_type, age, year.
    """
    q = query.lower()
    result = {}

    for field, value_map in CLOSED_VOCAB_FIELD_MAPS.items():
        value = match_from_map(q, value_map)
        if value is not None:
            result[field] = value

    # Age range: "3-5", "6 to 8", "3–5"
    age_match = re.search(r"\b(\d{1,2})\s*[-–to]+\s*(\d{1,2})\b", query, re.IGNORECASE)
    if age_match:
        result["age"] = f"{age_match.group(1)}-{age_match.group(2)} years"

    # Year: 2018-2025 (from training data range)
    year_match = re.search(r"\b(201[8-9]|202[0-5])\b", query)
    if year_match:
        result["year"] = year_match.group(1)

    return result


def post_process(query: str, model_output: dict) -> dict:
    """
    Hybrid merge:
      - Closed-vocab fields: rule-based extraction OVERRIDES model output
        (rules are deterministic and handle unseen synonyms correctly).
      - Open-vocab fields (actor/director/title/similar_to): keep model
        output ONLY if it passes a basic sanity check against the query,
        i.e. the value (or part of it) actually appears in the query text.
        This catches hallucinations like "Sallu" for "Saurav Chakraborty".
    """
    if not isinstance(model_output, dict):
        model_output = {}

    q_lower = query.lower()
    result = {}

    # 1. Rule-based closed-vocab fields take priority
    rule_result = rule_based_extract(query)
    result.update(rule_result)

    # 2. Open-vocab fields from model output, with sanity check
    for field in OPEN_VOCAB_FIELDS:
        value = model_output.get(field)
        if not value or not isinstance(value, str):
            continue
        parts = [p for p in value.lower().split() if len(p) > 2]
        if parts and any(p in q_lower for p in parts):
            result[field] = value
        # else: hallucinated value not present in query -> dropped

    # 3. Any remaining model-output fields not already handled by rules
    #    or open-vocab logic above (e.g. badge) — keep only if value text
    #    appears verbatim in the query, to avoid hallucinations.
    HANDLED_FIELDS = set(CLOSED_VOCAB_FIELD_MAPS.keys()) | OPEN_VOCAB_FIELDS | {"age", "year"}
    for field, value in model_output.items():
        if field in HANDLED_FIELDS or field in result:
            continue
        if isinstance(value, str) and value.lower() in q_lower:
            result[field] = value

    return result


def safe_json_load(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rstrip("`\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_prompt(query: str) -> str:
    return PROMPT_TEMPLATE.format(query=query)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a finetuned smollm2 adapter on custom queries.")
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="artifacts_full_dataset/models/smollm2/adapter")
    parser.add_argument("--query", type=str, default=None, help="Single query to run.")
    parser.add_argument("--query-file", type=str, default=None, help="Path to newline-separated test queries.")
    parser.add_argument("--output-file", type=str, default="smollm2_test_output.txt", help="Path to save generated outputs.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda.")
    return parser.parse_args()


def load_queries(args):
    queries = []
    if args.query is not None:
        queries.append(args.query.strip())
    if args.query_file is not None:
        query_path = Path(args.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        queries.extend([line.strip() for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if not queries:
        raise ValueError("Please provide --query or --query-file with at least one query.")
    return queries


def main():
    args = parse_args()
    queries = load_queries(args)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device)
    model.eval()

    output_path = Path(args.output_file)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for idx, query in enumerate(queries, start=1):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(query)},
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
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            parsed = safe_json_load(generated)

            # Apply rule-based value mapper on top of the raw model output
            final_result = post_process(query, parsed if parsed is not None else {})

            result_text = [
                "---",
                f"Query {idx}: {query}",
                "Generated output (raw model):",
                generated,
                "Parsed JSON (raw model):",
                json.dumps(parsed, indent=2, ensure_ascii=False) if parsed is not None else "<invalid JSON>",
                "Final output (after rule-based value mapping):",
                json.dumps(final_result, indent=2, ensure_ascii=False),
                "",
            ]
            output_file.write("\n".join(result_text) + "\n")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()