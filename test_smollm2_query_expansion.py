import argparse
import json
from collections import Counter
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

EXPANSION_SYSTEM_PROMPT = (
    "You are a query paraphraser. Given a user query, return exactly {n} "
    "paraphrased versions of it as a JSON array of strings. "
    "Preserve the original meaning but vary wording and phrasing. "
    "Return only valid JSON, no markdown, no extra text."
)

EXPANSION_PROMPT_TEMPLATE = (
    "Paraphrase the following query {n} different ways.\n"
    "Return a JSON array of {n} strings.\n\n"
    "Query: {query}\n"
    "JSON:"
)


def safe_json_load(text: str):
    text = text.strip()
    # strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])   # drop first line
        text = text.rstrip("`\n").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_prompt(query: str) -> str:
    return PROMPT_TEMPLATE.format(query=query)


def build_expansion_prompt(query: str, n: int) -> str:
    return EXPANSION_PROMPT_TEMPLATE.format(query=query, n=n)


def generate(model, tokenizer, system: str, user: str, device: str, args) -> str:
    """Shared generation helper for both expansion and intent extraction."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.expansion_do_sample,          # sampling ON for expansions, OFF for extraction
        temperature=args.temperature if args.expansion_do_sample else 1.0,
        top_p=args.top_p if args.expansion_do_sample else 1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )


def expand_query(model, tokenizer, query: str, n: int, device: str, args) -> list[str]:
    raw = generate(
        model, tokenizer,
        system=EXPANSION_SYSTEM_PROMPT.format(n=n),
        user=build_expansion_prompt(query, n),
        device=device,
        args=args,
    )
    
    # ── DEBUG ──────────────────────────────────────────────
    print("\n[EXPANSION DEBUG]", flush=True)
    print(f"  Original query  : {query}", flush=True)
    print(f"  Raw model output: {raw!r}", flush=True)   # <-- shows EXACT characters
    print(f"  Repr hex check  : {[hex(ord(c)) for c in raw[:30]]}", flush=True)
    # ───────────────────────────────────────────────────────

    parsed = safe_json_load(raw)

    # ── DEBUG ──────────────────────────────────────────────
    print(f"  Parsed result  : {parsed}")
    if isinstance(parsed, list):
        for i, p in enumerate(parsed, 1):
            print(f"  Paraphrase {i}: {p}")
    else:
        print("  WARNING: model did not return a valid JSON list — falling back to original only")
    # ───────────────────────────────────────────────────────

    if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed) and parsed:
        expansions = parsed[:n]
    else:
        expansions = []

    return [query] + expansions


def majority_vote(candidates: list[dict | None]) -> tuple[dict | None, dict]:
    """
    Pick the most frequent valid JSON object among candidates.
    Returns (winner, vote_counts_by_canonical_string).
    """
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None, {}

    # Canonicalise each dict to a stable string for counting
    canonical = [json.dumps(v, sort_keys=True, ensure_ascii=False) for v in valid]
    counts = Counter(canonical)
    winner_str, _ = counts.most_common(1)[0]
    return json.loads(winner_str), dict(counts)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a finetuned smollm2 adapter on custom queries.")
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="artifacts_full_dataset/models/smollm2/adapter")
    parser.add_argument("--query", type=str, default=None, help="Single query to run.")
    parser.add_argument("--query-file", type=str, default=None, help="Path to newline-separated test queries.")
    parser.add_argument("--output-file", type=str, default="smollm2_test_output_query_expansion.txt", help="Path to save generated outputs.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda.")

    # ── Query expansion knobs ──────────────────────────────────────────────────
    parser.add_argument(
        "--expand", action="store_true",
        help="Enable query expansion (paraphrase + majority vote).",
    )
    parser.add_argument(
        "--num-expansions", type=int, default=3,
        help="Number of paraphrases to generate per query (default: 3).",
    )
    parser.add_argument(
        "--expansion-do-sample", action="store_true", default=True,
        help="Use sampling when generating paraphrases (recommended; default: True).",
    )
    return parser.parse_args()


def load_queries(args):
    queries = []
    if args.query is not None:
        queries.append(args.query.strip())
    if args.query_file is not None:
        query_path = Path(args.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        queries.extend(
            [line.strip() for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    if not queries:
        raise ValueError("Please provide --query or --query-file with at least one query.")
    return queries

def load_base_model(args, device):
    """Load the base model WITHOUT the LoRA adapter, used only for expansion."""
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer

    
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

            # ── Step 1: build candidate query list ────────────────────────────
            if args.expand:
                candidate_queries = expand_query(
                    model, tokenizer, query,
                    n=args.num_expansions,
                    device=device,
                    args=args,
                )
            else:
                candidate_queries = [query]

            # ── Step 2: run intent extraction on every candidate ───────────────
            per_candidate_results = []
            for cq in candidate_queries:
                raw = generate(
                    model, tokenizer,
                    system=SYSTEM_PROMPT,
                    user=build_prompt(cq),
                    device=device,
                    args=args,
                )
                parsed = safe_json_load(raw)
                per_candidate_results.append(
                    {"query": cq, "raw": raw, "parsed": parsed}
                )

            # ── Step 3: majority vote across all candidates ────────────────────
            all_parsed = [r["parsed"] for r in per_candidate_results]
            final_parsed, vote_counts = majority_vote(all_parsed)

            # ── Step 4: write results ──────────────────────────────────────────
            lines = [
                "---",
                f"Query {idx}: {query}",
            ]

            if args.expand:
                lines.append(f"Expansions ({len(candidate_queries) - 1} paraphrases + original):")
                for i, r in enumerate(per_candidate_results):
                    tag = "ORIGINAL" if i == 0 else f"PARAPHRASE {i}"
                    lines += [
                        f"  [{tag}] {r['query']}",
                        f"    Raw  : {r['raw'].strip()}",
                        f"    Valid: {json.dumps(r['parsed'], ensure_ascii=False) if r['parsed'] else '<invalid>'}",
                    ]
                lines.append(f"Vote counts: {vote_counts}")
            else:
                lines += [
                    "Generated output:",
                    per_candidate_results[0]["raw"],
                ]

            lines += [
                "Final JSON (majority vote):" if args.expand else "Parsed JSON:",
                json.dumps(final_parsed, indent=2, ensure_ascii=False)
                if final_parsed is not None
                else "<invalid JSON>",
                "",
            ]

            output_file.write("\n".join(lines) + "\n")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()