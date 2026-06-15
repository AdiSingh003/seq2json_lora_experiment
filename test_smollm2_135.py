import argparse
import json
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

SYSTEM_PROMPT = "You are a structured intent extractor. Return only valid JSON.Check if all required fields exist.Fix if needed"


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
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="artifacts_full_dataset/models/smollm2_135m/adapter")
    parser.add_argument("--query", type=str, default=None, help="Single query to run.")
    parser.add_argument("--query-file", type=str, default=None, help="Path to newline-separated test queries.")
    parser.add_argument("--output-file", type=str, default="smollm2_135m_test_output.txt", help="Path to save generated outputs.")
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

            result_text = [
                "---",
                f"Query {idx}: {query}",
                "Generated output:",
                generated,
                "Parsed JSON:",
                json.dumps(parsed, indent=2, ensure_ascii=False) if parsed is not None else "<invalid JSON>",
                "",
            ]
            output_file.write("\n".join(result_text) + "\n")

            
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
