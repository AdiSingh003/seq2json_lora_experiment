import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import dspy


# ---------------------------------------------------------------------------
# DSPy: detailed signature describing the full intent schema and mapping rules
# ---------------------------------------------------------------------------
# This docstring IS the prompt — DSPy turns it into the model instructions.
# It encodes the field list, value-mapping rules, and anti-hallucination
# rules discussed earlier (exact-name copying, content_type vs genre vs
# audience disambiguation, closed-vocab lists, age/year formatting, etc.)

class QueryToIntentJSON(dspy.Signature):
    """You are a structured intent extractor for an Indian OTT content
    discovery platform. Convert a free-form user search query into a
    single valid JSON object describing the user's intent.

    Allowed keys (use ONLY these, and ONLY when clearly supported by the
    query — do not invent or guess fields that aren't present):

    - actor: a person's name appearing as a performer in the query.
      Copy the name EXACTLY as written in the query, character for
      character. Never substitute a nickname, abbreviation, or a
      different (more familiar) name for the one actually written.
    - director: a person's name appearing as a director in the query.
      Same exact-copy rule as actor.
    - title: the name of a movie/show/song explicitly mentioned as the
      subject of the query (not as a comparison target).
    - similar_to: a movie/show title the user wants recommendations
      similar to (e.g. "shows like X", "something like X").
    - content_type: the FORMAT of content requested. Map query words to
      these canonical values:
        "songs"/"music"/"tracks"      -> "Songs"
        "movies"/"films"/"flicks"     -> "Movies"
        "shows"/"series"              -> "Shows"
        "web series"                  -> "Web Series"
        "trailers"/"trailer"          -> "Trailers"
        "episodes"                    -> "Episodes"
      Only set this field if one of these format words (or a clear
      synonym) appears in the query.
    - genre: one of Action, Biopic, Comedy, Crime, "Dark comedy", Drama,
      Family, Historical, Horror, Mystery, "Psychological thriller",
      Romance, "Sci-Fi", Sports, Supernatural, Thriller — ONLY if the
      query explicitly names or clearly implies one of these genres.
    - language: one of Bengali, Bhojpuri, Gujarati, Hindi, Kannada,
      Malayalam, Marathi, Punjabi, Tamil, Telugu — ONLY if that language
      is named in the query.
    - mood: one of "action packed", dark, emotional, "feel good", funny,
      happy, inspirational, intense, "light hearted", motivational,
      romantic, sad, scary, suspenseful, tearjerker, "thriller wala" —
      ONLY if the query expresses that mood/vibe.
    - platform: one of "Amazon Prime", "Disney+ Hotstar", "Jio Cinema",
      "MX Player", Netflix, SonyLIV, YouTube, ZEE5 — ONLY if a streaming
      platform is named (including common short forms like "Prime",
      "Hotstar", "Disney").
    - rating_type: one of "award winning", blockbuster, classic, cult,
      flop, hit, "must watch", overrated, "top rated", underrated — ONLY
      if such a quality/rating descriptor appears in the query.
    - query_type: "availability" if the user is asking WHERE/whether
      content can be watched (e.g. "is X available on Y", "where can I
      watch"); "rating/review" if the user is asking about ratings or
      reviews. Omit otherwise.
    - audience: "kids" (also for "children"/"child"), "adults"
      (also for "adult"), or "family" — ONLY if an audience/age-group
      word appears in the query. Do NOT confuse this with genre: a
      query about "kids" content sets audience="kids", NOT
      genre="Family".
    - age: a numeric age range exactly as implied by the query, written
      as "<low>-<high> years" (e.g. "3-5 years" for "3-5" or "3 to 5").
      ONLY set this if a numeric range is present in the query.
    - badge: "Original" — ONLY if the query explicitly mentions an
      "Original" / platform-original label.
    - recency: "latest" — ONLY if the query uses words like "latest",
      "new", "newest", or "recent".
    - year: a 4-digit year — ONLY if that exact
      year appears in the query.

    General rules:
    1. Output ONLY a single JSON object — no markdown fences, no
       explanations, no extra text before or after.
    2. Include a key ONLY if the query gives clear evidence for it.
       Never include a key with a guessed or default value.
    3. Never invent person names, titles, or values that do not appear
       in (or are not directly implied by) the query text.
    4. If the query mentions multiple concepts (e.g. an actor AND a
       content type AND an audience), include all of the corresponding
       keys in the same JSON object.

    Examples:
    Query: Salman Khan ki songs
    JSON: {"actor": "Salman Khan", "content_type": "Songs"}

    Query: Saurav Chakraborty kids 3-5
    JSON: {"actor": "Saurav Chakraborty", "audience": "kids", "age": "3-5 years"}

    Query: Aamir Khan Hindi action movies on Netflix
    JSON: {"actor": "Aamir Khan", "language": "Hindi", "genre": "Action", "content_type": "Movies", "platform": "Netflix"}

    Query: latest Telugu movies on Amazon Prime
    JSON: {"recency": "latest", "language": "Telugu", "content_type": "Movies", "platform": "Amazon Prime"}

    Query: Rajinikanth top rated films
    JSON: {"actor": "Rajinikanth", "rating_type": "top rated", "content_type": "Movies"}
    """

    query: str = dspy.InputField(
        desc="A free-form natural language search query typed by a user "
             "on an Indian OTT content discovery platform."
    )
    intent_json: str = dspy.OutputField(
        desc="A single valid JSON object (as a string) representing the "
             "extracted intent, following the allowed keys and mapping "
             "rules described in the instructions. No markdown fences."
    )


# ---------------------------------------------------------------------------
# DSPy LM wrapper around the local fine-tuned HF model (no retraining)
# ---------------------------------------------------------------------------

class HFLocalLM(dspy.LM):
    """Wraps an already fine-tuned local HF causal model for use as a DSPy LM."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 128, **kwargs):
        super().__init__(model="local-hf-causal", **kwargs)
        self.hf_model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def _generate(self, text_prompt: str) -> str:
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    def forward(self, prompt=None, messages=None, **kwargs):
        if messages is not None:
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                text_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        else:
            text_prompt = prompt or ""
        return self._generate(text_prompt)

    def __call__(self, prompt=None, messages=None, **kwargs):
        completion = self.forward(prompt=prompt, messages=messages, **kwargs)
        return [completion]


def safe_json_load(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rstrip("`\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Test a finetuned smollm2 adapter on custom queries using DSPy.")
    parser.add_argument("--base-model-id", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="artifacts_full_dataset/models/smollm2/adapter")
    parser.add_argument("--query", type=str, default=None, help="Single query to run.")
    parser.add_argument("--query-file", type=str, default=None, help="Path to newline-separated test queries.")
    parser.add_argument("--output-file", type=str, default="smollm2_test_output.txt", help="Path to save generated outputs.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda.")
    parser.add_argument("--dspy-program", type=str, default="artifacts_full_dataset/models/smollm2/dspy/optimized_program.json",
                        help="Optional path to a saved optimized DSPy program "
                             "(e.g. artifacts_custom/models/<model>/dspy/optimized_program.json). "
                             "If not provided, the detailed signature is used directly with no few-shot demos.")
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

    # ------------------------------------------------------------------
    # Wire up DSPy: wrap the fine-tuned model as a DSPy LM and build the
    # extraction program from the detailed QueryToIntentJSON signature.
    # ------------------------------------------------------------------
    lm = HFLocalLM(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
    dspy.settings.configure(lm=lm)

    program = dspy.Predict(QueryToIntentJSON)
    # print("Initialized DSPy program with detailed signature.\n Program: ", program)

    if args.dspy_program:
        program_path = Path(args.dspy_program)
        if program_path.exists():
            program.load(str(program_path))
            print(f"Loaded optimized DSPy program from: {program_path}")
        else:
            print(f"Warning: --dspy-program path not found ({program_path}); "
                  f"using the detailed signature with no few-shot demos.")

    output_path = Path(args.output_file)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for idx, query in enumerate(queries, start=1):
            prediction = program(query=query)
            generated = prediction.intent_json
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