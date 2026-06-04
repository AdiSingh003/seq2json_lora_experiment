import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class ModelSpec:
    name: str
    model_id: str
    kind: str
    task_type: str


MODEL_SPECS = [
    ModelSpec(name="smollm2", model_id="HuggingFaceTB/SmolLM2-360M-Instruct", kind="causal", task_type="CAUSAL_LM"),
    ModelSpec(name="indicbart", model_id="ai4bharat/IndicBART", kind="seq2seq", task_type="SEQ_2_SEQ_LM"),
]


PROMPT_TEMPLATE = (
    "Convert the user query into a valid JSON intent object.\n"
    "Return only valid JSON, with no markdown fences and no extra text.\n\n"
    "Query: {query}\n"
    "JSON:"
)

PROBE_QUERIES = [
    "recent panka",
    "find the cast of Scam 1992",
    "what new Tamil releases are available on Netflix",
    "play the latest songs from the movie soundtrack",
    "who acted in the movie related to the film scam",
]


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning and evaluation for intent JSON generation.")
    parser.add_argument("--data-url", type=str, default="https://docs.google.com/spreadsheets/d/1y54Zzxgrs3EGMPpqXBvfGacDyn1MqNEt/export?format=csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model-epochs", type=str, default='{"smollm2":5,"indicbart":10}', help="JSON mapping of model name to epoch count")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_directories(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "raw").mkdir(parents=True, exist_ok=True)
    (base_dir / "models").mkdir(parents=True, exist_ok=True)
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    (base_dir / "probes").mkdir(parents=True, exist_ok=True)


def download_data(csv_url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(csv_url, timeout=120)
    r.raise_for_status()
    output_path.write_text(r.text)
    return output_path


def safe_json_load(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_json(text: str):
    loaded = safe_json_load(text)
    if loaded is None:
        return None
    return json.dumps(loaded, ensure_ascii=False, sort_keys=True)


def build_prompt(query: str) -> str:
    return PROMPT_TEMPLATE.format(query=query)


def parse_model_epochs(raw: str) -> Dict[str, int]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--model-epochs must be a JSON object mapping model names to epoch counts")
    normalized = {}
    for model_name, epochs in parsed.items():
        normalized[str(model_name)] = int(epochs)
    return normalized


def tokenize_seq2seq_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples["input_text"],
            text_target=examples["target_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = [
            [token if token != tokenizer.pad_token_id else -100 for token in label_ids]
            for label_ids in tokenized["labels"]
        ]
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["input_text", "target_text"],
        desc="Tokenizing seq2seq dataset",
    )


def generate_probe_outputs(model, tokenizer, spec, queries, max_new_tokens: int = 256):
    rows = []
    with torch.no_grad():
        for query in queries:
            if spec.kind == "causal":
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a structured intent extractor. Return only valid JSON,no markdown, no extra text."},
                        {"role": "user", "content": build_prompt(query)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                pred = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            else:
                inputs = tokenizer(build_prompt(query), return_tensors="pt", truncation=True, max_length=512).to(model.device)
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            rows.append({
                "query": query,
                "generated": pred,
                "parsed_json": safe_json_load(pred),
            })
    return pd.DataFrame(rows)


def make_causal_dataset(df: pd.DataFrame) -> Dataset:
    records = []
    for _, row in df.iterrows():
        query = str(row["query"])
        intent_json = str(row["intent_json"])
        system_prompt = "You are a structured intent extractor. Return only valid JSON."
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(query)},
            {"role": "assistant", "content": intent_json},
        ]
        records.append({"text": conversation})
    return Dataset.from_list(records)


def main():
    args = parse_args()
    base_dir = Path(args.output_dir)
    ensure_directories(base_dir)

    raw_path = base_dir / "raw" / "query_intent_pairs.csv"
    download_data(args.data_url, raw_path)

    df = pd.read_csv(raw_path)
    df = df[["id", "query", "intent_json"]].copy()
    df["intent_json"] = df["intent_json"].apply(normalize_json)
    df = df.dropna(subset=["intent_json"])
    df["query"] = df["query"].astype(str).str.strip()
    df["input_prompt"] = df["query"].apply(build_prompt)
    df["target_json"] = df["intent_json"]
    if args.max_train_samples is not None:
        df = df.head(args.max_train_samples).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=args.seed,
        shuffle=True,
    )

    train_df.to_csv(base_dir / "reports" / "train_split.csv", index=False)
    val_df.to_csv(base_dir / "reports" / "val_split.csv", index=False)

    # create metrics summary file
    summary_rows = []

    model_epochs = parse_model_epochs(args.model_epochs)

    for spec in MODEL_SPECS:
        model_name = spec.name
        epochs = model_epochs.get(model_name, args.epochs)
        model_dir = base_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if spec.kind == "causal":
            tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            base_model.eval()
            baseline_probe = generate_probe_outputs(base_model, tokenizer, spec, PROBE_QUERIES)
            baseline_probe.to_csv(base_dir / "probes" / f"{model_name}_probe_before.csv", index=False)
            del base_model
            torch.cuda.empty_cache()

            tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

            train_dataset = Dataset.from_list([
                {
                    "text": tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "You are a structured intent extractor. Return only valid JSON."},
                            {"role": "user", "content": row["input_prompt"]},
                            {"role": "assistant", "content": row["target_json"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                }
                for _, row in train_df.iterrows()
            ])
            eval_dataset = Dataset.from_list([
                {
                    "text": tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "You are a structured intent extractor. Return only valid JSON."},
                            {"role": "user", "content": row["input_prompt"]},
                            {"role": "assistant", "content": row["target_json"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                }
                for _, row in val_df.iterrows()
            ])

            training_args = SFTConfig(
                output_dir=str(model_dir / "checkpoints"),
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                num_train_epochs=epochs,
                logging_steps=100,
                save_strategy="epoch",
                eval_strategy="epoch",
                report_to="none",
                bf16=torch.cuda.is_available(),
                fp16=False,
                dataloader_num_workers=2,
                remove_unused_columns=False,
                warmup_ratio=0.03,
                weight_decay=0.01,
                seed=args.seed,
                save_total_limit=2,
                max_length=args.max_length,
                dataset_text_field="text",
                packing=False,
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
            trainer.train()
            trainer.save_model(str(model_dir / "adapter"))
            tokenizer.save_pretrained(str(model_dir / "adapter"))

            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, str(model_dir / "adapter"))
            model.eval()

        else:
            tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                spec.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            base_model.eval()
            baseline_probe = generate_probe_outputs(base_model, tokenizer, spec, PROBE_QUERIES)
            baseline_probe.to_csv(base_dir / "probes" / f"{model_name}_probe_before.csv", index=False)
            del base_model
            torch.cuda.empty_cache()

            model = AutoModelForSeq2SeqLM.from_pretrained(
                spec.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, peft_config)

            train_dataset = tokenize_seq2seq_dataset(
                Dataset.from_dict({
                    "input_text": train_df["input_prompt"].tolist(),
                    "target_text": train_df["target_json"].tolist(),
                }),
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
            eval_dataset = tokenize_seq2seq_dataset(
                Dataset.from_dict({
                    "input_text": val_df["input_prompt"].tolist(),
                    "target_text": val_df["target_json"].tolist(),
                }),
                tokenizer=tokenizer,
                max_length=args.max_length,
            )

            training_args = Seq2SeqTrainingArguments(
                output_dir=str(model_dir / "checkpoints"),
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                num_train_epochs=epochs,
                logging_steps=100,
                save_strategy="epoch",
                eval_strategy="epoch",
                report_to="none",
                bf16=torch.cuda.is_available(),
                fp16=False,
                dataloader_num_workers=2,
                remove_unused_columns=False,
                warmup_ratio=0.03,
                weight_decay=0.01,
                seed=args.seed,
                save_total_limit=2,
                predict_with_generate=True,
                generation_max_length=256,
            )

            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
            )
            trainer.train()
            trainer.save_model(str(model_dir / "adapter"))
            tokenizer.save_pretrained(str(model_dir / "adapter"))

            model = AutoModelForSeq2SeqLM.from_pretrained(
                spec.model_id,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, str(model_dir / "adapter"))
            model.eval()

        # evaluate on validation set
        model_outputs = []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Evaluating {model_name}"):
            target = row["target_json"]
            if spec.kind == "causal":
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a structured intent extractor. Return only valid JSON."},
                        {"role": "user", "content": row["input_prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                pred = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            else:
                inputs = tokenizer(row["input_prompt"], return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
                output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            parsed = safe_json_load(pred)
            pred_json = json.dumps(parsed, ensure_ascii=False, sort_keys=True) if parsed is not None else None
            target_json = json.loads(target)
            if pred_json is not None:
                pred_obj = json.loads(pred_json)
                pred_keys = set(pred_obj.keys())
                target_keys = set(target_json.keys())
                key_jaccard = len(pred_keys & target_keys) / max(len(pred_keys | target_keys), 1)
            else:
                pred_obj = None
                key_jaccard = 0.0

            precision, recall, f1 = compute_slot_metrics(parsed, target_json)

            model_outputs.append({
                "id": int(row["id"]),
                "query": row["query"],
                "target_json": target,
                "pred_json": pred_json,
                "valid_json": pred_json is not None,
                "exact_match": pred_json == target,
                "key_jaccard": key_jaccard,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

        val_metrics_df = pd.DataFrame(model_outputs)
        metrics = {
            "model": model_name,
            "valid_json_rate": float(val_metrics_df["valid_json"].mean()),
            "exact_match_rate": float(val_metrics_df["exact_match"].mean()),
            "avg_key_jaccard": float(val_metrics_df["key_jaccard"].mean()),
            "avg_precision": float(mean_or_zero(val_metrics_df["precision"])),
            "avg_recall": float(mean_or_zero(val_metrics_df["recall"])),
            "avg_f1": float(mean_or_zero(val_metrics_df["f1"])),
        }
        summary_rows.append(metrics)
        val_metrics_df.to_csv(base_dir / "reports" / f"{model_name}_val_metrics.csv", index=False)

        # fixed probe to compare before and after
        probe_df = generate_probe_outputs(model, tokenizer, spec, PROBE_QUERIES)
        probe_df.to_csv(base_dir / "probes" / f"{model_name}_probe_after.csv", index=False)

        baseline_df = pd.read_csv(base_dir / "probes" / f"{model_name}_probe_before.csv")
        compare_df = baseline_df.merge(probe_df, on="query", suffixes=("_before", "_after"))
        compare_df.to_csv(base_dir / "probes" / f"{model_name}_probe_comparison.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "reports" / "model_summary.csv", index=False)


if __name__ == "__main__":
    main()
