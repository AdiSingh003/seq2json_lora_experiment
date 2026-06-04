# Seq2JSON LoRA experiment

This folder contains a complete pipeline to:
1. download the query-intent spreadsheet as CSV,
2. split into 90/10 train/validation,
3. fine-tune SLMs with LoRA,
4. evaluate them on validation data,
5. compare base model outputs vs fine-tuned outputs on a probe set.
