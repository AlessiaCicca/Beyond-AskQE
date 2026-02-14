# Summaries with DistilBART
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import os
import argparse

# Model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_distilbart(text, max_input_tokens=800, max_new_tokens=200):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize PubMed articles with DistilBART")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data_pubmed = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for ex in tqdm(data_pubmed):
            sid = ex["stringID"]

            try:
                summary = summarize_distilbart(ex["source"])
            except Exception as e:
                print("\nError on", sid, e)
                summary = "ERROR"

            record = {
                "stringID": sid,
                "summary_distilbart": summary
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
