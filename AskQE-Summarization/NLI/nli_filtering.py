import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLI-based filtering of atomic facts using sliding window entailment"
    )
    parser.add_argument("--pubmed", type=str, required=True,
                        help="Original PubMed JSON file")
    parser.add_argument("--atomic_facts", type=str, required=True,
                        help="JSONL file with extracted atomic facts")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with entailed facts only")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment probability threshold")
    parser.add_argument("--window_tokens", type=int, default=350,
                        help="Sliding window size in tokens")
    parser.add_argument("--stride", type=int, default=200,
                        help="Stride size in tokens")
    return parser.parse_args()


def is_entailed_sliding(source,fact,tokenizer,model,device,threshold,window_tokens,stride):
    """
    Returns True if fact is entailed by ANY chunk of source.
    """
    source_tokens = tokenizer(
        source,
        truncation=False,
        return_tensors=None
    )["input_ids"]

    max_start = max(len(source_tokens) - window_tokens + 1, 1)

    for start in range(0, max_start, stride):
        chunk_tokens = source_tokens[start:start + window_tokens]

        if len(chunk_tokens) < 50:
            continue

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        inputs = tokenizer(
            chunk_text,
            fact,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1)
        entailment_score = probs[0][2].item()  # index 2 = entailment

        if entailment_score >= threshold:
            return True

    return False


def main():
    args = parse_args()

    nli_model_name = "roberta-large-mnli"

    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("NLI model loaded on", device)

    # Load original PubMed data
    with open(args.pubmed, "r", encoding="utf-8") as f:
        data_pubmed = json.load(f)

    id_to_source = {
        ex["stringID"]: ex["source"]
        for ex in data_pubmed
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.atomic_facts, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(tqdm(fin), start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON error on line {line_num}: {e}")
                continue

            sid = obj.get("stringID")
            facts = obj.get("atomic_facts", [])
            source = id_to_source.get(sid, "")

            if not source:
                continue

            # Ensure facts is a list
            if isinstance(facts, str):
                try:
                    facts = json.loads(facts)
                except Exception:
                    facts = []

            if not isinstance(facts, list):
                continue

            entailed = []

            for fact in facts:
                try:
                    if is_entailed_sliding(source,fact,tokenizer,model,device,args.threshold,args.window_tokens,args.stride):
                        entailed.append(fact)
                except Exception as e:
                    print("NLI error on", sid, fact[:50], e)

            out_record = {
                "stringID": sid,
                "source": source,
                "facts_entailed": entailed
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()

            print(sid, "→", len(entailed), "/", len(facts), "entailed")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
