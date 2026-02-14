import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute semantic similarity using SBERT (all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with QA results"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file with SBERT similarity scores"
    )
    return parser.parse_args()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            data = json.loads(line)

            preds = data.get("answers_summary", [])
            refs = data.get("answers_source", [])

            if not preds or not refs or len(preds) != len(refs):
                continue

            cos_list = []
            scores = []

            for pred, ref in zip(preds, refs):

                if pred == "No_Answer" and ref == "No_Answer":
                    continue

                if pred == "No_Answer" or ref == "No_Answer":
                    scores.append({"cos_sim": 0.0})
                    cos_list.append(0.0)
                    continue

                enc_pred = tokenizer(
                    pred,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                enc_ref = tokenizer(
                    ref,
                    return_tensors="pt",
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    out_pred = model(**enc_pred)
                    out_ref = model(**enc_ref)

                emb_pred = F.normalize(
                    mean_pooling(out_pred, enc_pred["attention_mask"]),
                    p=2,
                    dim=1
                )

                emb_ref = F.normalize(
                    mean_pooling(out_ref, enc_ref["attention_mask"]),
                    p=2,
                    dim=1
                )

                cos = F.cosine_similarity(emb_pred, emb_ref).item()

                scores.append({"cos_sim": cos})
                cos_list.append(cos)

            data["sbert_scores"] = scores
            data["avg_sbert"] = (
                sum(cos_list) / len(cos_list)
                if cos_list else None
            )

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
