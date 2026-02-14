import json
import argparse
from tqdm import tqdm
import sacrebleu
from utils import normalize_answer, f1_score, exact_match

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute lexical string-based metrics (F1, EM, chrF, BLEU)"
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
        help="Output JSONL file with string comparison metrics"
    )
    return parser.parse_args()

def chrf_score(pred, ref, normalize=True):
    if normalize:
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
    return sacrebleu.sentence_chrf(pred, [ref]).score

def bleu_score(pred, ref, normalize=True):
    if normalize:
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
    return sacrebleu.sentence_bleu(pred, [ref]).score

def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            data = json.loads(line)

            preds = data.get("answers_summary", [])
            refs = data.get("answers_source", [])

            if not preds or not refs or len(preds) != len(refs):
                continue

            row_scores = []
            f1s, ems, chrfs, bleus = [], [], [], []

            for pred, ref in zip(preds, refs):

                if pred == "No_Answer" and ref == "No_Answer":
                    continue

                if pred == "No_Answer" or ref == "No_Answer":
                    row_scores.append({
                        "f1": 0.0,
                        "em": False,
                        "chrf": 0.0,
                        "bleu": 0.0
                    })
                    f1s.append(0.0)
                    ems.append(0)
                    chrfs.append(0.0)
                    bleus.append(0.0)
                    continue

                f1 = f1_score(pred, ref, normalize=True)
                em = exact_match(pred, ref, normalize=True)
                chrf = chrf_score(pred, ref, normalize=True)
                bleu = bleu_score(pred, ref, normalize=True)

                row_scores.append({
                    "f1": f1,
                    "em": em,
                    "chrf": chrf,
                    "bleu": bleu
                })

                f1s.append(f1)
                ems.append(1 if em else 0)
                chrfs.append(chrf)
                bleus.append(bleu)

            data["string_scores"] = row_scores
            data["avg_f1"] = sum(f1s) / len(f1s) if f1s else None
            data["avg_em"] = sum(ems) / len(ems) if ems else None
            data["avg_chrf"] = sum(chrfs) / len(chrfs) if chrfs else None
            data["avg_bleu"] = sum(bleus) / len(bleus) if bleus else None

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            
if __name__ == "__main__":
    main()
