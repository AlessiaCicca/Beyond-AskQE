import json
import os
import argparse
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Question Answering over source and summary texts using Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--pubmed", type=str, required=True,
                        help="PubMed JSON file with original sources")
    parser.add_argument("--summaries", type=str, required=True,
                        help="JSON file with generated summaries")
    parser.add_argument("--questions", type=str, required=True,
                        help="JSONL file with generated questions")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with QA results")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()

QA_PROMPT = """You are answering factual questions using ONLY the provided text.

Rules:
- Answers must be short spans copied verbatim from the text.
- Do NOT repeat the question.
- Do NOT repeat the article.
- If the answer is not explicitly stated, output exactly: No_Answer
- Output ONLY a Python list of answers.

Text:
{text}

Questions:
{questions}

Answers:
"""

def run_llm(prompt, tokenizer, model, max_new_tokens):
    messages = [
        {"role": "system", "content": "You are a helpful QA assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return response

def parse_answers(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [a.strip() for a in parsed]
    except Exception:
        pass

    return [
        line.strip("-• ").strip()
        for line in raw.split("\n")
        if line.strip()
    ]

def main():
    args = parse_args()
    model_id = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load PubMed sources
    with open(args.pubmed, "r", encoding="utf-8") as f:
        pubmed = {ex["stringID"]: ex["source"] for ex in json.load(f)}

    # Load summaries
    summaries = {}

    with open(args.summaries, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            summaries[ex["stringID"]] = ex["summary_distilbart"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    processed = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(json.loads(line)["stringID"])

    with open(args.questions, "r", encoding="utf-8") as fin, \
         open(args.output, "a", encoding="utf-8") as fout:

        for line in tqdm(fin):
            obj = json.loads(line)
            sid = obj["stringID"]

            if sid in processed:
                continue

            questions = obj.get("questions", [])
            if not questions:
                continue

            q_str = json.dumps(questions, ensure_ascii=False)

            # SOURCE QA
            prompt_src = QA_PROMPT.format(text=pubmed.get(sid, ""),questions=q_str)
            raw_src = run_llm(prompt_src,tokenizer,model,args.max_new_tokens)
            answers_src = parse_answers(raw_src)

            # SUMMARY QA
            prompt_sum = QA_PROMPT.format(text=summaries.get(sid, ""),questions=q_str)
            raw_sum = run_llm(prompt_sum,tokenizer,model,args.max_new_tokens)
            answers_sum = parse_answers(raw_sum)

            fout.write(json.dumps({
                "stringID": sid,
                "questions": questions,
                "answers_source": answers_src,
                "answers_summary": answers_sum
            }, ensure_ascii=False) + "\n")

            fout.flush()
            print(sid, ":", len(questions), "questions")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
