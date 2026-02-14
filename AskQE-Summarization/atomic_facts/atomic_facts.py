import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract atomic facts from PubMed articles using Qwen2.5-3B-Instruct")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file (PubMed)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    return parser.parse_args()

def main():
    args = parse_args()
    model_id = "Qwen/Qwen2.5-3B-Instruct"

    atomic_fact_prompt = """Task: You will be given an English biomedical text. Your goal is to identify a list of atomic facts from the sentence. Atomic fact is a short sentence conveying one piece of information. Output the list of atomic facts in Python list format without giving any additional explanation.

*** Example Starts ***

Sentence: In a cohort of 312 patients, 45% showed a significant reduction in systolic blood pressure after 12 weeks of treatment, while adverse events were reported in 8% of cases.

Atomic facts: ["The study included a cohort of 312 patients.","45% of patients showed a significant reduction in systolic blood pressure after 12 weeks of treatment.","Adverse events were reported in 8% of cases."]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.

Atomic facts: ["The number of accessory proteins is unique depending on the specific coronavirus.","The function of accessory proteins is unique depending on the specific coronavirus."]

*** Example Ends ***

Sentence: {{sentence}}
Atomic facts:
"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    with open(args.input, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_file:
        for data in tqdm(data_list):
            if "source" not in data:
                continue

            sentence = data["source"]
            prompt = atomic_fact_prompt.replace("{{sentence}}", sentence)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            try:
                atomic_facts = json.loads(response.replace("'", '"'))
            except Exception:
                atomic_facts = response

            data["atomic_facts"] = atomic_facts
            out_file.write(json.dumps(data, ensure_ascii=False) + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
