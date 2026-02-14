# ASKQE-NLI: NLI-Based Factual Consistency Metric for AskQE

### Description
This repository introduces an extension to the AskQE framework to evaluate translations using **Natural Language Inference** (NLI). 
Inspired by prior work, specifically the paper [here]([https://aclanthology.org/2021.emnlp-main.619.pdf](https://arxiv.org/abs/2104.08202)), we implement an NLI pipeline to assess the quality of translation outputs by 
comparing the answers from the source (SRC) and backtranslated text (BT).

## Core Idea
The evaluation metric employs a two-step process using NLI using `RoBERTa-large-MNLI` model to compare question-answer pairs from the source and backtranslation:

- **Premise**: The concatenation of question and answer pair from the source text (SRC).

- **Hypothesis**: The concatenation of question and answer pair from the backtranslation (BT).

The NLI model checks for entailment, contradiction, or neutrality between the premise and the hypothesis:

- *Entailment*: The hypothesis logically follows from the premise, meaning the translation preserves the meaning of the source answer.

- *Contradiction*: The hypothesis contradicts the premise, indicating a translation error.

- *Neutral*: The hypothesis is neither entailed nor contradictory to the premise, meaning the translation may be ambiguous or unclear.

This helps evaluate whether the translation retains the semantic meaning of the source answer or if there are discrepancies in meaning.

Here is an example:

<p align="center">
  <img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/5e1f9cf7-7ce6-4491-96e3-ffab5b322a8f" />
</p>

---

## Project Structure

```
ASKQE-BASELINE/
├── QA/                            #Scripts for Question Answering
├── QG/                            #Scripts for Question Answering  
├── evaluation/                    # Scripts for evaluation assessment (including NLI pipeline)
├── backtranslation.py             # Backtranslation with Google Translate
├── dev_with_backtranslation.py    # BioMQM dataset used in the original paper
├── requirements.txt               # Requirements
└── README.md                      # This file
```

## Usage


### Question Generation
```python
!python QG/code/qwen-3b.py --output_path <output_file> --prompt <prompt_type>
```
Arguments for the QG code are as follows:
- --output_path: save path of output file (after question generation).
- --prompt: QG variant (whether vanilla / semantic / atomic). All our experiments were performed with the vanilla prompt.

### Back-translation
Since the BioMQM dataset already provides MT outputs with error annotations, we only have to run backtranslation on those.
```python
python backtranslation.py --input_path <input_file> --output_path <output_file> 
```

### Question Answering

```python
python QA/code/qwen-3b-noanswer.py --input_path <input_file> --output_path <output_file> --sentence_type <sentence_type>
```
Where `sentence_type` can be src / bt_tgt. This script must be ran with both configurations to obtain answers both on source text and backtranslated text.

### Evaluation

#### Basic evaluation: sbert.py and string-comparison.py

These scripts compute the similarity scores for each pair of `answer_src` and `answer_bt` related to one question, along with the average similarity among all answers for a sentence.
While `sbert.py` focuses on calculating cosine similarity, `stringcomparison.py` extends this by calculating the four usual metrics for string comparison: F1, BLEU, ChrF, and EM. 


```python
python evaluation/sbert/sbert-noanswer.py
```

```python
python evaluation/string-comparison/string-comparison-noanswer.py
```

#### NLI evaluation: nli_metric.py 
This is the core script of the extension. For each sentence, it runs an **NLI pipeline** as described above and then computes a custom metric based on the label assigned 
among 'contradiction', 'neutral', and 'entailment'.

```python
python evaluation/nli/nli_metric.py \
  --input_f1 <input_path_string_comparison> \
  --input_sbert <input_path_sbert> \
  --output <output_path>
```
#### Comparison evaluation: assign_severity.py and compare_results.py
The script `assign_severity.py` assigns each sentence to a severity label among "Critical", "Major", "Minor", "Neutral", "No error" based on the highest severity of the errors annotated in the sentence.

```python
python evaluation/assign_severity.py \
  --input <input_path_nli> \
  --output <output_path>
```

The script `compare_results.py` computes two main comparisons among results and outputs two csv files, in paricular:
- a disaggregated comparison of average SBERT, F1, BLEU, ChrF, and EM scores based on severity and language-pairs;
- a comparison of global average SBERT, F1, BLEU, ChrF, and EM scores.

```python
python evaluation/compare_results.py \
  --input <input_path_nli_severity> \
  --output_grouped <ouput_path_grouped_results> \
  --output_global <ouput_path_global_results>
```

#### Human Simulation: annotation_rule.py, gmm.py and compare.py
All these 3 scripts can be run for **different metrics**: SBERT, F1, BLEU, ChrF, EM, NLI.

The script `annotation_rule.py` similar to the original annotation rule file, but it has been modified to accommodate nested lists, allowing for multiple severity levels. If errors are marked with a severity of "Critical" or "Major," the translation is rejected; otherwise, it is accepted.

```python
python evaluation/annotation_rule.py \
  --input <input_path_nli_severity> \
  --output <output_path_annotated>
```

The script `gmm.py` first calculates the average value of the chosen metric (SBERT, F1, BLEU, ChrF, EM or NLI) across all 5–6 generated responses. The Gaussian Mixture Model (GMM) then attempts to fit two distributions: one with a lower mean (representing low quality) and one with a higher mean (representing high quality). The average of the two centroids is taken as the threshold to decide whether the translation should be accepted or rejected, based on the average metric value.

```python
python evaluation/gmm.py \
  --input <input_path_annotated> \
  --output <output_path_gmm> \
  --metric <metric>
```

The script `compare.py` is similar to the original comparison file. It compares the files generated by `annotation_rule.py` and `gmm.py`, checking how often the same translations are accepted or rejected. The results are evaluated using precision, recall, and a confusion matrix.

```python
!python evaluation/compare.py \
  --gold <input_path_annotated> \
  --pred <input_path_gmm>
```


