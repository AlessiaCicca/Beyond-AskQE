# Beyond AskQE: Expanding to Summarization and Enhancing MT Quality Estimation via NLI, Multi-Turn Questioning, and Hallucination Detection


## Overview
This research extends the AskQE framework for evaluating machine translation (MT) quality through question generation and answering. The original framework compares answers from source sentences and their backtranslated outputs to enable monolingual quality assessment.

## Problem Statement
Current MT quality estimation methods produce either difficult-to-interpret scalar scores or target-language annotations inaccessible to monolingual users. This gap is critical in high-stakes domains like healthcare, where source-language speakers need actionable feedback without understanding the target language.

## Key Limitations Addressed
1. **Semantic nuance handling**: Treating semantically similar but factually distinct answers as equivalent
2. **Shallow evaluation**: Missing errors requiring deeper probing
3. **Hallucination detection**: Failing to identify when translations introduce incorrect information
4. **Limited scope**: Restricted to MT evaluation only

## Four Main Extensions

### A. NLI-Based Answer Comparison
- Uses Natural Language Inference (RoBERTa-large-MNLI) to assess factual consistency
- Assigns binary entailment scores (1 for entailment, 0 for contradiction)
- Retains F1 scores for neutral cases
- **Result**: Improved decision accuracy from 47.34% to 56.99%

### B. Multi-Turn Questioning
- Iterative follow-up questions across multiple rounds (Nturn iterations)
- Each round builds on previous responses for deeper semantic analysis
- Uses weighted averaging: 30% for question similarity, 70% for answer similarity
- **Result**: Improved precision and recall, particularly with ChrF and SBERT metrics

### C. Backtranslation-Based QA for Hallucination Detection
Inverts the AskQE paradigm by generating questions from backtranslations:
- **Unanswerable Content Rate (UCR)**: Detects facts not grounded in source (32% sentence-level detection)
- **BERTScore filtering**: Identifies semantic mismatches
- **Yes/No verification**: Binary validation of claims (56.25% sentence-level detection)

### D. Summarization Evaluation
Adapts framework to measure information preservation in biomedical summaries:
- Generates questions from source documents (not summaries)
- Evaluates: "If the document states X, is X preserved in the summary?"
- Provides graded measurement of content coverage


More detailed information about each extension, including implementation details, experimental setups, and additional results, can be found in the respective project folders and in the complete research paper.

---


## Baseline: Original AskQE Framework

The baseline AskQE implementation can be found in the folder of the same name in this repository.

This work builds upon the AskQE framework introduced by Ki et al. (2025), which pioneered the use of question answering for automatic machine translation evaluation. The baseline repository provides implementations and evaluation scripts for applying the **AskQE framework** (Ki, Duh, and Carpuat) to perform Machine Translation (MT) quality estimation on the BioMQM dataset, utilizing a **small LLM** for Question Answering (QA) and Question Generation (QG).

Unlike the large LLMs used in the original paper, this repository leverages a smaller LLM, `Qwen2.5-3B-Instruct`, to demonstrate the framework's efficiency in quality estimation tasks while maintaining computational feasibility.


## Core Idea
Monolingual source speakers cannot effectively evaluate machine translation (MT) quality in languages they don’t understand, and existing quality estimation (QE) methods fail to address this by offering hard-to-interpret scores 
or target-language annotations. This is especially problematic in high-stakes contexts like healthcare, where accurate, source-language feedback is crucial. 
The AskQE framework addresses this by generating questions from the source text and comparing answers from both the source and the back-translated MT output, highlighting potential translation errors.

---

## Project Structure

```
ASKQE-BASELINE/
├── QA/                            #Scripts for Question Answering
├── QG/                            #Scripts for Question Answering  
├── evaluation/                    # Scripts for evaluation assessment
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
python QA/code/qwen-3b.py --input_path <input_file> --output_path <output_file> --sentence_type <sentence_type>
```
Where `sentence_type` can be src / bt_tgt. This script must be ran with both configurations to obtain answers both on source text and backtranslated text.

### Evaluation

The evaluation scripts are designed to process and assess the quality of translations through multiple steps and metrics. Below is a description of the key scripts and their functionality:

### evaluation/sbert/sbert.py and evaluation/string-comparison/string-comparison.py
These scripts compute the similarity scores for each pair of `answer_src` and `answer_bt` related to one question, along with the average similarity among all answers for a sentence.
While `sbert.py` focuses on calculating cosine similarity, `stringcomparison.py` extends this by calculating the four usual metrics for string comparison: F1, BLEU, ChrF, and EM. 

### evaluation/assign_severity.py
This scripts assigns each sentence to a severity label among "Critical", "Major", "Minor", "Neutral", "No error" based on the highest severity of the errors annotated in the sentence.

### evaluation/compare_results.py
This script computes two main comparisons among results and outputs two csv files, in paricular:
- a disaggregated comparison of average SBERT, F1, BLEU, ChrF, and EM scores based on severity and language-pairs;
- a comparison of global average SBERT, F1, BLEU, ChrF, and EM scores.

### evaluation/annotation_rule.py
This script is similar to the original annotation rule file, but it has been modified to accommodate nested lists, allowing for multiple severity levels. If errors are marked with a severity of "Critical" or "Major," the translation is rejected; otherwise, it is accepted.

### evaluation/gmm.py
This script first calculates the average value of the chosen metric across all 5–6 generated responses. The Gaussian Mixture Model (GMM) then attempts to fit two distributions: one with a lower mean (representing low quality) and one with a higher mean (representing high quality). The average of the two centroids is taken as the threshold to decide whether the translation should be accepted or rejected, based on the average metric value.

### evaluation/compare.py
This script is similar to the original comparison file. It compares the files generated by `annotation_rule.py` and `gmm.py`, checking how often the same translations are accepted or rejected. The results are evaluated using precision, recall, and a confusion matrix.




