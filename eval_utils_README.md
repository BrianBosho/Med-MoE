# Medical VQA Evaluation Utilities

This package provides tools for evaluating medical visual question answering (VQA) model outputs against ground truth answers. It supports both open-ended and closed (yes/no) type questions and calculates various evaluation metrics.

## Features

- Evaluation of both open-ended and closed-type medical questions
- Multiple evaluation metrics:
  - Exact match
  - F1 score, precision, and recall
  - BLEU scores (1-gram, 2-gram, 3-gram)
  - Binary accuracy for closed questions
- Export of detailed results with per-question metrics
- Summary statistics for model performance
- Support for wrong answer analysis
- Jupyter notebook example for interactive analysis

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas nltk
python -m nltk.downloader punkt
```

The code also depends on the custom evaluation metrics in the `moellava` package.

## Usage

### As a Command-Line Tool

```bash
python eval_utils.py --gt test.json --pred answer-file.jsonl --output results.csv --wrong_answers wrong.json
```

Arguments:
- `--gt`: Path to ground truth file (JSON format)
- `--pred`: Path to predictions file (JSONL format)
- `--output`: Path to save detailed results (CSV format)
- `--wrong_answers`: Path to save wrong answers (JSON format)

### As an Imported Module

```python
import json
from eval_utils import evaluate_with_metrics, summarize_metrics, load_jsonl

# Load data
gt_data = json.load(open("test.json", 'r'))
pred_data = load_jsonl("answer-file.jsonl")

# Evaluate predictions
results_df = evaluate_with_metrics(gt_data, pred_data, "wrong_answers.json")

# Get summary statistics
summary = summarize_metrics(results_df)

# Print summary
for metric, value in summary.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value}")

# Save results
results_df.to_csv("evaluation_results.csv", index=False)
```

## Jupyter Notebook Example

See `evaluation_example.ipynb` for a complete example of how to use these utilities in a notebook, including visualization of results and comparison between different models.

## Expected Data Format

### Ground Truth Data (JSON)

```json
[
  {
    "id": "question_1",
    "answer_type": "open",
    "conversations": [
      {"from": "human", "value": "What abnormality is seen in this chest X-ray?"},
      {"from": "gpt", "value": "The chest X-ray shows a right upper lobe consolidation."}
    ]
  },
  {
    "id": "question_2",
    "answer_type": "closed",
    "conversations": [
      {"from": "human", "value": "Is there a pneumothorax visible in this image?"},
      {"from": "gpt", "value": "yes"}
    ]
  }
]
```

### Prediction Data (JSONL)

```jsonl
{"prompt": "What abnormality is seen in this chest X-ray?", "text": "There is a consolidation in the right upper lobe."}
{"prompt": "Is there a pneumothorax visible in this image?", "text": "No, I don't see evidence of pneumothorax."}
```

## Output Format

The evaluation generates a detailed CSV file with per-question results and a summary of overall performance metrics. 