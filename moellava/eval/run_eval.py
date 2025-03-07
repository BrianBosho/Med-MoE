import argparse
import json
import collections
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to ground truth file')
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file')
    parser.add_argument('--output', type=str, default="wrong_answers.json", help='path to output file for wrong answers')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred):    
    scores = collections.defaultdict(list)
    closed_scores = collections.defaultdict(list)
    closed_questions_count=0
    closed_questions_correct=0
    wrong_answers = []

    # Print sample data to understand the format
    print("GT sample item keys:", list(gt[0].keys()) if gt else "Empty GT")
    print("Pred sample item keys:", list(pred[0].keys()) if pred else "Empty Pred")

    for gt_item, pred_item in zip(gt, pred):
        # Handle different GT formats
        if 'conversations' in gt_item:
            gt_results = gt_item['conversations']
            gt_value = gt_results[1]['value'].lower()
        elif 'conversatons' in gt_item:
            gt_results = gt_item['conversatons']
            gt_value = gt_results[1]['value'].lower()
        elif 'answer' in gt_item:
            # Direct answer field
            gt_value = gt_item['answer'].lower()
        elif 'gt_answers' in gt_item:
            # Some datasets use gt_answers field
            gt_value = gt_item['gt_answers'].lower() if isinstance(gt_item['gt_answers'], str) else gt_item['gt_answers'][0].lower()
        else:
            # Try to find any field that might contain the answer
            possible_answer_fields = ['ans', 'gt', 'ground_truth', 'label']
            for field in possible_answer_fields:
                if field in gt_item:
                    gt_value = gt_item[field].lower()
                    break
            else:
                print(f"Warning: Could not find answer field in GT item: {gt_item.keys()}")
                continue  # Skip this item if we can't find an answer
        
        # Get prediction
        pred_value = pred_item['text'].lower()
        
        # Get answer type, default to 'open' if not specified
        answer_type = gt_item.get('answer_type', 'open')
        
        # Continue with existing evaluation logic
        if answer_type == 'open' or answer_type == 'OPEN':
            scores['exact_match'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            scores['f1'].append(f1_score)
            scores['precision'].append(precision)
            scores['recall'].append(recall)

            weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]  
            bleu_scores = []
            for w in weights:
                bleu_score = sentence_bleu([gt_value.split()], pred_value.split(), weights=w, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)
            scores['bleu_scores'].append(bleu_scores)
            
        elif answer_type == 'close' or answer_type == 'CLOSED' :
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            closed_scores['f1'].append(f1_score)
            closed_scores['precision'].append(precision)
            closed_scores['recall'].append(recall)
            closed_questions_count += 1
            if gt_value not in pred_value:  # Check if gt_value is not contained within pred_value
                wrong_answers.append({'prompt': pred_item['prompt'], 'correct_answer': gt_value, 'predicted_answer': pred_value})
            else:
                closed_questions_correct += 1  # Update the count of correct answers

    # Safe average calculation with empty list handling
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0

    exact_match_avg = safe_avg(scores['exact_match'])
    f1_score_avg = safe_avg(scores['f1'])
    precision_avg = safe_avg(scores['precision']) 
    recall_avg = safe_avg(scores['recall'])
    
    # Handle empty bleu scores list
    if scores['bleu_scores']:
        bleu_scores_avg = [sum(score_list) / len(score_list) for score_list in zip(*scores['bleu_scores'])]
    else:
        bleu_scores_avg = [0, 0, 0]  # Default to zeros
    
    closed_score = (closed_questions_correct / closed_questions_count * 100) if closed_questions_count else 0
    closed_f1_score_avg = safe_avg(closed_scores['f1'])
    closed_precision_avg = safe_avg(closed_scores['precision'])
    closed_recall_avg = safe_avg(closed_scores['recall'])

    results_table = tabulate(
        [
            ['Exact Match Score', exact_match_avg*100],
            ['F1 Score', f1_score_avg*100],
            ['Precision', precision_avg*100],
            ['Recall', recall_avg*100],
            ['BLEU Score (Weight 1)', bleu_scores_avg[0]*100],
            ['BLEU Score (Weight 2)', bleu_scores_avg[1]*100],
            ['BLEU Score (Weight 3)', bleu_scores_avg[2]*100],
            ['yes/no accuracy', closed_score], 
            ['Closed F1 Score', closed_f1_score_avg*100],
            ['Closed Precision', closed_precision_avg*100],
            ['Closed Recall', closed_recall_avg*100],
        ],
        headers=['Metric', 'Performance (%)']
    )
    
    with open(args.output, 'w') as f:
        json.dump(wrong_answers, f, indent=4)

    return results_table

if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    results = evaluate(gt, pred)
    print(results)
