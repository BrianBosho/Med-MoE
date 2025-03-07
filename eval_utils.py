import argparse
import json
import collections
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from moellava.eval.eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization

import warnings
warnings.simplefilter('ignore')

def parse_option():
    """Parse command-line arguments for standalone script usage."""
    parser = argparse.ArgumentParser('Evaluation for MedVQA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to ground truth file')
    parser.add_argument('--pred', type=str, default="answer-file.jsonl", help='path to prediction file')
    parser.add_argument('--output', type=str, default="evaluation_results.csv", help='path to output CSV file for results')
    parser.add_argument('--wrong_answers', type=str, default="wrong_answers.json", help='path to output file for wrong answers')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    """Load data from a JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

def evaluate_with_metrics(gt_data, pred_data, output_wrong_answers=None):
    """
    Evaluate predictions against ground truth and add evaluation metrics as new columns.
    
    Args:
        gt_data: List of ground truth data items
        pred_data: List of prediction data items
        output_wrong_answers: Optional path to save wrong answers
        
    Returns:
        pandas.DataFrame: DataFrame containing the original data with added metric columns
    """
    results = []
    wrong_answers = []
    
    for gt_item, pred_item in zip(gt_data, pred_data):
        gt_results = gt_item.get('conversations', gt_item.get('conversatons'))
        gt_value = gt_results[1]['value'].lower()
        pred_value = pred_item['text'].lower()
        answer_type = gt_item['answer_type']
        
        # Create a result item with original data
        result_item = {
            'id': gt_item.get('id', ''),
            'question': pred_item.get('prompt', ''),
            'ground_truth': gt_value,
            'prediction': pred_value,
            'answer_type': answer_type
        }
        
        # Calculate metrics based on answer type
        if answer_type == 'open' or answer_type == 'OPEN':
            # Exact match
            exact_match = calculate_exactmatch(pred_value, gt_value)
            result_item['exact_match'] = exact_match
            
            # F1 score
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            result_item['f1_score'] = f1_score
            result_item['precision'] = precision
            result_item['recall'] = recall
            
            # BLEU scores
            weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
            bleu_scores = []
            for i, w in enumerate(weights):
                bleu_score = sentence_bleu([gt_value.split()], pred_value.split(), 
                                          weights=w, 
                                          smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)
                result_item[f'bleu_{i+1}'] = bleu_score
                
        elif answer_type == 'close' or answer_type == 'CLOSED':
            # For closed questions
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            result_item['f1_score'] = f1_score
            result_item['precision'] = precision
            result_item['recall'] = recall
            
            # Binary accuracy for closed questions
            result_item['is_correct'] = 1 if gt_value in pred_value else 0
            
            # Add to wrong answers if incorrect
            if gt_value not in pred_value:
                wrong_answers.append({
                    'prompt': pred_item.get('prompt', ''),
                    'correct_answer': gt_value,
                    'predicted_answer': pred_value
                })
        
        results.append(result_item)
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    
    # Save wrong answers if path provided
    if output_wrong_answers:
        with open(output_wrong_answers, 'w') as f:
            json.dump(wrong_answers, f, indent=4)
    
    return df_results

def summarize_metrics(df):
    """
    Generate a summary of evaluation metrics from the results DataFrame.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        dict: Dictionary with average metrics
    """
    summary = {}
    
    # Open-ended questions
    open_df = df[df['answer_type'].str.lower() == 'open']
    if len(open_df) > 0:
        summary['open_questions_count'] = len(open_df)
        summary['exact_match_avg'] = open_df['exact_match'].mean() * 100
        summary['f1_score_avg'] = open_df['f1_score'].mean() * 100
        summary['precision_avg'] = open_df['precision'].mean() * 100
        summary['recall_avg'] = open_df['recall'].mean() * 100
        
        # BLEU scores
        for i in range(1, 4):
            if f'bleu_{i}' in open_df.columns:
                summary[f'bleu_{i}_avg'] = open_df[f'bleu_{i}'].mean() * 100
    
    # Closed questions
    closed_df = df[df['answer_type'].str.lower() == 'closed']
    if len(closed_df) > 0:
        summary['closed_questions_count'] = len(closed_df)
        summary['closed_accuracy'] = closed_df['is_correct'].mean() * 100
        summary['closed_f1_score_avg'] = closed_df['f1_score'].mean() * 100
        summary['closed_precision_avg'] = closed_df['precision'].mean() * 100
        summary['closed_recall_avg'] = closed_df['recall'].mean() * 100
    
    return summary

def main():
    """Main function for standalone script execution."""
    args = parse_option()
    
    # Load data
    gt_data = json.load(open(args.gt, 'r'))
    pred_data = load_jsonl(args.pred)
    
    # Evaluate
    df_results = evaluate_with_metrics(gt_data, pred_data, args.wrong_answers)
    
    # Save results
    df_results.to_csv(args.output, index=False)
    
    # Print summary
    summary = summarize_metrics(df_results)
    print("\nEvaluation Summary:")
    for metric, value in summary.items():
        print(f"{metric}: {value:.2f}" if isinstance(value, float) else f"{metric}: {value}")
    
    return df_results, summary

if __name__ == '__main__':
    main() 