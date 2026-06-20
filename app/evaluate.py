"""
Script to evaluate PJM live decoding accuracy using Precision, Recall, and F1-Score.

This tool compares a ground truth text file with the application's prediction log
and calculates standard classification metrics to assess the system's performance.
"""

import os
from collections import Counter


def evaluate_batch(ground_truth_file: str, prediction_file: str) -> None:
    """
    Reads ground truth and prediction files, extracts words, and calculates 
    Precision, Recall, and F1-Score based on word occurrences.

    Args:
        ground_truth_file (str): Path to the text file containing the actual sentences.
        prediction_file (str): Path to the text file containing the model's predictions.

    Returns:
        None
    """
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            target_text = " ".join(line.strip() for line in f.readlines())
    except FileNotFoundError:
        print(f"Error: Could not find {ground_truth_file}")
        return

    try:
        with open(prediction_file, "r", encoding="utf-8") as f:
            predicted_text = " ".join(line.strip() for line in f.readlines())
    except FileNotFoundError:
        print(f"Error: Could not find {prediction_file}")
        return

    target_words = target_text.strip().upper().split()
    predicted_words = predicted_text.strip().upper().split()

    if not target_words:
        print("Error: Ground truth file is empty.")
        return

    target_counts = Counter(target_words)
    predicted_counts = Counter(predicted_words)

    true_positives = 0
    for word, count in predicted_counts.items():
        if word in target_counts:
            true_positives += min(count, target_counts[word])

    total_predicted = len(predicted_words)
    total_target = len(target_words)

    precision = (true_positives / total_predicted) * 100 if total_predicted > 0 else 0.0
    recall = (true_positives / total_target) * 100

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print("========================================")
    print("BATCH EVALUATION RESULTS")
    print("========================================")
    print(f"Ground Truth File : {ground_truth_file}")
    print(f"Prediction File   : {prediction_file}")
    print("----------------------------------------")
    print(f"Total Target Words    : {total_target}")
    print(f"Total Predicted Words : {total_predicted}")
    print(f"True Positives        : {true_positives} (Correctly identified)")
    print("----------------------------------------")
    print(f"Recall (Sensitivity) : {recall:.1f}%")
    print(f"Precision (Accuracy) : {precision:.1f}%")
    print(f"F1-Score             : {f1_score:.1f}%")
    print("========================================")


if __name__ == "__main__":
    truth_file = "avatar_testing_sentences.txt"
    pred_file = "prediction_log.txt"

    if os.path.exists(truth_file) and os.path.exists(pred_file):
        evaluate_batch(truth_file, pred_file)
    else:
        print(f"Please ensure both '{truth_file}' and '{pred_file}' exist in this folder.")