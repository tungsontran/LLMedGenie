import transformers
import os
import requests
import json
import argparse

import numpy as np
import pandas as pd

from LLMedGenie.project_path import DATASET_DIR, OUTPUT_DIR, ROOT_DIR
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from typing import List, Dict, Any


def parse_arguments():
    parser = argparse.ArgumentParser(description='LLM Backend Service')
    parser.add_argument('--dataset', type=str,
                        help='Classify documents from the dataset')

    return parser.parse_args()


def encode_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_encoded = []
    for sample in data:
        sample_encoded = {}
        sample_encoded['Label'] = sample['medical_specialty']
        sample = json.dumps(sample)
        sample_encoded['Document'] = model.encode(sample).tolist()
        data_encoded.append(sample_encoded)
    return data_encoded


def knn_classification_test(data, k, metric='euclidean', target_class=None, n_splits=5, random_state=42):
    X = np.array([d['Document'] for d in data])
    y = np.array([d['Label'] for d in data])

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    class_reports = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and fit KNeighborsClassifier
        if metric == 'euclidean':
            knn_classifier = KNeighborsClassifier(
                n_neighbors=k, metric='euclidean')
        elif metric == 'cosine':
            knn_classifier = KNeighborsClassifier(
                n_neighbors=k, metric='cosine')
        knn_classifier.fit(X_train, y_train)

        # Predict classes for the test set
        y_pred = knn_classifier.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(
            y_test, y_pred, average='weighted', zero_division=0.0)
        precision = precision_score(
            y_test, y_pred, average='weighted', zero_division=0.0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0.0)

        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

        # If target_class is specified, compute metrics for that specific class
        if target_class:
            class_report = classification_report(
                y_test, y_pred, labels=target_class, target_names=target_class, zero_division=0.0)
            return accuracy, recall, precision, f1, class_report

    # Calculate average metrics across folds
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    if target_class:
        avg_class_report = "\n\n".join(class_reports)
        return avg_accuracy, avg_recall, avg_precision, avg_f1, avg_class_report

    return avg_accuracy, avg_recall, avg_precision, avg_f1


def knn_classification(input, data, k, metric='euclidean', target_class=None):
    # Extract embeddings and labels from data
    X = [np.array(d['Document']) for d in data]
    y = [d['Label'] for d in data]

    # Create and fit KNeighborsClassifier
    if metric == 'euclidean':
        knn_classifier = KNeighborsClassifier(
            n_neighbors=k, metric='euclidean')
    elif metric == 'cosine':
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn_classifier.fit(X, y)

    # Predict class for the sample text
    if isinstance(input, dict):
        predicted_class = knn_classifier.predict([input])[0]
        return predicted_class
    elif isinstance(input, list):
        X_test = []
        y_test = []
        for sample in input:
            X_test.append(sample['Document'])
            y_test.append(sample['Label'])

        y_pred = knn_classifier.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(
            y_test, y_pred, average='weighted', zero_division=0.0)
        precision = precision_score(
            y_test, y_pred, average='weighted', zero_division=0.0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0.0)

        # If target_class is specified, compute metrics for that specific class
        if target_class:
            class_report = classification_report(
                y_test, y_pred, labels=target_class, target_names=target_class, zero_division=0.0)
            return accuracy, recall, precision, f1, class_report

        return accuracy, recall, precision, f1


if __name__ == '__main__':
    args = parse_arguments()
    file_path = f'{DATASET_DIR}/mtsamples.csv'
    df = pd.read_csv(file_path, index_col=0)
    for _, row in df.iterrows():
        row['medical_specialty'] = row['medical_specialty'][1:]

    mtsamples = json.loads(df.to_json(orient='records', indent=2))
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    if not os.path.exists(f'{DATASET_DIR}/mtsamples_encoded.json'):
        mtsamples_encoded = encode_data(mtsamples)
        with open(f'{DATASET_DIR}/mtsamples_encoded.json', 'w') as json_file:
            json.dump(mtsamples_encoded, json_file)
    else:
        with open(f'{DATASET_DIR}/mtsamples_encoded.json') as json_file:
            mtsamples_encoded = json.load(json_file)

    test_data = args.dataset
    with open(f'{OUTPUT_DIR}/{test_data}') as json_file:
        test_data = json.load(json_file)
        test_data = encode_data(test_data)

    target_class = df['medical_specialty'].unique().tolist()
    accuracy, recall, precision, f1, class_report = knn_classification(
        test_data, mtsamples_encoded, 5, 'cosine', target_class=target_class)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall:, {recall:.4f}")
    print(f"Precision:, {precision:.4f}")
    print(f"F1-score:, {f1:.4f}")
    print("Classification Report for", target_class, "class:")
    print(class_report)
