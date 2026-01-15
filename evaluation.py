"""
Evaluation utilities for diffu2kg project.
"""
import json
import numpy as np
import torch
from typing import List, Tuple


def calculate_distance(a, b):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def evaluate_predictions(
    decoded_vectors: List[np.ndarray],
    ground_true_vectors: List[np.ndarray],
    entity_embeddings_path: str,
    val_data_path: str,
) -> dict:
    """
    Evaluate predictions using entity embeddings and ground truth.
    
    Args:
        decoded_vectors: List of predicted entity vectors
        ground_true_vectors: List of ground truth entity vectors
        entity_embeddings_path: Path to entity embeddings JSON file
        val_data_path: Path to validation/test data file
        
    Returns:
        Dictionary containing evaluation metrics (hit@1, hit@3, hit@10, mrr)
    """
    with open(entity_embeddings_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    eva_matrix = []
    for v in decoded_vectors:
        eva_vector = []
        for i, k in enumerate(entities):
            eva_vector.append((i, calculate_distance(v, k)))
        eva_vector = sorted(eva_vector, key=lambda x: x[1])
        eva_matrix.append([a for a, b in eva_vector])
    
    data = []
    with open(val_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip().split('\t'))
    
    true_id = []
    for p in data:
        true_id.append(int(p[2]))
    
    ranks = []
    for i, arr in enumerate(eva_matrix):
        x = true_id[i]
        for j in range(len(arr)):
            if x == arr[j]:
                ranks.append(j + 1)
                break
    
    ranks = torch.tensor(ranks)
    mrr = torch.mean(1.0 / ranks.float())
    hit1 = torch.mean((ranks <= 1).float())
    hit3 = torch.mean((ranks <= 3).float())
    hit10 = torch.mean((ranks <= 10).float())
    
    return {
        "hit@1": hit1.item(),
        "hit@3": hit3.item(),
        "hit@10": hit10.item(),
        "mrr": mrr.item()
    }


def save_results(decoded_vectors: np.ndarray, ground_true_vectors: np.ndarray, out_dir: str):
    """
    Save evaluation results to disk.
    
    Args:
        decoded_vectors: Predicted vectors
        ground_true_vectors: Ground truth vectors
        out_dir: Output directory path
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "decoded_vectors.npy"), decoded_vectors)
    np.save(os.path.join(out_dir, "ground_true_vectors.npy"), ground_true_vectors)
