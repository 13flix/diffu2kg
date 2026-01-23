import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def save_results(cands, ground_true_samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'decoded_vectors.npy'), cands)
    np.save(os.path.join(out_dir, 'ground_true_vectors.npy'), ground_true_samples)


def evaluate_predictions(decoded_vectors, ground_true_vectors, entity2emb_path, test_path):
    with open(entity2emb_path, 'r', encoding='utf-8') as f:
        entity_embeddings = json.load(f)
    
    entity_embeddings = np.array(entity_embeddings)
    
    hit1 = 0.0
    hit3 = 0.0
    hit10 = 0.0
    mrr = 0.0
    
    num_samples = len(decoded_vectors)
    
    for i in range(num_samples):
        pred_vector = decoded_vectors[i]
        true_vector = ground_true_vectors[i]
        
        similarities = cosine_similarity([pred_vector], entity_embeddings)[0]
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        pred_similarity = cosine_similarity([pred_vector], [true_vector])[0][0]
        
        true_rank = None
        for rank, idx in enumerate(sorted_indices):
            if np.allclose(entity_embeddings[idx], true_vector, atol=1e-6):
                true_rank = rank + 1
                break
        
        if true_rank is None:
            true_rank = len(sorted_indices)
        
        if true_rank <= 1:
            hit1 += 1
        if true_rank <= 3:
            hit3 += 1
        if true_rank <= 10:
            hit10 += 1
        
        mrr += 1.0 / true_rank
    
    metrics = {
        'hit@1': hit1 / num_samples,
        'hit@3': hit3 / num_samples,
        'hit@10': hit10 / num_samples,
        'mrr': mrr / num_samples
    }
    
    return metrics
