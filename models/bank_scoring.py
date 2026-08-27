import torch
from sklearn.cluster import MiniBatchKMeans
import tqdm

# Shared memory-bank construction and scoring for the *BankEmbeddingPAI detectors.
# metric='cosine': embeddings are L2-normalized, score = 1 - mean top-k cosine similarity.
# metric='euclidean': embeddings are NOT normalized (norm may carry amplitude),
#                     score = mean of the k smallest L2 distances to the bank.
# Following the official PAI code, the k-means coreset is always clustered in
# L2-normalized space but the bank stores the embeddings in their scoring space.


def _prepare_embeddings(embeddings, metric):
    if metric == 'cosine':
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
    return embeddings


def score_against_bank(embeddings, bank_embedding, metric='cosine', top_k=3):
    embeddings = embeddings.to(bank_embedding.device)
    if metric == 'cosine':
        cos_similarity = torch.matmul(embeddings, bank_embedding.T)
        topk_values, _ = torch.topk(cos_similarity, k=top_k, dim=-1, largest=True)
        scores = (1 - topk_values).mean(dim=-1)
    elif metric == 'euclidean':
        distances = torch.cdist(embeddings, bank_embedding, p=2)
        topk_values, _ = torch.topk(distances, k=top_k, dim=-1, largest=False)
        scores = topk_values.mean(dim=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return scores


def get_bank_embedding(encoder, dataloader, num_cores=128, metric='cosine', device=None, return_embeddings=False):
    # return_embeddings=True additionally returns all train embeddings (in scoring space, on cpu),
    # e.g. to compute the bank scores of the train windows without a second embedding pass
    encoder.eval()

    all_embeddings = []
    with torch.inference_mode():
        for batch, _ in tqdm.tqdm(dataloader, total=len(dataloader), leave=True, desc="Bank embeddings"):
            if device is not None:
                batch = batch.to(device)
            embeddings = encoder(batch)
            all_embeddings.append(embeddings.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    if metric == 'cosine':
        all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1, p=2)

    n_samples = all_embeddings.size(0)

    if num_cores > n_samples:
        print(f"Number of cores ({num_cores}) is greater than number of samples ({n_samples}), returning all embeddings without clustering.")
        return (all_embeddings, all_embeddings) if return_embeddings else all_embeddings

    cluster_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1, p=2, eps=1e-12)
    kmeans = MiniBatchKMeans(
        n_clusters=num_cores,
        init='k-means++',
        random_state=42,
        batch_size=max(8192, num_cores),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01
    )
    kmeans.fit(cluster_embeddings.numpy())
    centers = torch.tensor(kmeans.cluster_centers_, dtype=cluster_embeddings.dtype)
    distances = torch.cdist(cluster_embeddings, centers, p=2)
    core_indices = torch.argmin(distances, dim=0)
    bank_embedding = all_embeddings[core_indices]
    return (bank_embedding, all_embeddings) if return_embeddings else bank_embedding


def compute_anomaly_score(encoder, test_loader, bank_embedding, metric='cosine', top_k=3, device=None):
    assert bank_embedding.size(0) >= top_k, f"top_k ({top_k}) must be less than or equal to the number of bank embeddings ({bank_embedding.size(0)})"
    encoder.eval()

    all_scores = []
    with torch.inference_mode():
        for batch, _ in tqdm.tqdm(test_loader, total=len(test_loader), leave=True, desc="Anomaly scores"):
            if device is not None:
                batch = batch.to(device)
            embeddings = encoder(batch)
            embeddings = _prepare_embeddings(embeddings, metric)
            scores = score_against_bank(embeddings, bank_embedding, metric=metric, top_k=top_k)
            all_scores.append(scores.cpu())
    all_scores = torch.cat(all_scores, dim=0)
    return all_scores.numpy()
