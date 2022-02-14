import numpy as np
import torch


def get_closest_id_from_emb(emb: torch.Tensor, model, *args):
    """Given an embedding returns the clostest token_id, with regard to it's
    embedding.

    Args:
        emb (torch.Tensor): Proposed 'free' embeddings
        model (_type_): Pretrained bert/destilbert

    Returns:
        int: Returns the token id for the found embedding
    """
    token_id = None
    embeddings = model.get_input_embeddings().weight.detach().clone()
    minDist = np.inf
    for i, curEmb in enumerate(embeddings):
        # torch.sqrt(torch.sum((curEmb - emb) ** 2))
        dist = (curEmb - emb).pow(2).sum().sqrt()
        if dist < minDist:
            token_id = i
            minDist = dist
    return token_id
