import numpy as np

def get_closest_id_from_emb(emb, model):
    """Given an embedding returns the clostest token_id, with regard to it's
    embedding.
    """
    token_id = None
    embeddings = model.get_input_embeddings().weight.detach().clone()
    minDist = np.inf
    for i, curEmb in enumerate(embeddings):
        dist = (curEmb - emb).pow(2).sum().sqrt() #  torch.sqrt(torch.sum((curEmb - emb) ** 2)) #
        if dist < minDist:
            token_id = i
            minDist = dist
    return token_id
