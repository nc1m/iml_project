import numpy as np
import pickle
import torch

def get_closest_id_from_emb_old(emb, model):
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




def get_closest_id_from_emb(intrplEmbs, model, knn_filePath):
    """Given an embedding returns the clostest token_id, with regard to it's
    embedding.
    """

    print(f'intrplEmbs.shape: {intrplEmbs.shape} = step X seq X emb')
    startEmbs = intrplEmbs[0]
    print(f'startEmbs.shape: {startEmbs.shape} = seq X emb')
    targetEmb = intrplEmbs[-1]
    print(f'targetEmb.shape: {targetEmb.shape} = seq X emb')
    print('target shape: step X seq')
    with open(knn_filePath, 'rb') as f:
        [word_idx_map, word_features, adj] = pickle.load(f)
        word_idx_map = dict(word_idx_map)

    # word_idx_map = vocab
    # word_features = model embedding weights
    # adj = sparse distance matrix
    # print(type(adj))
    # print(adj[0])
    # print()
    intrplIds = torch.ones((intrplEmbs.shape[:2]))
    print(intrplIds)
    print(f'intrplIds.shape: {intrplIds.shape}')
    embeddings = model.get_input_embeddings().weight.detach().clone()
    # find initial starting id
    for i_seq, seqEmb in enumerate(startEmbs):
        # print(seqEmb.shape)
        token_id = None
        minDist = np.inf
        for curId, curEmb in enumerate(embeddings):
            dist = (curEmb - seqEmb).pow(2).sum().sqrt()  # torch.sqrt(torch.sum((curEmb - emb) ** 2)) #
            if dist < minDist:
                token_id = curId
                minDist = dist
        intrplIds[0][i_seq] = token_id
    print(intrplIds)
    print(intrplIds[0])
    # use starting id and knn to build path to end embedding
    # Go over steps
    for i_step, stepIds in enumerate(intrplIds):
        if i_step == 0:
            continue
        # Go over seq in steps
        for i_seq, seqId in enumerate(stepIds):
            # Get previous token id
            prevId = intrplIds[i_step-1][i_seq]
            # Get nearest neighbors of prev id these are candidates for the next token id
            candidates = adj[prevId]
            candidates = candidates.tocoo().col
            # print(candidates)

            token_id = None
            minDist = np.inf
            # Search for candidate thats closest to input emb (last interpolation step)
            for cand in candidates:
                candEmb = embeddings[cand]
                dist = (candEmb - targetEmb[i_seq]).pow(2).sum().sqrt()  # torch.sqrt(torch.sum((curEmb - emb) ** 2)) #
                if dist < minDist and cand != prevId:
                    if i_step > 1:
                        if cand != intrplIds[i_step-2][i_seq]:
                            token_id = cand
                            minDist = dist
                    else:
                        token_id = cand
                        minDist = dist
            intrplIds[i_step, i_seq] = token_id
        print(intrplIds)


    exit()
    token_id = None
    embeddings = model.get_input_embeddings().weight.detach().clone()
    minDist = np.inf
    for i, curEmb in enumerate(embeddings):
        dist = (curEmb - emb).pow(2).sum().sqrt()  # torch.sqrt(torch.sum((curEmb - emb) ** 2)) #
        if dist < minDist:
            token_id = i
            minDist = dist
    return token_id
