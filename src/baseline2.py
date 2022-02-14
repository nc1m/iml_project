import torch
import numpy as np
from utils import get_closest_id_from_emb


def create_uniform_embedding(input_ids, tokenizer):
    """ Randomly (uniform) select a token for each word in the sentence

        Args:
            sentence (str): [description]
    """
    sentence = []
    for token_id in input_ids:
        sentence.append(tokenizer.decode(token_id))

    baseline_sentence = []
    for word in sentence:
        if word in tokenizer.all_special_tokens:
            baseline_sentence.append(torch.tensor(tokenizer.encode(word)[1]))
            continue
        baseline_sentence.append(np.random.randint(1, len(tokenizer.get_vocab())))
    return torch.tensor([baseline_sentence])


def create_gaussian_embedding(input_ids, tokenizer, model):
    """_summary_

    Args:
        input (str): _description_

    Returns:
        int: _description_
    """
    sentence = []
    # print(input_ids)
    for token_id in input_ids:
        # print(sentence)
        sentence.append(tokenizer.decode(token_id))

    full_embs = model.get_input_embeddings().weight.detach().clone()
    mean = torch.mean(full_embs)
    std = torch.std(full_embs)
    # sentence = tokenizer.tokenize(sentence)

    baseline_sentence = []
    for word in sentence:

        if word in tokenizer.all_special_tokens:
            baseline_sentence.append(torch.tensor(tokenizer.encode(word)[1]))
            continue

        sampled_emb = torch.Tensor(np.random.normal(mean, std, size=full_embs.shape[1]))
        baseline_sentence.append(get_closest_id_from_emb(sampled_emb, model))
    return torch.tensor([baseline_sentence])
