import torch
import numpy as np
from src.utils import get_closest_id_from_emb


def input_ids_to_sentence(input_ids, tokenizer):
    sentence = []
    for token_id in input_ids:
        sentence.append(tokenizer.decode(token_id))
    return sentence

def create_uniform_embedding(input_ids, tokenizer):
    """ Randomly (uniform) select a token for each word in the sentence

        Args:
            sentence (str): [description]
    """
    sentence = input_ids_to_sentence(input_ids, tokenizer)
    # sentence = []
    # for token_id in input_ids:
    #     sentence.append(tokenizer.decode(token_id))

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
    sentence = input_ids_to_sentence(input_ids, tokenizer)
    # sentence = []
    # # print(input_ids)
    # for token_id in input_ids:
    #     # print(sentence)
    #     sentence.append(tokenizer.decode(token_id))

    full_embs = model.get_input_embeddings().weight.detach().clone()
    mean = torch.mean(full_embs).cpu()
    std = torch.std(full_embs).cpu()
    # sentence = tokenizer.tokenize(sentence)

    baseline_sentence = []
    for word in sentence:

        if word in tokenizer.all_special_tokens:
            baseline_sentence.append(torch.tensor(tokenizer.encode(word)[1]))
            continue

        sampled_emb = torch.Tensor(np.random.normal(mean, std, size=full_embs.shape[1]))
        baseline_sentence.append(get_closest_id_from_emb(sampled_emb, model))
    return torch.tensor([baseline_sentence])


def create_max_distance_baseline(input_ids, tokenizer, model):
    """[summary]

    Args:
        sentence (torch.Tensor): [description]
    """
    full_embs = model.get_input_embeddings().weight.detach().clone()
    lowest_emb_val = full_embs.min(0).values
    highest_emb_val = full_embs.max(0).values
    between = (highest_emb_val + lowest_emb_val) / 2
    # sentence = input_ids_to_sentence(input_ids, tokenizer)
    # sentence = []
    # # print(input_ids)
    # for token_id in input_ids:
    #     # print(sentence)
    #     sentence.append(tokenizer.decode(token_id))

    baseline_sentence = []
    for word in input_ids:

        if word in tokenizer.all_special_ids:
            baseline_sentence.append(word)
            continue

        word_emb = model.get_input_embeddings()(word)

        # Set all values to the maximum if value is closer to the minimum (since we want max distance)
        masked = word_emb > between
        masked_emb = torch.zeros_like(word_emb)
        masked_emb[masked == True] = lowest_emb_val[masked == True]
        masked_emb[masked == False] = highest_emb_val[masked == False]

        baseline_sentence.append(get_closest_id_from_emb(masked_emb, model))

    return torch.tensor([baseline_sentence])
