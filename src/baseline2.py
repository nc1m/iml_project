import torch
import numpy as np
from src.utils import get_closest_id_from_emb


def input_ids_to_sentence(input_ids, tokenizer):
    """Convert input ids to words

    Args:
        input_ids : Tensor of input ids
        tokenizer : Used tokenizer

    Returns:
        List: Words decoded from their ids
    """
    sentence = []
    for token_id in input_ids:
        sentence.append(tokenizer.decode(token_id))
    return sentence


def create_uniform_embedding(input_ids, tokenizer):
    """Creates a baseline uniformly sampled from word space

    Args:
        input_ids (_type_): Tensor of input ids
        tokenizer (_type_): Used tokenizer

    Returns:
        torch.Tensor: Input ids for the calculated baseline
    """
    sentence = input_ids_to_sentence(input_ids, tokenizer)

    baseline_sentence = []
    for word in sentence:
        if word in tokenizer.all_special_tokens:
            baseline_sentence.append(torch.tensor(tokenizer.encode(word)[1]))
            continue
        baseline_sentence.append(
            np.random.randint(1, len(tokenizer.get_vocab()))
            )
    return torch.tensor([baseline_sentence])


def create_gaussian_embedding(input_ids, tokenizer, model):
    """Creates a baseline using drawing samples from gaussian distribution
     around input

    Args:
        input_ids (_type_): Tensor of input ids
        tokenizer (_type_): Used tokenizer
        model (_type_): Pretrained Bert/Destilbert model

    Returns:
        torch.Tensor: Input ids for the calculated baseline
    """
    sentence = input_ids_to_sentence(input_ids, tokenizer)

    full_embs = model.get_input_embeddings().weight.detach().clone()
    mean = torch.mean(full_embs).cpu()
    std = torch.std(full_embs).cpu()

    baseline_sentence = []
    for word in sentence:

        if word in tokenizer.all_special_tokens:
            baseline_sentence.append(torch.tensor(tokenizer.encode(word)[1]))
            continue

        sampled_emb = torch.Tensor(
            np.random.normal(mean, std, size=full_embs.shape[1])
            )
        baseline_sentence.append(get_closest_id_from_emb(sampled_emb, model))
    return torch.tensor([baseline_sentence])


def create_max_distance_baseline(input_ids, tokenizer, model):
    """Creates a baseline which has a maximum distance to the input

    Args:
        input_ids (_type_): Tensor of input ids
        tokenizer (_type_): Used tokenizer
        model (_type_): Pretrained Bert/Destilbert model

    Returns:
        torch.Tensor: Input ids for the calculated baseline
    """
    full_embs = model.get_input_embeddings().weight.detach().clone()
    lowest_emb_val = full_embs.min(0).values
    highest_emb_val = full_embs.max(0).values
    between = (highest_emb_val + lowest_emb_val) / 2

    baseline_sentence = []
    for word in input_ids:
        # Skip special tokens
        if word in tokenizer.all_special_ids:
            baseline_sentence.append(word)
            continue

        word_emb = model.get_input_embeddings()(word)

        # Set all values to the maximum if value is closer to the minimum
        masked = word_emb > between
        masked_emb = torch.zeros_like(word_emb)
        masked_emb[masked is True] = lowest_emb_val[masked is True]
        masked_emb[masked is False] = highest_emb_val[masked is False]

        baseline_sentence.append(get_closest_id_from_emb(masked_emb, model))

    return torch.tensor([baseline_sentence])
