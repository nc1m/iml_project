import os
import logging
import argparse

import random
import numpy as np
import torch

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torchtext

from captum.attr import TokenReferenceBase
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization

# special tokens
SEPERATOR = '[SEP]'

##################################
# TODO: WORK IN JUPYTERLAB FIRST #
##################################

# models:
MODEL_NAMES = {0: 'distilbert-base-uncased-finetuned-sst-2-english'}

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', help='Set this if cuda is availbable, but you do NOT want to use it.', action='store_true')
    args = parser.parse_args()
    return args

def p_special_tokens(tokenizer):
    if tokenizer._bos_token is not None:
        print(tokenizer.bos_token, tokenizer.encode(tokenizer.bos_token))
    if tokenizer._eos_token is not None:
        print(tokenizer.eos_token, tokenizer.encode(tokenizer.eos_token))
    if tokenizer._unk_token is not None:
        print(tokenizer.unk_token, tokenizer.encode(tokenizer.unk_token))
    if tokenizer._sep_token is not None:
        print(tokenizer.sep_token, tokenizer.encode(tokenizer.sep_token))
    if tokenizer._pad_token is not None:
        print(tokenizer.pad_token, tokenizer.encode(tokenizer.pad_token))
    if tokenizer._cls_token is not None:
        print(tokenizer.cls_token, tokenizer.encode(tokenizer.cls_token))
    if tokenizer._mask_token is not None:
        print(tokenizer.mask_token, tokenizer.encode(tokenizer.mask_token))
    return None

def main():
    set_seed(42)
    args = parse_args()
    print(args)
    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        pass
        # use_cuda = True
    print('CUDA enabled:', use_cuda)

    modelName = 'distilbert-base-uncased-finetuned-sst-2-english'
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    print(model.config)
    id2label = model.config.to_dict()['id2label']
    if use_cuda:
        model = model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    if tokenizer._pad_token is not None:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.encode(pad_token)
        print(pad_token, pad_token_id)
    else:
        logging.error("Using pad_token, but it is not set yet.")
    p_special_tokens(tokenizer)
    print(tokenizer.decode(0))
    print(tokenizer.decode(101))
    print(tokenizer.decode(102))
    print(tokenizer.decode(103))

    token_reference = TokenReferenceBase(reference_token_idx=pad_token_id[0])
    # vocab = torchtext.vocab.vocab(tokenizer.get_vocab())

    def custom_forward(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
        # print(outputs)
        # probs = torch.softmax(outputs.logits, dim=1)
        # print(probs)
        # label_idx = torch.argmax(probs, dim=1)
        # print(label_idx)
        # return label_idx.unsqueeze(0)

    lig = LayerIntegratedGradients(custom_forward, model.get_input_embeddings())

    samples = [('It was a fantastic performance !', 1), ('Best film ever', 1), ('Such a great show!', 1), ('It was a horrible movie', 0), ('I\'ve never watched something as bad', 0), ('That is a terrible movie.', 0)]
    vis_result = []


    for sentence, label in samples:
        print(sentence, label)
        inputs = tokenizer(sentence,
                           padding=True,
                           truncation=True,
                           max_length=512,
                           return_tensors="pt")
        print(inputs)
        model.zero_grad()
        if use_cuda:
            inputs = inputs.cuda()
        outputs = model(**inputs)
        print(outputs)
        probs = torch.softmax(outputs.logits, dim=1)
        print(probs)
        label_idx = torch.argmax(probs, dim=1)
        print(label_idx)
        print("inputs['input_ids'].shape:", inputs['input_ids'].shape)
        reference_indices = token_reference.generate_reference(inputs['input_ids'].shape[1], device=inputs['input_ids'].device).unsqueeze(0)

        # for key in inputs:
        #     inputs[key] = inputs[key].unsqueeze(0)
        print("inputs['input_ids'].shape:", inputs['input_ids'].shape)
        print('reference_indices.shape:', reference_indices.shape)
        print(inputs['input_ids'].dtype)
        print(reference_indices.dtype)
        # attributions_ig, delta = lig.attribute(inputs=(inputs['input_ids'], inputs['attention_mask']), baselines=reference_indices, target=label, n_steps=500, return_convergence_delta=True)
        attributions_ig, delta = lig.attribute(inputs=inputs['input_ids'],
                                               baselines=reference_indices,
                                               additional_forward_args=inputs['attention_mask'],
                                               target=label,
                                               n_steps=10,
                                               return_convergence_delta=True)

        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        prob = torch.max(probs).item()
        print(attributions)
        print(probs)
        print(id2label[label_idx.item()])
        print(id2label[label])
        print(pad_token)
        print(attributions.sum())
        print(len(sentence))
        print(delta)

        text = [tokenizer.decode(x) for x in inputs['input_ids']][0].split(' ')

        # storing couple samples in an array for visualization purposes
        vis_result.append(visualization.VisualizationDataRecord(attributions,
                                                                prob,
                                                                id2label[label_idx.item()],
                                                                id2label[label],
                                                                pad_token,
                                                                attributions.sum(),
                                                                text,
                                                                delta))
    visualization.visualize_text(vis_result)


if __name__ == '__main__':
    main()
