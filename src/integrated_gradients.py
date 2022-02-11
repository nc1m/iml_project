# Standrad lib
import os
import logging
import argparse
import webbrowser

# Own modules
import bert_datasets

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torchtext

from captum.attr import TokenReferenceBase
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization

#from IPython.core.display import HTML, display

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


MODEL_CHOICES = dict()
# Predicts NEGATIVE/POSITIVE sentiment fine tunde on SST (2 classes)
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you
MODEL_CHOICES['sentimentSST2'] = 'distilbert-base-uncased-finetuned-sst-2-english'

# Predicts 1 star/2 stars/3 stars/4 stars/ 5 stars sentiment for product reviews (5 classes)
# https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you
MODEL_CHOICES['proudctReviews5'] = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Predicts negative/positive/neutral sentiment for financial texts (3 classes)
# https://huggingface.co/ProsusAI/finbert?text=Stocks+rallied+and+the+British+pound+gained.
MODEL_CHOICES['sentimentFin3'] = 'ProsusAI/finbert'

# # Predicts anger/disgust/fear/joy/neutral/sadness/surprise emotions (7 classes)
# # https://huggingface.co/j-hartmann/emotion-english-distilroberta-base?text=This+movie+always+makes+me+cry..
# No suitable test dataset found
# MODEL_CHOICES['emotion7'] = 'j-hartmann/emotion-english-distilroberta-base'

# Predicts sadness/joy/love/anger/fear/surprise emotion (6 classes)
# https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion?text=I+like+you.+I+love+you
MODEL_CHOICES['emotion6'] = 'bhadresh-savani/distilbert-base-uncased-emotion'

# Predicts POS/NEG/NEU sentiment trained on tweets (3 classes)
# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis?text=I+like+you.+I+love+you
# No suitable test dataset found
# MODEL_CHOICES['sentimentTweet3'] = 'finiteautomata/bertweet-base-sentiment-analysis'

BASELINE_TYPES = ['constant', 'maxDist', 'blurred', 'uniform', 'gaussian']


def parse_args():
    parser = argparse.ArgumentParser(description='TODO: Add general description AND add model and baseline type description here (why here? because newlines are possible here.\ndemo1\ndemo2)', formatter_class=argparse.RawDescriptionHelpFormatter)  # TODO read description
    parser.add_argument('-m', '--model',
                        default='sentimentSST2',
                        type=str,
                        choices=[*list(MODEL_CHOICES.keys())],
                        help='Set BERT-based text classification model. Possible values are: '+', '.join(list(MODEL_CHOICES.keys())),
                        metavar='MODEL')
    parser.add_argument('-b', '--baselineType',
                        default='constant',
                        type=str,
                        choices=BASELINE_TYPES,
                        help='Set which baseline type will be used. Possible values are: '+', '.join(BASELINE_TYPES),
                        metavar='BASELINE_TYPE')
    # parser.add_argument('-o', '--outputDir', default='./', type=str, help='TODO')
    parser.add_argument('-n', '--numSamples', default=10, type=int, help='Number of samples')
    parser.add_argument('--onlyFalse', action='store_true', help='Set this if only wrong predicted samples should be shown')
    parser.add_argument('--no_cuda', action='store_true', help='Set this if cuda is availbable, but you do NOT want to use it.')
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

def calc_Attribution(token_reference, inputs, sample,lig, probs, tokenizer, id2label, label_idx, pad_token, vis_result):
    reference_indices = token_reference.generate_reference(inputs['input_ids'].shape[1], device=inputs['input_ids'].device).unsqueeze(0)

    # for key in inputs:
    #     inputs[key] = inputs[key].unsqueeze(0)
    # print("inputs['input_ids'].shape:", inputs['input_ids'].shape)
    # print('reference_indices.shape:', reference_indices.shape)
    # print(inputs['input_ids'].dtype)
    # print(reference_indices.dtype)
    # attributions_ig, delta = lig.attribute(inputs=(inputs['input_ids'], inputs['attention_mask']), baselines=reference_indices, target=label, n_steps=500, return_convergence_delta=True)
    attributions_ig, delta = lig.attribute(inputs=inputs['input_ids'],
                                        baselines=reference_indices,
                                        additional_forward_args=inputs['attention_mask'],
                                        target=sample['label'],
                                        n_steps=10,
                                        return_convergence_delta=True)

    attributions = attributions_ig.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    prob = torch.max(probs).item()
    # print(attributions)
    # print(probs)
    # print(id2label[label_idx.item()])
    # print(dataset.features["label"].int2str(sample['label']))
    # print(pad_token)
    # print(attributions.sum())
    # print(len(sample['inputString']))
    # print(delta)

    text = [tokenizer.decode(x) for x in inputs['input_ids']][0].split(' ')
        
    vis_result.append(visualization.VisualizationDataRecord(attributions,
                                                        prob,
                                                        id2label[label_idx.item()],
                                                        id2label[sample['label']],
                                                        pad_token,
                                                        attributions.sum(),
                                                        text,
                                                        delta))

def main():
    set_seed(42)
    args = parse_args()
    print(args)
    use_cuda = False
    
    if torch.cuda.is_available() and not args.no_cuda:
        pass
        # use_cuda = True
    print('CUDA enabled:', use_cuda)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHOICES[args.model])
    # print(model.config)
    id2label = model.config.to_dict()['id2label']
    print(f'Labels: {id2label}')
    if use_cuda:
        model = model.cuda()
    model.eval()

    # Load tokenizer and get pad tokenls
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHOICES[args.model])
    if tokenizer._pad_token is not None:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.encode(pad_token)
        print(f'PAD_TOKEN:\t{pad_token}')
        print(f'PAD_TOKEN_ID:\t{pad_token_id}')
    else:
        logging.error("Using pad_token, but it is not set yet.")
    p_special_tokens(tokenizer)
    # print(tokenizer.decode(0))
    # print(tokenizer.decode(101))
    # print(tokenizer.decode(102))
    # print(tokenizer.decode(103))

    token_reference = TokenReferenceBase(reference_token_idx=pad_token_id[0])
    # vocab = torchtext.vocab.vocab(tokenizer.get_vocab())

    # Custom forward function
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

    dataset = bert_datasets.build_dataset(args.model, tokenizer)
    dataset = dataset.shuffle()
    # help(dataset.features['label'])
    # print(dataset.features['label']._int2str)
    # print(dataset.features['label']._str2int)

    #print(dataset.info.features)
    #print(dataset[0])

    vis_result = []
    numAttr = 0
    for sample in dataset:
        #print(sample)
        inputs = tokenizer(sample['inputString'],
                           padding=True,
                           truncation=True,
                           max_length=512,
                           return_tensors="pt")
        #print(f'INPUTS:\n {inputs}')

        model.zero_grad()
        if use_cuda:
            inputs = inputs.cuda()
        outputs = model(**inputs)
        #print(outputs)
        probs = torch.softmax(outputs.logits, dim=1)
        #print(probs)
        label_idx = torch.argmax(probs, dim=1)
        #print("label_idx",label_idx)
        #print("inputs['input_ids'].shape:", inputs['input_ids'].shape)        

        #TODO: DELETE THIS######
        sample['label']+=1
        ########################
        print( id2label[label_idx.item()], id2label[sample['label']])
        if args.onlyFalse and id2label[label_idx.item()] != id2label[sample['label']]:
            calc_Attribution(token_reference, inputs, sample,lig, probs, tokenizer, id2label, label_idx, pad_token, vis_result)
        elif not args.onlyFalse:
            calc_Attribution(token_reference, inputs, sample,lig, probs, tokenizer, id2label, label_idx, pad_token, vis_result)
            
        numAttr += 1
        if numAttr >= args.numSamples:
            break
        
    data = visualization.visualize_text(vis_result)
    # print(data.data)
    with open("data.html", "w") as file:
        file.write(data.data)

    webbrowser.open_new('data.html')




if __name__ == '__main__':
    main()
