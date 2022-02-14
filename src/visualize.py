import argparse
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from captum.attr import TokenReferenceBase
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from customized_lig import CustomizedLayerIntegratedGradients
import baseline2  # import Baseline
from bert_datasets import build_dataset
from utils import get_closest_id_from_emb

BASELINE_TYPES = ['constant', 'uniform', 'gaussian']  # ['constant', 'maxDist', 'blurred', 'uniform', 'gaussian']
MODEL_CHOICES = dict()
# Predicts NEGATIVE/POSITIVE sentiment fine tunde on SST (2 classes)
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you
MODEL_CHOICES['sentimentSST2'] = 'distilbert-base-uncased-finetuned-sst-2-english'

# Predicts 1 star/2 stars/3 stars/4 stars/ 5 stars sentiment for product reviews (5 classes)
# https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you
MODEL_CHOICES['productReviews5'] = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Predicts negative/positive/neutral sentiment for financial texts (3 classes)
# https://huggingface.co/ProsusAI/finbert?text=Stocks+rallied+and+the+British+pound+gained.
MODEL_CHOICES['sentimentFin3'] = 'ProsusAI/finbert'

# Predicts sadness/joy/love/anger/fear/surprise emotion (6 classes)
# https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion?text=I+like+you.+I+love+you
MODEL_CHOICES['emotion6'] = 'bhadresh-savani/distilbert-base-uncased-emotion'



def create_method(method, forward, input_embeddings):
    if method == 'ig':
        custom_ig = CustomizedLayerIntegratedGradients(forward, input_embeddings)  # LayerIntegratedGradients(forward, input_embeddings)
    elif method == 'dig':
        raise NameError(f'Method for "{method}" not YET implemented')
        custom_dig = 'LayerDiscreteIntegratedGradients(forward, input_embeddings'
    else:
        raise NameError(f'Method for "{method}" not implemented')
    return custom_ig

# TODO modelName necessarry?
def create_baseline(blType, padTokenId, inputString, inputs, tokenizer, model):
    if blType == 'constant':
        token_reference = TokenReferenceBase(reference_token_idx=padTokenId)
        bl = token_reference.generate_reference(inputs['input_ids'].shape[1], device=inputs['input_ids'].device).unsqueeze(0)
    elif blType == 'uniform':
        bl = baseline2.create_uniform_embedding(inputs['input_ids'][0], tokenizer)
    elif blType == 'gaussian':
        bl = baseline2.create_gaussian_embedding(inputs['input_ids'][0], tokenizer, model)
    elif blType == '':
        pass
    else:
        raise NameError(f'Baseline type {blType} not implemented')
    return bl

def summarize_attr(attr, n_steps):
    newVals = []
    for i in range(n_steps):
        temp = attr[i]
        temp = temp.sum(dim=1)
        temp = temp / torch.norm(temp)
        temp = temp.cpu().detach().numpy()
        newVals.append(temp)
    return np.vstack(newVals)


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
    parser.add_argument('-m', '--model',
                        default='sentimentSST2',
                        type=str,
                        choices=[*list(MODEL_CHOICES.keys())],
                        help='Set BERT-based text classification model. Possible values are: '+', '.join(list(MODEL_CHOICES.keys())),
                        metavar='MODEL')
    parser.add_argument('--method', default='ig', type=str, choices=['ig', 'dig'])
    parser.add_argument('-s', '--seed', default=42, type=int, help='TODO')
    parser.add_argument('-n', '--numSteps', default=10, type=int, help='TODO')
    parser.add_argument('--no_cuda', action='store_true', help='Set this if cuda is availbable, but you do NOT want to use it.')
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    startTime = time.time()
    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        pass # TODO
        # use_cuda = True
    print('CUDA enabled:', use_cuda)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHOICES[args.model])
    model.eval()

    # Dict that tanslates prediction id to class label
    id2label = model.config.id2label

    # Custom forward function
    def custom_forward(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    # Load tokenizer and get pad tokenls
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHOICES[args.model])
    if tokenizer._pad_token is not None:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.encode(pad_token)
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[1]
        # print(f'PAD_TOKEN:\t{pad_token}')
        # print(f'PAD_TOKEN_ID:\t{pad_token_id}')
    else:
        raise ValueError("Using pad_token, but it is not set yet.")

    # Load dataset
    dataset = build_dataset(args.model, tokenizer)
    dataset = dataset.shuffle()
    dataset_iter = iter(dataset)
    sample = next(dataset_iter)
    # print(sample)
    # sample = next(dataset_iter)
    # print(sample)

    xig = create_method(args.method, custom_forward, model.get_input_embeddings())
    inputs = tokenizer(sample['input_string'], return_tensors="pt")
    # print(inputs)
    model.zero_grad()
    if use_cuda:
        inputs = inputs.cuda()

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_pred = torch.argmax(probs, dim=1)

    # Things to update:
    # fig.suptitle: sample['input_string'], id2label, label, label_pred
    fig, axs = plt.subplots(len(BASELINE_TYPES), 3)
    # print(axs.shape)     # NUM_BL_TYPES X 3
    plt.tick_params(axis='x', which='major', labelsize=3)
    fig.suptitle(f'input: {sample["input_string"]}\nlabel={id2label[sample["label"]]}\nprediction={id2label[label_pred.item()]}')
    fig.tight_layout()
    axs[0, 0].set_title(f'Gradient at Interpolation')
    axs[0, 1].set_title(f'Cumulative Gradients')
    axs[0, 2].set_title(f'Sum of Cumulative Gradients')
    # todo store information from ig and axs in dict
    axs_bl = dict()
    intrplTokens_bl = dict()
    gradAtIntrpl_bl = dict()
    cmlGrad_bl = dict()
    sum_cmlGrad_bl = dict()
    gradAtIntrpl_bar_bl = dict()
    cmlGrad_bar_bl = dict()
    sum_cmlGrad_line_bl = dict()
    for i, blType in enumerate(BASELINE_TYPES):
        # Map baseline triple to blType
        axs_bl[blType] = axs[i]

        # print(blType)
        bl = create_baseline(blType, pad_token_id, sample['input_string'], inputs, tokenizer, model)
        # print(sample)
        # print()
        # print(inputs)
        # print()
        # print(bl)
        # print(inputs['input_ids'].shape)
        # print(bl.shape)


        attributions, intrplIds, gradAtIntrpl, cmlGrad = xig.attribute(inputs=inputs['input_ids'],
                                                                       baselines=bl,
                                                                       additional_forward_args=inputs['attention_mask'],
                                                                       target=sample['label'],
                                                                       n_steps=args.numSteps,
                                                                       return_convergence_delta=False)

        seqLen = gradAtIntrpl.shape[1]
        alphas = np.linspace(0, 1, args.numSteps)

        # print(gradAtIntrpl.shape)
        # print(args.numSteps)

        inputString = sample['input_string']
        label = sample['label']
        label_pred = label_pred

        # intrplTokens = []
        # for step in intrplIds:
        #     curStep = []
        #     for token_emb in step:
                # print(token_emb.shape)
        #         token_id = get_closest_id_from_emb(token_emb, model)
        #         # print(tokenizer.decode(token_id))
        #         curStep.append(tokenizer.decode(token_id))
        #     intrplTokens.append(curStep)
        # print(intrplTokens)
        intrplTokens = [['[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[unused174]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'], ['[unused174]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[unused814]', '[PAD]', '[PAD]', '[PAD]', '[unused174]'], ['[unused174]', '[unused533]', '[PAD]', '[unused814]', '[unused814]', '[unused814]', '[unused533]', '[unused814]', '[PAD]', '[unused155]', '[unused155]', '[unused155]', '[unused23]', '[unused814]', '[unused299]', '[unused814]', '[unused533]', '[unused814]', '[unused155]', '##訁'], ['[CLS]', 'even', 'die', '-', 'hard', 'fans', 'of', 'japanese', 'animation', '.', '.', '.', 'will', 'find', 'this', 'one', 'a', 'challenge', '.', '[SEP]'], ['[CLS]', 'even', 'die', '-', 'hard', 'fans', 'of', 'japanese', 'animation', '.', '.', '.', 'will', 'find', 'this', 'one', 'a', 'challenge', '.', '[SEP]'], ['[CLS]', 'even', 'die', '-', 'hard', 'fans', 'of', 'japanese', 'animation', '.', '.', '.', 'will', 'find', 'this', 'one', 'a', 'challenge', '.', '[SEP]'], ['[CLS]', 'even', 'die', '-', 'hard', 'fans', 'of', 'japanese', 'animation', '.', '.', '.', 'will', 'find', 'this', 'one', 'a', 'challenge', '.', '[SEP]'], ['[CLS]', 'even', 'die', '-', 'hard', 'fans', 'of', 'japanese', 'animation', '.', '.', '.', 'will', 'find', 'this', 'one', 'a', 'challenge', '.', '[SEP]']]
        intrplTokens_bl[blType] = intrplTokens

        gradAtIntrpl = summarize_attr(gradAtIntrpl, args.numSteps)
        cmlGrad = summarize_attr(cmlGrad, args.numSteps)
        sum_cmlGrad = np.sum(cmlGrad, axis=1)
        sum_cmlGrad = np.cumsum(sum_cmlGrad, axis=0)

        # Save data for updating
        gradAtIntrpl_bl[blType] = gradAtIntrpl
        cmlGrad_bl[blType] = cmlGrad
        sum_cmlGrad_bl[blType] = sum_cmlGrad

        axs[i][0].set_ylim(bottom=gradAtIntrpl.min(), top=gradAtIntrpl.max())
        gradAtIntrpl_bar = axs[i][0].bar(range(seqLen), gradAtIntrpl[0])
        gradAtIntrpl_bar_bl[blType] = gradAtIntrpl_bar
        axs[i][0].set_xticks(range(seqLen))
        axs[i][0].set_xticklabels(intrplTokens[0], rotation=90)
        axs[i][0].set_ylabel(f'{blType.capitalize()} Baseline Atrb.')

        axs[i][1].set_ylim(bottom=cmlGrad.min(), top=cmlGrad.max())
        cmlGrad_bar = axs[i][1].bar(range(seqLen), cmlGrad[0])
        cmlGrad_bar_bl[blType] = cmlGrad_bar
        axs[i][1].set_xticks(range(seqLen))
        axs[i][1].set_xticklabels(intrplTokens[0], rotation=90)

        axs[i][2].set_ylim(bottom=sum_cmlGrad.min(), top=sum_cmlGrad.max())
        axs[i][2].set_xlim(left=alphas.min(), right=alphas.max())
        sum_cmlGrad_line = axs[i][2].plot(alphas[0], sum_cmlGrad[0])
        sum_cmlGrad_line = sum_cmlGrad_line[0]
        sum_cmlGrad_line_bl[blType] = sum_cmlGrad_line

    plt.subplots_adjust(bottom=0.25)
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    alphaSlider = Slider(slider_ax, 'alpha', 0.0, 1.0, 0.0, valstep=alphas)
    def update(val):
        if val == 0.0:
            indx = 0
        elif val == 1.0:
            indx = args.numSteps - 1
        else:
            indx = int(args.numSteps * val)

        for blType in BASELINE_TYPES:
            sum_cmlGrad_line_bl[blType].set_xdata(alphas[:indx])
            sum_cmlGrad_line_bl[blType].set_ydata(sum_cmlGrad_bl[blType][:indx])

            axs_bl[blType][0].set_xticklabels(intrplTokens_bl[blType][indx], rotation=90) # TODO
            axs_bl[blType][1].set_xticklabels(intrplTokens_bl[blType][indx], rotation=90)
            for i in range(seqLen):
                gradAtIntrpl_bar_bl[blType][i].set_height(gradAtIntrpl_bl[blType][indx][i])
                cmlGrad_bar_bl[blType][i].set_height(cmlGrad_bl[blType][indx][i])

    alphaSlider.on_changed(update)
    plt.show()




    print(f'run time: {str(timedelta(seconds=(time.time() - startTime)))}')
######################################################################


        # token_reference = TokenReferenceBase(reference_token_idx=pad_token_id[1])
        # reference_indices = token_reference.generate_reference(inputs['input_ids'].shape[1], device=inputs['input_ids'].device).unsqueeze(0)
        # n_steps = None
        # tempData = xig.attribute(inputs=inputs['input_ids'],
        #                         baselines=reference_indices,
        #                         additional_forward_args=inputs['attention_mask'],
        #                         target=1,
        #                         # n_steps=n_steps,
        #                         return_convergence_delta=False)

        # _, intrplIds, gradAtIntrpl, cmlGrad = tempData
        # n_steps = intrplIds.shape[0]
        # seqLen = gradAtIntrpl.shape[1]
        # alphas = np.linspace(0, 1, n_steps)
        # def summarize_attr(attr, n_steps):
        #     newVals = []
        #     for i in range(n_steps):
        #         temp = attr[i]
        #         temp = temp.sum(dim=1)
        #         temp = temp / torch.norm(temp)
        #         temp = temp.cpu().detach().numpy()
        #         newVals.append(temp)
        #     return np.vstack(newVals)

        # # print(intrplIds.shape)
        # intrplTokens = []
        # for step in intrplIds:
        #     curStep = []
        #     for token_emb in step:
        #         # print(token_emb.shape)
        #         token_id = get_closest_id_from_emb(token_emb, model)
        #         # print(tokenizer.decode(token_id))
        #         curStep.append(tokenizer.decode(token_id))
        #     intrplTokens.append(curStep)
        # # print(intrplTokens)

        # gradAtIntrpl = summarize_attr(gradAtIntrpl, n_steps)
        # cmlGrad = summarize_attr(cmlGrad, n_steps)
        # sum_cmlGrad = np.sum(cmlGrad, axis=1)
        # # print(f'sum_cmlGrad.shape: {sum_cmlGrad.shape}')
        # sum_cmlGrad = np.cumsum(sum_cmlGrad, axis=0)
        # # print(f'sum_cmlGrad.shape: {sum_cmlGrad.shape}')

        # # print(f'intrplIds.shape: {intrplIds.shape}')
        # # print(f'gradAtIntrpl.shape: {gradAtIntrpl.shape}')
        # # print(f'cmlGrad.shape: {cmlGrad.shape}')

        # seqLen = gradAtIntrpl.shape[1]

        # alphas = np.linspace(0, 1, n_steps)
        # fig, axs = plt.subplots(1, 3)
        # print(range(n_steps))
        # print(gradAtIntrpl[0].shape)


        # blType = 'constant'

        # fig.suptitle(f'input: {sample["input_string"]}\nlabel=1\prediction=1')
        # axs[0].set_title(f'Gradient at Interpolation')
        # axs[0].set_ylim(bottom=gradAtIntrpl.min(), top=gradAtIntrpl.max())
        # gradAtIntrpl_bar = axs[0].bar(range(seqLen), gradAtIntrpl[0], tick_label=intrplTokens[0])

        # axs[1].set_title(f'Cumulative Gradients')
        # axs[1].set_ylim(bottom=cmlGrad.min(), top=cmlGrad.max())
        # cmlGrad_bar = axs[1].bar(range(seqLen), cmlGrad[0], tick_label=intrplTokens[0])

        # axs[2].set_title(f'Sum of Cumulative Gradients')
        # axs[2].set_ylim(bottom=sum_cmlGrad.min(), top=sum_cmlGrad.max())
        # axs[2].set_xlim(left=alphas.min(), right=alphas.max())
        # sum_cmlGrad_line = axs[2].plot(alphas[0], sum_cmlGrad[0])
        # sum_cmlGrad_line = sum_cmlGrad_line[0]


        # plt.subplots_adjust(bottom=0.25)
        # slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
        # alphaSlider = Slider(slider_ax, 'alpha', 0.0, 1.0, 0.0, valstep=alphas)
        # def update(val):
        #     if val == 0.0:
        #         indx = 0
        #     elif val == 1.0:
        #         indx = n_steps - 1
        #     else:
        #         indx = int(n_steps * val)

        #     sum_cmlGrad_line.set_xdata(alphas[:indx])
        #     sum_cmlGrad_line.set_ydata(sum_cmlGrad[:indx])

        #     axs[0].set_xticklabels(intrplTokens[indx])
        #     axs[1].set_xticklabels(intrplTokens[indx])
        #     for i in range(seqLen):
        #         gradAtIntrpl_bar[i].set_height(gradAtIntrpl[indx][i])
        #         cmlGrad_bar[i].set_height(cmlGrad[indx][i])

        # alphaSlider.on_changed(update)
        # plt.show()
        # exit()

    # TODO WHEN BASELINE_TYPES IS IMPLEMENTED
    # data = dict()
    # for blType in BASELINE_TYPES:
    #     bl = create_baseline(blType)
    #     tempData = xig.attribute(inputs=inputs['input_ids'],
    #                            baselines=reference_indices,
    #                            additional_forward_args=inputs['attention_mask'],
    #                            target=sample['label'],
    #                            n_steps=10,
    #                            return_convergence_delta=True)
    #     interpolated_ids, gradientsAt_interpolation, cumulativeGradients, sumOf_cumulativeGradients = tempData
    #     if blType not in data:
    #         data[blType] = dict()
    #     data[blType]['intrplIds'] = interpolated_ids
    #     data[blType]['gradAtIntrpl'] = gradientsAt_interpolation
    #     data[blType]['cmlGrad'] = cumulativeGradients
    #     data[blType]['sumCmlGrad'] = sumOf_cumulativeGradients

    # data['constant'] = dict()

    # prototype_input_string = 'this is a test input_string'


    # alphas = np.linspace(0, 1, n_steps)
    # # Create the figure and the line that we will manipulate
    # fig, axs = plt.subplots(5, 3)
    # constBl_axs = axs[0]
    # # constBl_bar_gradientsAt_interpolation = constBl_axs[0].bar(


    # # maxDistBl_axs = axs[1]
    # # blurrBl_axs = axs[2]
    # # unifBl_axs = axs[3]
    # # gausBl_axs = axs[4]
    # # print(axs.shape)






if __name__ == '__main__':
    args = parse_args()
    main(args)
