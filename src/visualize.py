import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from captum.attr import TokenReferenceBase
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from customized_lig import CustomizedLayerIntegratedGradients

BASELINE_TYPES = ['constant', 'maxDist', 'blurred', 'uniform', 'gaussian']
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        default='sentimentSST2',
                        type=str,
                        choices=[*list(MODEL_CHOICES.keys())],
                        help='Set BERT-based text classification model. Possible values are: '+', '.join(list(MODEL_CHOICES.keys())),
                        metavar='MODEL')
    parser.add_argument('--method', default='ig', type=str, choices=['ig', 'dig'])
    return parser.parse_args()


def main():
    args = parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHOICES[args.model])
    model.eval()
    # Custom forward function
    def custom_forward(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    # Load tokenizer and get pad tokenls
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHOICES[args.model])
    if tokenizer._pad_token is not None:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.encode(pad_token)
        print(f'PAD_TOKEN:\t{pad_token}')
        print(f'PAD_TOKEN_ID:\t{pad_token_id}')
    else:
        raise ValueError("Using pad_token, but it is not set yet.")

    xg = create_method(args.method, custom_forward, model.get_input_embeddings())
    test_input_string = 'what a beautiful day'
    inputs = tokenizer(test_input_string, return_tensors="pt")

    # TODO diff baselines
    token_reference = TokenReferenceBase(reference_token_idx=pad_token_id[1])
    reference_indices = token_reference.generate_reference(inputs['input_ids'].shape[1], device=inputs['input_ids'].device).unsqueeze(0)
    n_steps = None
    tempData = xg.attribute(inputs=inputs['input_ids'],
                            baselines=reference_indices,
                            additional_forward_args=inputs['attention_mask'],
                            target=1,
                            # n_steps=n_steps,
                            return_convergence_delta=False)

    _, intrplIds, gradAtIntrpl, cmlGrad = tempData
    n_steps = intrplIds.shape[0]
    def summarize_attr(attr, n_steps):
        newVals = []
        for i in range(n_steps):
            temp = attr[i]
            temp = temp.sum(dim=1)
            temp = temp / torch.norm(temp)
            temp = temp.cpu().detach().numpy()
            newVals.append(temp)
        return np.vstack(newVals)

    # print(intrplIds.shape)
    intrplTokens = []
    for step in intrplIds:
        curStep = []
        for token_emb in step:
            # print(token_emb.shape)
            token_id = get_closest_id_from_emb(token_emb, model)
            # print(tokenizer.decode(token_id))
            curStep.append(tokenizer.decode(token_id))
        intrplTokens.append(curStep)
    # print(intrplTokens)

    gradAtIntrpl = summarize_attr(gradAtIntrpl, n_steps)
    cmlGrad = summarize_attr(cmlGrad, n_steps)
    sum_cmlGrad = np.sum(cmlGrad, axis=1)
    # print(f'sum_cmlGrad.shape: {sum_cmlGrad.shape}')
    sum_cmlGrad = np.cumsum(sum_cmlGrad, axis=0)
    # print(f'sum_cmlGrad.shape: {sum_cmlGrad.shape}')

    # print(f'intrplIds.shape: {intrplIds.shape}')
    # print(f'gradAtIntrpl.shape: {gradAtIntrpl.shape}')
    # print(f'cmlGrad.shape: {cmlGrad.shape}')

    seqLen = gradAtIntrpl.shape[1]

    alphas = np.linspace(0, 1, n_steps)
    fig, axs = plt.subplots(1, 3)
    print(range(n_steps))
    print(gradAtIntrpl[0].shape)


    blType = 'constant'

    fig.suptitle(f'input: {test_input_string}\nlabel=1\prediction=1')
    axs[0].set_title(f'Gradient at Interpolation')
    axs[0].set_ylim(bottom=gradAtIntrpl.min(), top=gradAtIntrpl.max())
    gradAtIntrpl_bar = axs[0].bar(range(seqLen), gradAtIntrpl[0], tick_label=intrplTokens[0])

    axs[1].set_title(f'Cumulative Gradients')
    axs[1].set_ylim(bottom=cmlGrad.min(), top=cmlGrad.max())
    cmlGrad_bar = axs[1].bar(range(seqLen), cmlGrad[0], tick_label=intrplTokens[0])

    axs[2].set_title(f'Sum of Cumulative Gradients')
    axs[2].set_ylim(bottom=sum_cmlGrad.min(), top=sum_cmlGrad.max())
    axs[2].set_xlim(left=alphas.min(), right=alphas.max())
    sum_cmlGrad_line = axs[2].plot(alphas[0], sum_cmlGrad[0])
    sum_cmlGrad_line = sum_cmlGrad_line[0]


    plt.subplots_adjust(bottom=0.25)
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    alphaSlider = Slider(slider_ax, 'alpha', 0.0, 1.0, 0.0, valstep=alphas)
    def update(val):
        if val == 0.0:
            indx = 0
        elif val == 1.0:
            indx = n_steps - 1
        else:
            indx = int(n_steps * val)

        sum_cmlGrad_line.set_xdata(alphas[:indx])
        sum_cmlGrad_line.set_ydata(sum_cmlGrad[:indx])

        axs[0].set_xticklabels(intrplTokens[indx])
        axs[1].set_xticklabels(intrplTokens[indx])
        for i in range(seqLen):
            gradAtIntrpl_bar[i].set_height(gradAtIntrpl[indx][i])
            cmlGrad_bar[i].set_height(cmlGrad[indx][i])

    alphaSlider.on_changed(update)
    plt.show()
    exit()

    # TODO WHEN BASELINE_TYPES IS IMPLEMENTED
    # data = dict()
    # for blType in BASELINE_TYPES:
    #     bl = create_baseline(blType)
    #     tempData = xg.attribute(inputs=inputs['input_ids'],
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
    main()
