import argparse
import os
import random
import time
import logging
from datetime import timedelta

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from captum.attr import TokenReferenceBase
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from src.customized_lig import CustomizedLayerIntegratedGradients
import src.baselines
from src.bert_datasets import build_dataset
from src.utils import get_closest_id_from_emb


BASELINE_TYPES = ['constant', 'uniform', 'gaussian', 'maxDist']  # ['constant', 'maxDist', 'blurred', 'uniform', 'gaussian']
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


def create_baseline(blType, padTokenId, inputString, inputs, tokenizer, model):
    """Choose a baseline and creates it according to an input

    Args:
        blType (_type_): constant,uniform,gaussian or maxdist
        padTokenId (_type_): Id for [pad]
        inputString (_type_): Proposed input as text
        inputs (_type_): Input ids
        tokenizer (_type_): Used pretrained tokenizer
        model (_type_): Used pretrained model

    Raises:
        NameError: Raised if baseline type not match witch proposed baselines

    Returns:
        torch.Tensor: Ids for the baseline
    """
    if blType == 'constant':
        token_reference = TokenReferenceBase(reference_token_idx=padTokenId)
        bl = token_reference.generate_reference(inputs['input_ids'].shape[1], device=inputs['input_ids'].device).unsqueeze(0)
    elif blType == 'uniform':
        bl = src.baselines.create_uniform_embedding(inputs['input_ids'][0], tokenizer)
    elif blType == 'gaussian':
        bl = src.baselines.create_gaussian_embedding(inputs['input_ids'][0], tokenizer, model)
    elif blType == 'maxDist':
        bl = src.baselines.create_max_distance_baseline(inputs['input_ids'][0], tokenizer, model)
    else:
        raise NameError(f'Baseline type {blType} not implemented')
    return bl


def summarize_attr(attr, n_steps):
    """Sum up all attribution and return the normalized sum

    Args:
        attr: Calculated attributions
        n_steps: number of steps

    Returns:
        _type_: normalized sum of attributions
    """
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
    parser = argparse.ArgumentParser(description="""
Visualization of Integrated Gradients for BERT NLP models inspired by the
Distill publication "Visualizing the Impact of Feature Attribution Baselines".
(see https://distill.pub/2020/attribution-baselines/#figure4_div)


The following BERT models are supported:

sentimentSST2: 'distilbert-base-uncased-finetuned-sst-2-english'
\t DistilBERT base uncased finetuned SST-2
\t This model is a fine-tune checkpoint of DistilBERT-base-uncased,
\t fine-tuned on SST-2. This model reaches an accuracy of 91.3 on the dev
\t set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).
\t https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you

productReviews5: 'nlptown/bert-base-multilingual-uncased-sentiment'
\t WARNING: Using the --visPath flag is not recommended
\t This a bert-base-multilingual-uncased model finetuned for sentiment
\t analysis on product reviews in six languages: English, Dutch, German,
\t French, Spanish and Italian. It predicts the sentiment of the review as a
\t number of stars (between 1 and 5).
\t This model is intended for direct use as a sentiment analysis model for
\t product reviews in any of the six languages above, or for further
\t finetuning on related sentiment analysis tasks.
\t https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you

sentimentFin3: 'ProsusAI/finbert'
\t FinBERT is a pre-trained NLP model to analyze sentiment of financial text.
\t It is built by further training the BERT language model in the finance
\t domain, using a large financial corpus and thereby fine-tuning it for
\t financial sentiment classification. Financial PhraseBank by Malo et al.
\t (2014) is used for fine-tuning. For more details, please see the paper
\t FinBERT: Financial Sentiment Analysis with Pre-trained Language Models and our related blog post on Medium.
\t The model will give softmax outputs for three labels: positive, negative or neutral.
\t https://huggingface.co/ProsusAI/finbert?text=Stocks+rallied+and+the+British+pound+gained.

emotion6: 'bhadresh-savani/distilbert-base-uncased-emotion'
\t Distilbert is created with knowledge distillation during the pre-training
\t phase which reduces the size of a BERT model by 40%, while retaining 97% of
\t  its language understanding. It's smaller, faster than Bert and any other Bert-based model.
\t Distilbert-base-uncased finetuned on the emotion dataset using HuggingFace Trainer with below Hyperparameters
\t learning rate 2e-5,
\t batch size 64,
\t num_train_epochs=8
\t https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion?text=I+like+you.+I+love+you""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--model',
                        default='sentimentSST2',
                        type=str,
                        choices=[*list(MODEL_CHOICES.keys())],
                        help='Set BERT-based text classification model (see description above for details). Possible values are: '+', '.join(list(MODEL_CHOICES.keys())),
                        metavar='MODEL')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='Sets the seed for this experiment and consequently the random sample shown.')
    parser.add_argument('-n', '--numSteps', default=10, type=int,
                        help='The number of steps used by the approximation method for integrated gradients.')
    parser.add_argument('--no_cuda', action='store_true', help='Set this if cuda is availbable, but you do NOT want to use it.')
    parser.add_argument('--visPath', action='store_true',
                        help='(CAUTION SOLW) Set this if you want to visualize the interpolated inputs on the path by decoding the interpolated embeddings to the closest-by tokens.')
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    startTime = time.time()
    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        # pass # TODO
        use_cuda = True
    print('CUDA enabled:', use_cuda)
    if args.visPath and args.model == 'productReviews5':
        logging.warning('WARNING: Due to the models vocabulary size, the computation for the tokens on the interpolation path could take a LONG time.')

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHOICES[args.model])
    model.eval()
    if use_cuda:
        model = model.cuda()

    # Dict that tanslates prediction id to class label
    id2label = model.config.id2label

    # Custom forward function
    def custom_forward(input_ids, attention_mask):
        """
        Custom forwand functions for bert models that returns logits.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    # Load tokenizer and get pad tokenls
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHOICES[args.model])
    if tokenizer._pad_token is not None:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.encode(pad_token)
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[1]
    else:
        raise ValueError("Using pad_token, but it is not set yet.")

    # Load dataset, shuffle and get random sample
    dataset = build_dataset(args.model, tokenizer)
    dataset = dataset.shuffle()
    dataset_iter = iter(dataset)
    sample = next(dataset_iter)

    xig = CustomizedLayerIntegratedGradients(custom_forward, model.get_input_embeddings())
    inputs = tokenizer(sample['input_string'], return_tensors="pt")

    model.zero_grad()
    if use_cuda:
        inputs = inputs.to('cuda')

    # Compute predicted label for input
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_pred = torch.argmax(probs, dim=1)

    # Create figures and axes
    fig, axs = plt.subplots(len(BASELINE_TYPES), 3)
    plt.tick_params(axis='x', which='major', labelsize=3)
    fig.suptitle(f'input: {sample["input_string"]}\nlabel={id2label[sample["label"]]}\nprediction={id2label[label_pred.item()]}')
    fig.tight_layout()
    axs[0, 0].set_title('Gradient at Interpolation')
    axs[0, 1].set_title('Cumulative Gradients')
    axs[0, 2].set_title('Sum of Cumulative Gradients')

    # Create dictionaries to stare data for later
    axs_bl = dict()
    intrplTokens_bl = dict()
    gradAtIntrpl_bl = dict()
    cmlGrad_bl = dict()
    sum_cmlGrad_bl = dict()
    gradAtIntrpl_bar_bl = dict()
    cmlGrad_bar_bl = dict()
    sum_cmlGrad_line_bl = dict()

    # Iterate over different baselines
    for i, blType in enumerate(BASELINE_TYPES):
        # Map axes triple to blType
        axs_bl[blType] = axs[i]

        bl = create_baseline(blType, pad_token_id, sample['input_string'], inputs, tokenizer, model)
        if use_cuda:
            bl = bl.cuda()

        attributions, intrplIds, gradAtIntrpl, cmlGrad = xig.attribute(inputs=inputs['input_ids'],
                                                                       baselines=bl,
                                                                       additional_forward_args=inputs['attention_mask'],
                                                                       target=sample['label'],
                                                                       n_steps=args.numSteps)

        seqLen = gradAtIntrpl.shape[1]
        alphas = np.linspace(0, 1, args.numSteps)

        # Compute token ids along path
        if args.visPath:
            intrplTokens = []
            for step in intrplIds:
                curStep = []
                for token_emb in step:
                    # print(token_emb.shape)
                    token_id = get_closest_id_from_emb(token_emb, model)
                    # print(tokenizer.decode(token_id))
                    curStep.append(tokenizer.decode(token_id))
                intrplTokens.append(curStep)
        else:
            # Else dublicate input string
            intrplTokens = []
            decodedInputIds = []
            for tokenId in inputs['input_ids'][0]:
                token = tokenizer.decode(tokenId)
                decodedInputIds.append(token)
            for _ in range(args.numSteps):
                intrplTokens.append(decodedInputIds)

        intrplTokens_bl[blType] = intrplTokens

        gradAtIntrpl = summarize_attr(gradAtIntrpl, args.numSteps)
        cmlGrad = summarize_attr(cmlGrad, args.numSteps)
        sum_cmlGrad = np.sum(cmlGrad, axis=1)
        sum_cmlGrad = np.cumsum(sum_cmlGrad, axis=0)

        # Save data for updating
        gradAtIntrpl_bl[blType] = gradAtIntrpl
        cmlGrad_bl[blType] = cmlGrad
        sum_cmlGrad_bl[blType] = sum_cmlGrad

        # Make plot for gradients at interpolation
        axs[i][0].set_ylim(bottom=gradAtIntrpl.min(), top=gradAtIntrpl.max())
        gradAtIntrpl_bar = axs[i][0].bar(range(seqLen), gradAtIntrpl[0])
        gradAtIntrpl_bar_bl[blType] = gradAtIntrpl_bar
        axs[i][0].set_xticks(range(seqLen))
        axs[i][0].set_xticklabels(intrplTokens[0], rotation=90)
        axs[i][0].set_ylabel(f'{blType.capitalize()} Baseline Atrb.')

        # Make plot for cumulative gradients
        axs[i][1].set_ylim(bottom=cmlGrad.min(), top=cmlGrad.max())
        cmlGrad_bar = axs[i][1].bar(range(seqLen), cmlGrad[0])
        cmlGrad_bar_bl[blType] = cmlGrad_bar
        axs[i][1].set_xticks(range(seqLen))
        axs[i][1].set_xticklabels(intrplTokens[0], rotation=90)

        # Make plot for sum of cumulative gradients
        axs[i][2].set_ylim(bottom=sum_cmlGrad.min(), top=sum_cmlGrad.max())
        axs[i][2].set_xlim(left=alphas.min(), right=alphas.max())
        sum_cmlGrad_line = axs[i][2].plot(alphas[0], sum_cmlGrad[0])
        sum_cmlGrad_line = sum_cmlGrad_line[0]
        sum_cmlGrad_line_bl[blType] = sum_cmlGrad_line

    # Add slider for alpha
    plt.subplots_adjust(bottom=0.25)
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    alphaSlider = Slider(slider_ax, 'alpha', 0.0, 1.0, 0.0, valstep=alphas)

    def update(val):
        """
        Update function that gets called when alpha/slider changes.
        """
        # Compute index (current numStep) from alpha values
        if val == 0.0:
            indx = 0
        elif val == 1.0:
            indx = args.numSteps - 1
        else:
            indx = int(args.numSteps * val)

        for blType in BASELINE_TYPES:
            # Update sum of cumulative gradients plot
            sum_cmlGrad_line_bl[blType].set_xdata(alphas[:indx+1])
            sum_cmlGrad_line_bl[blType].set_ydata(sum_cmlGrad_bl[blType][:indx+1])

            # Update x tick label to visualize tokens on path
            axs_bl[blType][0].set_xticklabels(intrplTokens_bl[blType][indx], rotation=90)
            axs_bl[blType][1].set_xticklabels(intrplTokens_bl[blType][indx], rotation=90)

            # Update bar plots (gradients at interpolation and cumulative gradients)
            for i in range(seqLen):
                gradAtIntrpl_bar_bl[blType][i].set_height(gradAtIntrpl_bl[blType][indx][i])
                cmlGrad_bar_bl[blType][i].set_height(cmlGrad_bl[blType][indx][i])

    alphaSlider.on_changed(update)
    print(f'run time: {str(timedelta(seconds=(time.time() - startTime)))}')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
