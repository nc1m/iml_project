import pickle
import argparse
import time
from datetime import timedelta
from pathlib import Path


from sklearn.neighbors import kneighbors_graph
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# modified from https://github.com/INK-USC/DIG/blob/main/knn.py

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

def parse_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs',
                        default=23,
                        type=int,
                        help='Set the number of processes for the kneighbors_graph algorithm.')
    parser.add_argument('--n_neighbors',
                        default=500,
                        type=int,
                        help='Set the number of processes for the kneighbors_graph algorithm.')
    parser.add_argument('--outputDir',
                        default='knn',
                        type=Path,
                        help='Set the number of processes for the kneighbors_graph algorithm.')
    return parser.parse_args()

def main():
    startTime = time.time()
    args = parse_args()
    outputDir = args.outputDir.resolve()
    if not outputDir.is_dir():
        print(f'Creating output directory: {outputDir}')
        outputDir.mkdir()
    print('Starting KNN computation..')

    for i, modelName in enumerate(MODEL_CHOICES.keys()):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHOICES[modelName])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHOICES[modelName])
        word_features = model.get_input_embeddings().weight.detach().clone().numpy()
        word_idx_map = tokenizer.get_vocab()
        A = kneighbors_graph(word_features, args.n_neighbors, mode='distance', n_jobs=args.n_jobs)
        knn_path = Path(outputDir, f'{modelName}_{args.n_neighbors}.pickle')
        with open(knn_path, 'wb') as f:
            pickle.dump([word_idx_map, word_features, A], f)
        print(f'Written KNN data ({i+1}/{len(MODEL_CHOICES.keys())}) at {knn_path}')

    print(f'run time: {str(timedelta(seconds=(time.time() - startTime)))}')


if __name__ == "__main__":
    main()
