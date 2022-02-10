from datasets import load_dataset_builder
from datasets import load_dataset
import datasets

DATASET_INFO = dict()
# https://huggingface.co/datasets/glue
DATASET_INFO['sentimentSST2'] = {'path': 'glue', 'name': 'sst2', 'split': 'test'}

# https://huggingface.co/datasets/amazon_reviews_multi
DATASET_INFO['productReviews5'] = {'path': 'amazon_reviews_multi', 'name': 'all_languages', 'split': 'test'}

# https://huggingface.co/datasets/financial_phrasebank
# has no test split
DATASET_INFO['sentimentFinancial3'] = {'path': 'financial_phrasebank', 'name': 'sentences_allagree', 'split': 'train'}

# https://huggingface.co/datasets/emotion
DATASET_INFO['emotion6'] = {'path': 'emotion', 'split': 'test'}


def prepare_sentimentSST2(modelName):
    rawDataset = load_dataset(**DATASET_INFO[modelName])
    # dataset = dataset.rename_column("Title", "Novel")


def build_dataset(modelName, tokenizer):
    if modelName == 'sentimentSST2':
        pass


def print_dataset_infos(datasetInfo):
    for key in datasetInfo:
        print(key)
        dataset_builder = load_dataset(**datasetInfo[key])
        print(dataset_builder.info.features)
        print(dataset_builder.split)
        # print(dataset_builder.cache_dir)
        # print(dataset_builder.info.features)
        print(dataset_builder.info.splits)
        print()


if __name__ == '__main__':
    print('running as main')

    print_dataset_infos(DATASET_INFO)

    exit()
    dataset = load_dataset(path='glue', name='sst2', split='test')
    print(dataset.info)
    print(dataset.shape)
    print(dataset.column_names)
    print(dataset.features)

    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    modelName = 'distilbert-base-uncased-finetuned-sst-2-english'
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    print(model.config)
    id2label = model.config.to_dict()['id2label']

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(modelName)
