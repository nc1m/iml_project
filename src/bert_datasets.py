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


def prepare_sentimentSST2(modelName, tokenizer):
    # standard columns: input_string, input_ids, attention_mask, label
    dataset = load_dataset(**DATASET_INFO[modelName])

    dataset = dataset.rename_column('sentence', 'inputString')
    print(dataset['label'])
    # TODO: change column label to -1=>0
    # def adjust_labels(e):
    #     e['label'] = [sentiment + 1 for sentiment in e['label']] #e['label'] + 1
    #     return e['label']
    # change -1 labels to 0 for conistency
    # dataset = dataset.map(lambda e: e['label'] + 1)
    # print(dataset['label'])
    # moved tokenization of inputString to integrated_gradients.py
    # https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__
    #dataset = dataset.map(lambda e: tokenizer(e['inputString']))#, truncation=True, padding='max_length'), batched=True)
    return dataset




def build_dataset(modelName, tokenizer):
    """Gets model argument and tokenzier and calls the appropiate method to
    create a dataset with standardised columns
    """
    if modelName == 'sentimentSST2':
        dataset = prepare_sentimentSST2(modelName, tokenizer)
    elif modelName == 'TODO':
        pass
    else:
        raise NameError(f'Dataset for model "{modelName}" not implemented')

    # If used and iterate over dataset => only returns 'input_ids', 'attention_mask', 'label'
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset



def print_dataset_infos(datasetInfo):
    for key in datasetInfo:
        print(key)
        dataset_builder = load_dataset(**datasetInfo[key])
        print(dataset_builder.info.features)
        # print(dataset_builder.split)
        # # print(dataset_builder.cache_dir)
        # # print(dataset_builder.info.features)
        # print(dataset_builder.info.splits)
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
