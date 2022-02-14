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
DATASET_INFO['sentimentFin3'] = {'path': 'financial_phrasebank', 'name': 'sentences_allagree', 'split': 'train'}

# https://huggingface.co/datasets/emotion
DATASET_INFO['emotion6'] = {'path': 'emotion', 'split': 'test'}


def prepare_sentimentSST2(modelName: str, tokenizer):
    """Loads the glue SST test dataset and changes the columns to the standard names.
    Strangely all the labels consist of -1 (negative) samples and to map them to the model
    predictions we mapped -1 to 0 which is the model class for negative.

    Args:
        modelName (str): model name to be load
        tokenizer: Used pretrained tokenizer

    Returns:
        _type_: Dataset
    """
    # standard columns: input_string, input_ids, attention_mask, label
    dataset = load_dataset(**DATASET_INFO[modelName])
    dataset = dataset.rename_column('sentence', 'input_string')
    dataset = dataset.map(lambda sample: {'label': sample['label'] + 1})
    # moved tokenization of inputString to integrated_gradients.py
    # https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__
    #dataset = dataset.map(lambda e: tokenizer(e['inputString']))#, truncation=True, padding='max_length'), batched=True)
    return dataset


def prepare_productReviews5(modelName):
    """Setup dataset for product review data

    Args:
        modelName: model name to be load

    Returns:
        _type_: Dataset
    """
    dataset = load_dataset(**DATASET_INFO[modelName])
    dataset = dataset.rename_column('stars', 'label')
    # Concatenate review title and review body into input_string column
    dataset = dataset.map(lambda sample: {'input_string': sample['review_title'] + ' ' + sample['review_body']})
    # Convert to model prediction space 0-4 instead of 1-5 stars
    dataset = dataset.map(lambda sample: {'label': sample['label'] - 1})
    # TODO? use dataset.filter to remove unkonwn language reviews
    return dataset


def prepare_sentimentFin3(modelName):
    """Loads the financial_phrasebank dataset, changes the colmun names to the
    standard names and changes the labels to the specifications of the model
    names.
    label_id = original label => model label
    0 = negative => positive
    1 = neutral => negative
    2 = positive => neutral
    """
    def convert_to_temp_labels(sample):
        """Converts the labels to temporary labels.
        """
        if sample['label'] == 2:
            sample['label'] = 4
        elif sample['label'] == 0:
            sample['label'] = 5
        elif sample['label'] == 1:
            sample['label'] = 6
        return sample

    def convert_to_model_labels(sample):
        """Converts the temporary labels to model labels
        """
        if sample['label'] == 4:
            sample['label'] = 0
        elif sample['label'] == 5:
            sample['label'] = 1
        elif sample['label'] == 6:
            sample['label'] = 2
        return sample

    dataset = load_dataset(**DATASET_INFO[modelName])
    dataset = dataset.rename_column('sentence', 'input_string')
    dataset = dataset.map(convert_to_temp_labels)
    dataset = dataset.map(convert_to_model_labels)
    return dataset

def prepare_emotion6(modelName):
    dataset = load_dataset(**DATASET_INFO[modelName])
    dataset = dataset.rename_column('text', 'input_string')
    return dataset


def build_dataset(modelName, tokenizer):
    """Gets model argument and tokenzier and calls the appropiate method to
    create a dataset with standardised columns
    """
    if modelName == 'sentimentSST2':
        dataset = prepare_sentimentSST2(modelName, tokenizer)
    elif modelName == 'productReviews5':
        dataset = prepare_productReviews5(modelName)
    elif modelName == 'sentimentFin3':
        dataset = prepare_sentimentFin3(modelName)
    elif modelName == 'emotion6':
        dataset = prepare_emotion6(modelName)
    else:
        raise NameError(f'Dataset for model "{modelName}" not implemented')

    # If used and iterate over dataset => only returns 'input_ids', 'attention_mask', 'label'
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset


def print_dataset_infos(datasetInfo):
    """Reveal informations for a dataset

    Args:
        datasetInfo (_type_): dataset informations
    """
    for key in datasetInfo:
        print(key)
        dataset_builder = load_dataset(**datasetInfo[key])
        print(dataset_builder.info.features)


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
