import unittest
import sys
import torch
#sys.path.append('src')
import baselines
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tokenizers import decoders

class TestUtils(unittest.TestCase):

    def test_input_ids_to_sentence(self):
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        a = baselines.input_ids_to_sentence([101, 102], tokenizer)
        self.assertIsNotNone(a)

    def test_create_uniform_embedding(self):
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        a = baselines.create_uniform_embedding([101, 102], tokenizer)
        self.assertIsNotNone(a)

    def test_create_gaussian_embedding(self):
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        a = baselines.create_gaussian_embedding([101, 102], tokenizer, model)
        self.assertIsNotNone(a)

    def create_max_distance_baseline(self):
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        a = baselines.create_max_distance_baseline([101, 102], tokenizer, model)
        self.assertIsNotNone(a)

if __name__ == '__main__':
    unittest.main()