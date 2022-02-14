import unittest
import sys
import torch
sys.path.append('src')
import bert_datasets 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TestUtils(unittest.TestCase):

    def test_prepare_sentimentSST2(self):
        a = bert_datasets.prepare_sentimentSST2('sentimentSST2')
        self.assertIsNotNone(a)
        
    def test_prepare_productReviews5(self):
        a = bert_datasets.prepare_productReviews5('productReviews5')
        self.assertIsNotNone(a)

    def test_prepare_sentimentFin3(self):
        a = bert_datasets.prepare_sentimentFin3('sentimentFin3')
        self.assertIsNotNone(a)
    
    def test_prepare_emotion6(self):
        a = bert_datasets.prepare_emotion6('emotion6')
        self.assertIsNotNone(a)
    # def test_build_dataset(self):
    
    # def test_print_dataset_infos(self):

if __name__ == '__main__':
    unittest.main()