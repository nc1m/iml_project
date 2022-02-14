from typing import List
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchvision.transforms.functional import gaussian_blur
from tokenizers import decoders

# Maximum distance baseline = Choose word embedding which is most far away (L1-wise) from the input
# Blurred baseline = Blurres out feature informations
# Uniform baseline = Sample a random uniform embedding
# Gaussian baseline = Sample from a gaussian around original embedding
# with some std. sigma (hyperparameter)


BASELINE_TYPES = ['constant', 'maxDist', 'blurred', 'uniform', 'gaussian']



class Baseline:
    def __init__(self,model_name):

        self.model_ = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer_ = AutoTokenizer.from_pretrained(model_name)

        self.full_embs = self.model_.get_input_embeddings().weight.detach().clone()
        self.voc = self.tokenizer_.get_vocab()

        with torch.no_grad():
            self.mean = torch.mean(self.full_embs)
            self.std = torch.std(self.full_embs)
            self.lowest_emb_val =  self.full_embs.min(0).values
            self.highest_emb_val = self.full_embs.max(0).values
            self.between = (self.highest_emb_val + self.lowest_emb_val ) / 2

    def baseline_by_type(self,baseline_type : str,input : str):
        """Return a list baseline tokens

        Args:
            baseline_type (str): Chosen baseline type
            input (str): Input string for baseline

        Returns:
            list: List of baseline tokens
        """
        if baseline_type == BASELINE_TYPES[1]:
            return self.maximum_distance_embedding(input)
        elif baseline_type == BASELINE_TYPES[2]:
            return self.blur_embedding(input)
        elif baseline_type == BASELINE_TYPES[3]:
            return self.uniform_embedding(input)
        elif baseline_type == BASELINE_TYPES[4]:
            return self.gaussian_embedding(input)
    
    def baseline_by_type(self,baseline_type : str,input : List):
        """Return a list baseline tokens

        Args:
            baseline_type (str): Chosen baseline type
            input (List): Input string as Token ids
        Returns:
            list: List of baseline tokens
        """
        input = ' '.join(self.tokenizer_.decode(input))

        if baseline_type == BASELINE_TYPES[1]:
            return self.maximum_distance_embedding(input)
        elif baseline_type == BASELINE_TYPES[2]:
            return self.blur_embedding(input)
        elif baseline_type == BASELINE_TYPES[3]:
            return self.uniform_embedding(input)
        elif baseline_type == BASELINE_TYPES[4]:
            return self.gaussian_embedding(input)

    
    def maximum_distance_embedding(self, sentence : str):
        """[summary]

        Args:
            sentence (torch.Tensor): [description]
        """
        sentence = self.tokenizer_.tokenize(sentence)

        baseline_sentence = []
        for word in sentence:

            if word in self.tokenizer_.all_special_tokens:
                baseline_sentence.append(torch.tensor(self.tokenizer_.encode(word)[1]))
                continue

            word_emb = self.model_.get_input_embeddings()(torch.tensor(self.tokenizer_.encode(word)[1]))

            # Set all values to the maximum if value is closer to the minimum (since we want max distance)
            masked = word_emb > self.between
            masked_emb = torch.zeros_like(word_emb)
            masked_emb[masked==True] = self.lowest_emb_val[masked==True]
            masked_emb[masked==False] = self.highest_emb_val[masked==False]

            baseline_sentence.append(self.get_nearest_word(masked_emb))
        
        return baseline_sentence
    
    def uniform_embedding(self, sentence : str):
        """ Randomly (uniform) select a token for each word in the sentence

        Args:
            sentence (str): [description]
        """
        sentence = self.tokenizer_.tokenize(sentence)

        baseline_sentence = []
        for word in sentence:

            if word in self.tokenizer_.all_special_tokens:
                baseline_sentence.append(torch.tensor(self.tokenizer_.encode(word)[1]))
                continue
            
            baseline_sentence.append(np.random.randint(1,len(self.voc)))
        return baseline_sentence

    def blur_embedding(self, sentence : str):
        """_summary_

        Args:
            sentence (str): _description_
        """
        sentence = self.tokenizer_.tokenize(sentence)

        baseline_sentence = []
        for word in sentence:

            if word in self.tokenizer_.all_special_tokens:
                baseline_sentence.append(torch.tensor(self.tokenizer_.encode(word)[1]))
                continue

            word_emb = self.model_.get_input_embeddings()(torch.tensor(self.tokenizer_.encode(word)[1]))

            baseline_sentence.append(self.get_nearest_word(word_emb))
        
        return baseline_sentence
    
    def gaussian_embedding(self,sentence : str) -> int:
        """_summary_

        Args:
            input (str): _description_

        Returns:
            int: _description_
        """
        sentence = self.tokenizer_.tokenize(sentence)

        baseline_sentence = []
        for word in sentence:
            
            if word in self.tokenizer_.all_special_tokens:
                baseline_sentence.append(torch.tensor(self.tokenizer_.encode(word)[1]))
                continue

            sampled_emb = torch.Tensor(np.random.normal(self.mean,self.std,size=self.full_embs.shape[1]))
            baseline_sentence.append(self.get_nearest_word(sampled_emb))
        return baseline_sentence



    def get_nearest_word(self,input : torch.Tensor) -> int:
        """_summary_

        Args:
            input (str): _description_

        Returns:
            _type_: _description_
        """
        
        minimal_dist = np.inf
        closest_word_token =  None
        for word in self.voc.keys():
            token = self.voc[word]
            curEmb = self.model_.get_input_embeddings()(torch.tensor(token))
            dist = torch.sqrt(torch.sum((curEmb - input) ** 2))

            if dist < minimal_dist:
                minimal_dist = dist
                closest_word_token = token
        return closest_word_token


# base = Baseline("distilbert-base-uncased-finetuned-sst-2-english")


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


# print(' '.join(tokenizer.decode(base.maximum_distance_embedding("I really hate you"))))

# print(tokenizer.decode(base.gaussian_embedding("I love you!")))
# print(tokenizer.decode(base.uniform_embedding("I love you!")))


# full_emb = model.get_input_embeddings().weight.detach().clone()
# mean = torch.mean(full_emb)
# va = torch.std(full_emb)
# samp = torch.Tensor(np.random.normal(mean,va,size=full_emb.shape[1]))

voc = tokenizer.get_vocab()

enc = tokenizer.encode("House")[1]
emb = model.get_input_embeddings()(torch.tensor(enc)).detach().clone()



emb = gaussian_blur(emb.ToTensor(),kernel_size=[1,3])
print(emb)


# print(emb)
# Encode gibt id von cls, <word> , sep zurück 
# print(tokenizer.encode("House"))
# print(tokenizer.decode(102))

# Gibt mir ein Embedding für ein Wort
# print(model.get_input_embeddings()(torch.tensor(tokenizer.encode("House")[1])))
# full_emb = model.get_input_embeddings().weight.detach().clone()
# with torch.no_grad():
#     mean = torch.mean(full_emb)

#     lowest/highest value per dimension
# lowest_emb_val = full_emb.min(0).values
# highest_emb_val = full_emb.max(0).values


# between = (highest_emb_val - lowest_emb_val ) / 2
#     print(between)

# house_emb = model.get_input_embeddings()(torch.tensor(tokenizer.encode("is")[1]))

# masked = house_emb > between
#     print(masked)


#     Set all values to highest or lowest value where value is higher/lower the middle value
# masked_emb = torch.zeros_like(house_emb)
# masked_emb[1] = True
# print(masked_emb)
# masked_emb[masked==False] = lowest_emb_val[masked==False]
# print(masked_emb)
# masked_emb[masked==True] = highest_emb_val[masked==True]
# print(highest_emb_val[1])
# print(masked_emb)

#     minimal_dist = np.inf
#     closest_word_token =  None
#     for word in voc.keys():
#         token = voc[word]
#         curEmb = model.get_input_embeddings()(torch.tensor(token))
#         dist = torch.sqrt(torch.sum((curEmb - masked_emb) ** 2))

#         if dist < minimal_dist:
#             minimal_dist = dist
#             closest_word_token = token
#     print(closest_word_token)
#     print(tokenizer.decode(closest_word_token))

