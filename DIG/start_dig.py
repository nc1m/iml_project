import sys, numpy as np, argparse, random
sys.path.append('../')
from dig import DiscretetizedIntegratedGradients
from attributions import run_dig_explanation
from metrics import eval_log_odds, eval_comprehensiveness, eval_sufficiency
import monotonic_paths
import torch

def calculate_attributions(inputs, device, factor, steps, topk, attr_func, base_token_emb, nn_forward_func, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]

	# compute attribution
	scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp

	attr, interpolated_input, gradientsAt_interpolation, cummulative_gradients= run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**factor)*(steps+1)+1)

	# compute metrics
	#log_odd, pred	= eval_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)
	#comp			= eval_comprehensiveness(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)
	#suff			= eval_sufficiency(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=args.topk)

	#return log_odd, comp, suff, interpolated_input, gradientsAt_interpolation, cummulative_gradients
	return interpolated_input, gradientsAt_interpolation, cummulative_gradients


def start_dig(input,model, seed, steps, factor, strategy, topk):

	# set seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# neural network specific imports
	if model == 'distilbert':
		from distilbert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
	elif model == 'bert':
		from bert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
	else:
		raise NotImplementedError

	auxiliary_data = load_mappings()

	# Fix the gpu to use
	device		= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# init model and tokenizer in cpu first
	_, tokenizer = nn_init(device, True)

	# Define the Attribution function
	attr_func = DiscretetizedIntegratedGradients(nn_forward_func)

	# get ref token embedding
	base_token_emb = get_base_token_emb(device)

	# compute the DIG attributions for all the inputs
	print('Starting attribution computation...')
	inputs = []
	log_odds, comps, suffs = 0, 0, 0

	#input = 'a good day to work on my muscles'
	inp = get_inputs(input, device)

	input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
	scaled_features 		= monotonic_paths.scale_inputs(input_ids.squeeze().tolist(), ref_input_ids.squeeze().tolist(),\
										device, auxiliary_data, steps=steps, factor=factor, strategy=strategy)
	inputs					= [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask]
	log_odds, comps, suffs, interpolated_input, gradientsAt_interpolation, cummulative_gradients = calculate_attributions(inputs, device, factor, steps, topk, attr_func, base_token_emb, nn_forward_func, get_tokens)
	#print(interpolated_input.shape, gradientsAt_interpolation.shape, cummulative_gradients.shape)
	# print the metrics
	print('Log-odds: ', np.round(log_odds, 4), 'Comprehensiveness: ', np.round(comps, 4), 'Sufficiency: ', np.round(suffs, 4))
	
	return interpolated_input, gradientsAt_interpolation, cummulative_gradients
	
	#return log_odds, comps, suffs, interpolated_input, gradientsAt_interpolation, cummulative_gradients


	# parser = argparse.ArgumentParser(description='IG Path')
	# parser.add_argument('-strategy', 	default='greedy', 		choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
	# parser.add_argument('-steps', 		default=10, type=int)	# m
	# parser.add_argument('-topk', 		default=20, type=int)	# k
	# parser.add_argument('-factor', 		default=0, 	type=int)	# f
	# parser.add_argument('-knn_nbrs',	default=500, type=int)	# KNN
	# parser.add_argument('-seed', 		default=42, type=int)



