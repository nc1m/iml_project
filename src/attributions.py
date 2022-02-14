import torch
from dig import DiscretetizedIntegratedGradients


def summarize_attributions(attributions):
	"""Sum up all attribution and return the normalized sum

	Args:
		attributions: Calculated attributions

	Returns:
		_type_: normalized sum of attributions
	"""
	attributions = attributions.sum(dim=-1).squeeze(0)
	attributions = attributions / torch.norm(attributions)
	return attributions


def run_dig_explanation(dig_func, all_input_embed, position_embed, type_embed, attention_mask, steps):
	"""Discretized integrated gradients for explanation

	Args:
		dig_func (_type_): Proposed DIG
		all_input_embed (_type_): Input embeddings
		position_embed (_type_): Position embedings
		type_embed (_type_): Type embeddings
		attention_mask (_type_): Attention mask
		steps (_type_): number of steps

	Returns:
		_type_: _description_
	"""
	attributions, interpolated_input, gradientsAt_interpolation, cummulative_gradients = dig_func.attribute(scaled_features=all_input_embed, additional_forward_args=(attention_mask, position_embed, type_embed), n_steps=steps)
	attributions_word = summarize_attributions(attributions)

	return attributions_word, interpolated_input, gradientsAt_interpolation, cummulative_gradients
