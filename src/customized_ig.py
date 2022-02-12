from captum.attr import IntegratedGradients
from typing import Callable, Tuple, Union, List, Any
from torch import Tensor
import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
)
from captum._utils.typing import TargetType
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import _reshape_and_sum


class CustomizedIntergratedGradients(IntegratedGradients):
    def __init__(self, forward_func: Callable, multiply_by_inputs: bool = True) -> None:
        IntegratedGradients.__init__(self, forward_func, multiply_by_inputs)

    def _attribute(self, inputs: Tuple[Tensor, ...], baselines: Tuple[Union[Tensor, int, float], ...], target: TargetType = None, additional_forward_args: Any = None, n_steps: int = 50, method: str = "gausslegendre", step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None, return_convergence_delta:bool = False) -> Tuple[Tensor, ...]:
        
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)

        # for inputcounter, input, baseline in enumerate(zip(inputs, baselines)):
        #     for alphacounter, alpha in enumerate(alphas):
        #         alpha_gradient_list[inputcounter][alphacounter]=torch.cat(baseline + alpha * (input - baseline),dim=0)

        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )
       
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        gradientsAt_interpolation = (inputs[0] - baselines[0]) * grads[0]

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]
        
        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        cummulative_gradients = tuple(
            _reshape_and_sum(
                scaled_grad, 1, grad.shape[0] // 1, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )
        
        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )

        #return attributions
        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return attributions, delta, scaled_features_tpl[0], gradientsAt_interpolation, cummulative_gradients[0]
        return attributions, scaled_features_tpl[0], gradientsAt_interpolation, cummulative_gradients[0]

