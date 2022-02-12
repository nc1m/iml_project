#!/usr/bin/env python3
import functools
import warnings
from typing import Any, Callable, List, Tuple, Union, overload

import torch
from captum._utils.common import (
    _extract_device,
    _format_additional_forward_args,
    _format_outputs,
)
from captum._utils.gradient import _forward_layer_eval, _run_forward
from captum._utils.typing import BaselineType, Literal, ModuleOrModuleList, TargetType
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _tensorize_baseline,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn.parallel.scatter_gather import scatter

from customized_ig import CustomizedIntergratedGradients
from dig import DiscretetizedIntegratedGradients


class CustomizedLayerIntegratedGradients(LayerAttribution, GradientAttribution):


    def __init__(
        self,
        forward_func: Callable,
        layer: ModuleOrModuleList,
        dig: bool,
        device_ids: Union[None, List[int]] = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        GradientAttribution.__init__(self, forward_func)
        #self.ig = IntegratedGradients(forward_func, multiply_by_inputs)
        self.dig = dig
        self.ig = DiscretetizedIntegratedGradients(forward_func, multiply_by_inputs) if dig else CustomizedIntergratedGradients(forward_func, multiply_by_inputs)

        if isinstance(layer, list) and len(layer) > 1:
            warnings.warn(
                "Multiple layers provided. Please ensure that each layer is"
                "**not** solely solely dependent on the outputs of"
                "another layer. Please refer to the documentation for more"
                "detail."
            )

    @overload
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Any,
        n_steps: int,
        method: str,
        internal_batch_size: Union[None, int],
        return_convergence_delta: Literal[False],
        attribute_to_layer_input: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
        ...

    @overload
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Any,
        n_steps: int,
        method: str,
        internal_batch_size: Union[None, int],
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool,
    ) -> Tuple[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tensor,
    ]:
        ...

    @overload
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]:
        ...

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to layer inputs or outputs of the model, depending on whether
        `attribute_to_layer_input` is set to True or False, using the approach
        described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:

            inputs (tensor or tuple of tensors):  Input for which layer integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:
                            - either a tensor with matching dimensions to
                              corresponding tensor in the inputs' tuple
                              or the first dimension is one and the remaining
                              dimensions match with the corresponding
                              input tensor.
                            - or a scalar, corresponding to a tensor in the
                              inputs' tuple. This scalar value is broadcasted
                              for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (string, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size at most internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially. internal_batch_size must be at least equal to
                        #examples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            Returns:
                **attributions** or 2-element tuple of **attributions**, **delta**:
                - **attributions** (*tensor*, tuple of *tensors* or tuple of *tensors*):
                        Integrated gradients with respect to `layer`'s inputs or
                        outputs. Attributions will always be the same size and
                        dimensionality as the input or output of the given layer,
                        depending on whether we attribute to the inputs or outputs
                        of the layer which is decided by the input flag
                        `attribute_to_layer_input`.

                        For a single layer, attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.

                        For multiple layers, attributions will always be
                        returned as a list. Each element in this list will be
                        equivalent to that of a single layer output, i.e. in the
                        case that one layer, in the given layers, inputs / outputs
                        multiple tensors: the corresponding output element will be
                        a tuple of tensors. The ordering of the outputs will be
                        the same order as the layers given in the constructor.
                - **delta** (*tensor*, returned if return_convergence_delta=True):
                        The difference between the total approximated and true
                        integrated gradients. This is computed using the property
                        that the total sum of forward_func(inputs) -
                        forward_func(baselines) must equal the total sum of the
                        integrated gradient.
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in inputs.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
                >>> # and the output of this layer has dimensions Nx12x32x32.
                >>> net = ImageClassifier()
                >>> lig = LayerIntegratedGradients(net, net.conv1)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes layer integrated gradients for class 3.
                >>> # attribution size matches layer output, Nx12x32x32
                >>> attribution = lig.attribute(input, target=3)
        """
        inps, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inps, baselines, n_steps, method)

        baselines = _tensorize_baseline(inps, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        def flatten_tuple(tup):
            return tuple(
                sum((list(x) if isinstance(x, (tuple, list)) else [x] for x in tup), [])
            )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)

        inputs_layer = _forward_layer_eval(
            self.forward_func,
            inps,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        # if we have one output
        if not isinstance(self.layer, list):
            inputs_layer = (inputs_layer,)

        num_outputs = [1 if isinstance(x, Tensor) else len(x) for x in inputs_layer]
        num_outputs_cumsum = torch.cumsum(
            torch.IntTensor([0] + num_outputs), dim=0  # type: ignore
        )
        inputs_layer = flatten_tuple(inputs_layer)

        baselines_layer = _forward_layer_eval(
            self.forward_func,
            baselines,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        baselines_layer = flatten_tuple(baselines_layer)

        # inputs -> these inputs are scaled
        def gradient_func(
            forward_fn: Callable,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target_ind: TargetType = None,
            additional_forward_args: Any = None,
        ) -> Tuple[Tensor, ...]:
            if self.device_ids is None or len(self.device_ids) == 0:
                scattered_inputs = (inputs,)
            else:
                # scatter method does not have a precise enough return type in its
                # stub, so suppress the type warning.
                scattered_inputs = scatter(  # type:ignore
                    inputs, target_gpus=self.device_ids
                )

            scattered_inputs_dict = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }

            with torch.autograd.set_grad_enabled(True):

                def layer_forward_hook(
                    module, hook_inputs, hook_outputs=None, layer_idx=0
                ):
                    device = _extract_device(module, hook_inputs, hook_outputs)
                    is_layer_tuple = (
                        isinstance(hook_outputs, tuple)
                        # hook_outputs is None if attribute_to_layer_input == True
                        if hook_outputs is not None
                        else isinstance(hook_inputs, tuple)
                    )

                    if is_layer_tuple:
                        return scattered_inputs_dict[device][
                            num_outputs_cumsum[layer_idx] : num_outputs_cumsum[
                                layer_idx + 1
                            ]
                        ]

                    return scattered_inputs_dict[device][num_outputs_cumsum[layer_idx]]

                hooks = []
                try:

                    layers = self.layer
                    if not isinstance(layers, list):
                        layers = [self.layer]

                    for layer_idx, layer in enumerate(layers):
                        hook = None
                        # TODO:
                        # Allow multiple attribute_to_layer_input flags for
                        # each layer, i.e. attribute_to_layer_input[layer_idx]
                        if attribute_to_layer_input:
                            hook = layer.register_forward_pre_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )
                        else:
                            hook = layer.register_forward_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )

                        hooks.append(hook)

                    output = _run_forward(
                        self.forward_func, tuple(), target_ind, additional_forward_args
                    )
                finally:
                    for hook in hooks:
                        if hook is not None:
                            hook.remove()

                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(torch.unbind(output), inputs)
            return grads

        self.ig.gradient_func = gradient_func
        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )

        attributions, interpolated_input, gradientsAt_interpolation, cummulative_gradients = None, None, None, None
        if self.dig:
            attributions, interpolated_input, gradientsAt_interpolation, cummulative_gradients= self.ig.attribute.__wrapped__(  # type: ignore
                self.ig,  # self
                inputs_layer,
                target=target,
                additional_forward_args=all_inputs,
                n_steps=n_steps,
                return_convergence_delta=False,
            )
            
        else:
            attributions, interpolated_input, gradientsAt_interpolation, cummulative_gradients= self.ig.attribute.__wrapped__(  # type: ignore
                self.ig,  # self
                inputs_layer,
                baselines=baselines_layer,
                target=target,
                additional_forward_args=all_inputs,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
            )


        # handle multiple outputs
        output: List[Tuple[Tensor, ...]] = [
            tuple(
                attributions[
                    int(num_outputs_cumsum[i]) : int(num_outputs_cumsum[i + 1])
                ]
            )
            for i in range(len(num_outputs))
        ]

        if return_convergence_delta:
            start_point, end_point = baselines, inps
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_outputs(isinstance(self.layer, list), output), delta, interpolated_input, gradientsAt_interpolation, cummulative_gradients
        return _format_outputs(isinstance(self.layer, list), output), interpolated_input, gradientsAt_interpolation, cummulative_gradients

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self):
        return self.ig.multiplies_by_inputs