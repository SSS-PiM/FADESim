import torch
from torch import Tensor
import math
from .commonConst import torch_int, torch_float, system_bit_width, data_flow_bit_width, TensorType, \
    RightShiftMode, WeightUpdateStrategy

test_iter_index = 0
test_layer_index = 0
test_num_layer = 16


def to_float(tensor):
    return tensor.view(dtype=torch_float)


def to_int(tensor):
    return tensor.detach().view(dtype=torch_int)

def _get_fixed_point_position(max_abs: float, bit_width: int, tensor_type=TensorType.Normal) -> int:
    if math.isclose(max_abs, 0.0):
        max_abs = 1e-12
    if tensor_type == TensorType.Normal:
        return math.ceil(math.log2(max_abs / ((1 << (bit_width - 1)) - 1)))
    elif tensor_type == TensorType.PN:
        return math.ceil(math.log2(max_abs / ((1 << bit_width) - 1)))
    else:
        raise Exception("Unknown tensor type : " + str(tensor_type.value))


def _parse_quantization_para(quantization_para):
    """"
    input: quantization parameters, should be int Tensor
    return: quantization para. {s, bit_width, tensor_type(Normal or ref or PN)}
    """
    quantization_para = to_int(quantization_para)
    s = quantization_para[0].item()
    tensor_type = TensorType(quantization_para[1].item())

    return s, tensor_type


def _parse_tensor_tuple_to_int(fp_tensor_tuple: tuple):
    int_tensor = to_int(fp_tensor_tuple[0])
    quantization_para = to_int(fp_tensor_tuple[1])

    return int_tensor, quantization_para


def _creat_quantization_para(s: int = None, tensor_type: TensorType = None, device=None):
    """
    [0]: s [1]: tensor_type
    :param s:
    :param tensor_type:
    :param device:
    :return: quantization_para
    """
    quantization_para = torch.empty(2, dtype=torch_int, device=device)
    if s is not None:
        quantization_para[0] = s
    # if bit_width is not None:
    #     quantization_para[1] = bit_width
    if tensor_type is not None:
        quantization_para[1] = tensor_type.value

    return quantization_para

def binarize_tensor(tensor, tensor_type: TensorType, quant_mode='det'):
    if quant_mode=='det':
        int_tensor = tensor.sign().to(torch_int)
    else:
        int_tensor = tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).to(torch_int)
    quantization_para = \
        _creat_quantization_para(s=0, tensor_type=tensor_type, device=tensor.device)
    return int_tensor, quantization_para

def quantization_tensor(tensor: Tensor, bit_width: int, tensor_type: TensorType):
    """
    :param tensor_type:
    :param bit_width:
    :param tensor: Float type tensor, that is the real number for you to quantize
    :return: int tensor, int para.
    """
    max_abs_value = tensor.abs().max().item()

    s = _get_fixed_point_position(max_abs_value, bit_width, tensor_type)
    quantization_para = \
        _creat_quantization_para(s=s, tensor_type=tensor_type, device=tensor.device)

    resolution = pow(2, s)

    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        int_tensor = tensor.div(resolution).round().to(torch_int)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))

    return int_tensor, quantization_para


def de_quantization(fp_tensor_tuple: tuple) -> Tensor:
    int_tensor, quantization_para = _parse_tensor_tuple_to_int(fp_tensor_tuple)
    s, tensor_type = _parse_quantization_para(quantization_para)

    resolution = pow(2, s)
    if tensor_type == TensorType.Normal or tensor_type == TensorType.PN:
        return int_tensor.to(torch_float).mul(resolution)
    else:
        raise Exception("Unknown tensor type: " + str(tensor_type.value))


