import torch
import math
from torch import Tensor
from torch.nn import Module, Parameter, init

from ..splitArrayArithmetic import splitArr_matmul, splitArr_matmul_nn
from ..commonConst import ConstForSplitArray as cfs, ir_drop_mode, phyArrParams, phyArrMode, TensorType
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from typing import Union
from ..fixedPointArithmetic import quantization_tensor, binarize_tensor, torch_float

# 目前版本的Conv2d_sp，在训练过程中是使用的float或double类型进行训练。
# 只有在推理过程中才分解映射到物理阵列进行计算
# 注：此处的fp指的是float point
# 目前暂不支持transposed、output_padding、device、dtype参数
'''
-----------------
| 1 | 2 | 3 | 0 |
-----------------   -------------   -----------       -----------
| 0 | 1 | 2 | 3 |   | 2 | 0 | 1 |   | 15 | 16 |       | 18 | 19 |
----------------- * ------------- = ----------- + 3 = -----------
| 3 | 0 | 1 | 2 |   | 0 | 1 | 2 |   |  6 | 15 |       |  9 | 18 |
-----------------   -------------   -----------       -----------
| 2 | 3 | 0 | 1 |   | 1 | 0 | 2 |
-----------------   -------------
'''
class BinarizedConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', input_bit_width: int = cfs.input_bit_width, output_bit_width: int = cfs.output_bit_width,
                 weight_bit_width: int = cfs.weight_bit_width, dac_bit_width = cfs.dac_bit_width, adc_bit_width = cfs.adc_bit_width,
                 phy_arr_mode: phyArrMode = phyArrParams.defaultArrMode, fp_type: torch.dtype = torch.float32, use_fp_bias: bool = True, 
                 bnn_first_layer_flag: bool = False):

        super(BinarizedConv2d, self).__init__()

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding) # 目前padding暂不支持"same", "valid"，仅支持padding大小
        self.dilation = _pair(dilation)
        self.groups = groups
        self.hasBias = bias
        self.padding_mode = padding_mode # 目前padding_mode只支持zeros
        self.useFpBias = use_fp_bias
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.bnn_first_layer_flag = bnn_first_layer_flag

        if self.bnn_first_layer_flag:
            self.inputBits = phyArrParams.bnn_first_layer_bit_width
        else:
            self.inputBits = input_bit_width
        # self.inputBits = input_bit_width
        self.outputBits = output_bit_width
        self.weightBits = weight_bit_width
        # now we only support 1-bit dac
        self.dacBits = dac_bit_width
        self.adcBits = adc_bit_width
        self.eval_flag = 0

        self.phyArrMode = phy_arr_mode

        self.fp_weight = Parameter(torch.empty((out_channels, in_channels // groups, *kernel_size), dtype=fp_type))
        self.fp_type = fp_type
        if bias:
            self.fp_bias = Parameter(torch.empty(out_channels, dtype=fp_type))
        else:
            self.register_parameter('fp_bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.fp_weight, a=math.sqrt(5))
        if self.hasBias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fp_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.fp_bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # todo: implement bnn conv layer
        if self.training or phyArrParams.use_bnn_origin_eval:
            if not self.bnn_first_layer_flag:
                input.data = binarize_tensor(input.data, TensorType.Normal)[0].to(torch_float)

            if not hasattr(self.fp_weight,'org'):
                self.fp_weight.org = self.fp_weight.data.clone()
            self.fp_weight.data = binarize_tensor(self.fp_weight.org, TensorType.Normal)[0].to(torch_float)

            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.fp_weight, self.fp_bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, self.fp_weight, self.fp_bias, self.stride,
                            self.padding, self.dilation, self.groups)

            # output = nn.functional.conv2d(input, self.weight, None, self.stride,
            #                         self.padding, self.dilation, self.groups)

            # if not self.fp_bias is None:
            #     self.fp_bias.org=self.fp_bias.data.clone()
            #     output += self.fp_bias.view(1, -1, 1, 1).expand_as(output)

            # return output
        else:
            # we only support split array, adc, ir_drop etc.. in eval mode.
            # currently, we only consider padding with zeros
            # if is no ir drop, we only print the first time when stepping into the eval mode
            # if is ir drop, print every time

            batch_size = input.size()[0]
            input_h = input.size()[2]
            input_w = input.size()[3]

            # unfold input tensor & weight tensor
            input = torch.nn.functional.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            input = input.transpose(1, 2).reshape(-1, self.in_channels * self.kernel_size[0] * self.kernel_size[1])
            fp_weight_interm = self.fp_weight.reshape(self.out_channels, -1)

            if self.eval_flag == 0 or phyArrParams.IRdropMode != ir_drop_mode.no_ir_drop:
                self.eval_flag += 1
                print("[Note] The {} time step into the eval mode, current layer: {}.".format(self.eval_flag, self), flush=True)
            if self.hasBias and not self.useFpBias: # bias is also mapped to phy array
                input = torch.cat((input, input.new_ones(input.size(0), 1)), 1)
                logicArr = torch.cat((fp_weight_interm.t(), self.fp_bias.unsqueeze(0)), 0)
            else:
                logicArr = fp_weight_interm.t()

            #do output = input * logic Arr
            output = splitArr_matmul_nn(input, logicArr, self.inputBits, self.dacBits, self.adcBits, self.outputBits, self.weightBits, self.phyArrMode, self.fp_type, self.bnn_first_layer_flag)
            #end out is size of [batch_size, outSize]

            if self.hasBias and self.useFpBias: # bias not mapped to phy array, use additional computing unit to calculate
                #[batch_size, outSize] + [outSize]
                #fp_bias will add to every row of output
                output = output + self.fp_bias

            # recover output tensor shape
            output = output.reshape(batch_size, -1, self.out_channels).transpose(1, 2)
            output_size = (math.floor((input_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                                      / self.stride[0] + 1),
                           math.floor((input_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                                      / self.stride[1] + 1))
            output = torch.nn.functional.fold(output, output_size, (1, 1))

            return output

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.hasBias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)