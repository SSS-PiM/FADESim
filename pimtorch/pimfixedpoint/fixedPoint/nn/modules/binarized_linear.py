import torch
import math
from torch import Tensor
from torch.nn import Module, Parameter, init

from ..splitArrayArithmetic import splitArr_matmul, splitArr_matmul_nn
# from .. import fixedPointArithmetic as fpA
from ..fixedPointArithmetic import quantization_tensor, binarize_tensor, torch_float
from ..commonConst import ConstForSplitArray as cfs, ir_drop_mode, phyArrParams, phyArrMode, TensorType
import torch.nn.functional as F


#为了简单器件，目前版本的linear_sp，在训练过程中是使用的float或double类型进行训练。
#只有在推理过程中才分解映射到物理阵列进行计算
class BinarizedLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, \
            input_bit_width: int = cfs.input_bit_width, output_bit_width: int = cfs.output_bit_width, \
            weight_bit_width: int = cfs.weight_bit_width, dac_bit_width = cfs.dac_bit_width, adc_bit_width = cfs.adc_bit_width, \
            phy_arr_mode: phyArrMode = phyArrParams.defaultArrMode, fp_type: torch.dtype = torch.float32, use_fp_bias: bool = True, bnn_first_layer_flag: bool = False):
        
        super(BinarizedLinear, self).__init__()
        self.inSize = in_features
        self.outSize = out_features
        self.hasBias = bias
        self.useFpBias = use_fp_bias

        self.bnn_first_layer_flag = bnn_first_layer_flag

        if self.bnn_first_layer_flag:
            self.inputBits = phyArrParams.bnn_first_layer_bit_width
        else:
            self.inputBits = input_bit_width
        self.outputBits = output_bit_width
        self.weightBits = weight_bit_width
        # now we only support 1-bit dac
        self.dacBits = dac_bit_width
        self.adcBits = adc_bit_width
        self.eval_flag = 0

        self.phyArrMode = phy_arr_mode

        self.fp_weight = Parameter(torch.empty((out_features, in_features), dtype = fp_type))
        self.fp_type = fp_type
        if bias:
            self.fp_bias = Parameter(torch.empty(out_features, dtype = fp_type))
        else:
            self.register_parameter('fp_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.fp_weight, a=math.sqrt(5))
        if self.fp_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fp_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.fp_bias, -bound, bound)


    def forward(self, input: Tensor) -> Tensor:
        if self.training or phyArrParams.use_bnn_origin_eval:  # training is normal software training
            # return F.linear(input, self.fp_weight, self.fp_bias)
            if not self.bnn_first_layer_flag:
                input.data = binarize_tensor(input.data, TensorType.Normal)[0].to(torch_float)

            if not hasattr(self.fp_weight,'org'):
                self.fp_weight.org = self.fp_weight.data.clone()
            self.fp_weight.data = binarize_tensor(self.fp_weight.org, TensorType.Normal)[0].to(torch_float)

            output = F.linear(input, self.fp_weight)

            if not self.fp_bias is None:
                self.fp_bias.org = self.fp_bias.data.clone()
                output += self.fp_bias.view(1, -1).expand_as(output)

            return output
        else: # we only support split array, adc, ir_drop etc.. in eval mode.
            # raise Exception("Inference process for BNN not implemented")

            #if is no ir drop, we only print the first time when stepping into the eval mode
            #if is ir drop, print every time 
    
            if self.eval_flag==0 or phyArrParams.IRdropMode!=ir_drop_mode.no_ir_drop:
                self.eval_flag += 1 
                print("[Note] The {} time step into the eval mode. Linear_sp size of {}x{}.".format(self.eval_flag, self.inSize, self.outSize), flush=True)
            if self.hasBias and not self.useFpBias: # bias is also mapped to phy array
                input = torch.cat((input, input.new_ones(input.size(0), 1)), 1)
                logicArr = torch.cat((self.fp_weight.t(), self.fp_bias.unsqueeze(0)), 0)
            else:
                logicArr = self.fp_weight.t()

            #do output = input * logic Arr
            output = splitArr_matmul_nn(input, logicArr, self.inputBits, self.dacBits, self.adcBits, self.outputBits, self.weightBits, self.phyArrMode, self.fp_type, self.bnn_first_layer_flag)
            #end out is size of [batch_size, outSize]

            if self.hasBias and self.useFpBias: # bias not mapped to phy array, use additional computing unit to calculate
                #[batch_size, outSize] + [outSize]
                #fp_bias will add to every row of output
                # output = output + self.fp_bias
                self.fp_bias.org=self.fp_bias.data.clone()
                output += self.fp_bias.view(1, -1).expand_as(output)

            return output    
            
