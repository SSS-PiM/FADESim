import math
import os
import torch
import numpy as np
from enum import Enum
from torch.utils.cpp_extension import load
from .conductance_variation import get_nonlinear_params_from_K2, nonlinear_IV 
import torch.nn as nn
import torch.optim as optim
import logging

class debug:
    detailed = False
    simple = False

class TensorType(Enum):
    Normal = 0  # two's complement representation: n+1 bits range: -2^n ~ 2^n-1
    # Ref = 1  # abandoned
    PN = 2  # pos neg representation: n bits range: -(2^n-1) ~ 2^n-1
    # SM = 3  # we will support it int the future, sign magnitude representation: n+1 bits range: -(2^n-1) ~ 2^n-1

class RightShiftMode(Enum):
    Abandon = 0
    Round = 1
    RoundToEvenNearest = 2


class WeightUpdateStrategy(Enum):
    StaticRange = 0
    DynamicRange = 1


system_bit_width = 32
data_flow_bit_width = 4
# data flow bit width must be half of system bit width to avoid overflow
if system_bit_width <= 32:
    torch_int = torch.int32
    torch_float = torch.float32
else:
    torch_int = torch.int64
    torch_float = torch.float64


# input/weight/output/adc/dac bits
class ConstForSplitArray():
    input_bit_width = 8
    output_bit_width = 10
    weight_bit_width = 8
    adc_bit_width = 10
    dac_bit_width = 1

class phyArrMode(Enum):
    PN = 0
    Ref = 1

class inputMode(Enum):
    Normal = 0
    PN = 1

class rectification_unit():
    hasTransistor = False
    hasSelector = False
    hasDiode = False
    TConduct = 1e-2


class Cmul(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size
        # self.w = torch.Tensor(*self.size)
        self.w = torch.ones(*self.size)
        # self.reset()
        self.w = nn.Parameter(self.w)
    
    def reset(self, stdv: float = None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1/math.sqrt(self.w.nelement())
        nn.init.uniform_(self.w, -stdv, stdv)
        # self.w.add_(1)
        # print(self.w)
        logging.debug("cmul stdv = {}, cmul data = {}".format(stdv, self.w))

    
    def forward(self, input: torch.Tensor):
        output = input * self.w
        # print(f"cmul sum = {self.w}")
        return output

# input: weight data quantized after original training of DNN (int)
# output: weight data after affected by IR drop (int)
class SCN(nn.Module):
    def __init__(self, layer_num, channel_size, cbsize):
        super().__init__()
        self.layer_num = layer_num        
        self.channel_size = channel_size        
        self.cbsize = cbsize
        self.c1 = Cmul(1, self.cbsize[0], self.cbsize[1])
        self.net = nn.Sequential()
        self.net.add_module("conv0", nn.Conv2d(1, self.channel_size, kernel_size=3, stride=1, padding=1))
        self.net.add_module("relu0", nn.ReLU())
        for i in range(1, self.layer_num-1):
            self.net.add_module("conv"+str(i), nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, stride=1, padding=1))
            self.net.add_module("relu"+str(i), nn.ReLU())

        self.net.add_module("conv"+str(self.layer_num-1), nn.Conv2d(self.channel_size, 1, kernel_size=3, stride=1, padding=1))
        
        self.c2 = Cmul(1, self.cbsize[0], self.cbsize[1])
        

    def forward(self, x):
        x = x.reshape(-1, self.cbsize[0], self.cbsize[1])
        y = self.c1(x)
        y = y.view(-1, 1, self.cbsize[0], self.cbsize[1])
        y = self.net(y)
        y = y.view(-1, self.cbsize[0], self.cbsize[1])
        y = self.c2(y)
        return y.view(-1, self.cbsize[0]*self.cbsize[1])

class variations():
    var_defaultSeed = 555
    rand_gen_memCMOS = torch.Generator()
    rand_gen_mirrored = torch.Generator()
    rand_gen_network = torch.Generator()

    rand_gen_memCMOS.manual_seed(var_defaultSeed)
    rand_gen_mirrored.manual_seed(var_defaultSeed+1)
    rand_gen_network.manual_seed(var_defaultSeed+2)

    # 为了inference过程中，每个batch测试时，固定的物理阵列的variation是一样的。
    # 添加了network_gen_reset函数来重置了rand_gen_network
    # 这个函数应该在每个batch测试完毕后，调用来重置rand_gen_network
    @staticmethod
    def network_gen_reset(seed = var_defaultSeed):
        # variations.rand_gen_network = torch.Generator()
        variations.rand_gen_network.manual_seed(seed+2)
    
    @staticmethod
    def mirrored_gen_reset(seed = var_defaultSeed):
        # variations.rand_gen_mirrored = torch.Generator()
        variations.rand_gen_mirrored.manual_seed(seed+1)
        
    @staticmethod
    def memCMOS_gen_reset(seed = var_defaultSeed):
        # variations.rand_gen_memCMOS = torch.Generator()
        variations.rand_gen_memCMOS.manual_seed(seed)
    
    @staticmethod
    def reset_all(seed = var_defaultSeed):
        variations.memCMOS_gen_reset(seed)
        variations.mirrored_gen_reset(seed)
        variations.network_gen_reset(seed)

# class process_mode(Enum):
#     normal = 0
#     OU = 1

class ir_drop_mode(Enum):
    no_ir_drop = 0
    ir2s = 1                # our proposed IR drop iterative refinement
    accurate_math = 2       # solve linear equation
    accurate_GS_math = 3    # our proposed GS method
    em = 4                  # sci china, Efficient evaluation model including interconnect resistance effect for large scale RRAM crossbar array matrix computing
    ahk = 5                 # IBM aihwkit mode
    scn = 6                 # scn predictive model
    other = 7

class input_v_mode(Enum):
    complement_code = 0
    neg_pulse = 1

class GS_method(Enum): # Gauss-Seidel method, this is only available in not cuda mode.  
    #batch by batch is much faster, due to the indexing in tensor is too slow. So batch operation will be better
    nodeBynode = 1
    batchBybatch = 2 # this batch is not batch size used in NN. it's a mode of updating ir drop voltage

class ir_drop_compensate_scheme(Enum):
    no_compensation = 0
    icon = 1 # only for OU
    mirrored = 2
    memCMOS = 3
    irdrop_deembedding = 4
    offline_training = 5

class phyArrParams():
    # 单元信息
    Ron = 1e3
    Roff = 1e6
    cellMaxConduct = 1 / (Ron)
    cellMinConduct = 1 / (Roff)
    sigma = 0                      # R = Rtarget * exp(q), q~N(0, theta^2), so C = Ctarget/exp(q), here simga = theta^2
    cellBits = 1                    # note that weightBits%cellBits must be 0
    nonlinear_cell_enable = False

    # K(p, V) = p*R(V/p)/R(V), usually p = 2
    # so K2(V) = 2*R(V/2)/R(V) 
    nonlinear_params = []
    if nonlinear_cell_enable:  #I = b*sinh(aV)
        nonlinear_Voltage = 1
        nonlinear_K2_on = 10.00001   # if you are linear, then K2 = 2, but you should set it to a number close 2 rather than exact 2. 
                                    # you will get an error of division by zero if you use exact 2 since you use non-linear cell but set a linear cell.
        nonlinear_K2_off = 20.00001
        a_on, b_on = get_nonlinear_params_from_K2(nonlinear_Voltage, 1/cellMaxConduct, nonlinear_K2_on)
        a_off, b_off = get_nonlinear_params_from_K2(nonlinear_Voltage, 1/cellMinConduct, nonlinear_K2_off)
        nonlinear_params = [1/cellMaxConduct, 1/cellMinConduct, a_on, b_on, a_off, b_off]


            
    
    if ConstForSplitArray.weight_bit_width%cellBits!=0:
        raise Exception("[Error] cellbit is not factor of weight_bit")

    # 阵列大小
    arrRowSize = 64
    arrColSize = 64
    crx_size = arrRowSize*arrColSize

    # pos/neg array, or ref colomn mode for weight
    defaultArrMode = phyArrMode.PN

    # input pulse
    inputVmode = input_v_mode.complement_code
    inputMaxVoltage = 1

 

    # 互连线电阻
    r_wire = 2.93
    r_load = 2.93

    # ADC, adc bit is configured before
    cellDeltaConduct = (cellMaxConduct-cellMinConduct)/(2**cellBits-1)
    maxSumI = arrRowSize*cellMaxConduct*inputMaxVoltage
    adcUseI = arrRowSize*(cellMaxConduct-cellMinConduct)*inputMaxVoltage

    # OU
    useOUSize = False
    OUSize = (8, 8)
    if useOUSize:
        adcOUuseI = OUSize[0]*(cellMaxConduct-cellMinConduct)*inputMaxVoltage
        arrOUnum = (arrRowSize//OUSize[0], arrColSize//OUSize[1])
        if arrRowSize%OUSize[0]!=0 or arrColSize%OUSize[1]!=0:
            raise Exception("[Error] ou size is not the factor of the crx size.")
        if OUSize[1]<2 or OUSize[0]<2:
            raise Exception("[Error] ou size is not less than 2.")

    # IR drop
    IRdropMode = ir_drop_mode.accurate_GS_math
    ir_drop_iter_time = 600  # 
    
    if IRdropMode==ir_drop_mode.accurate_GS_math:
        ir_drop_GS_method_beta_scale = 1.90 # in fact this is called the omega(w) in the paper
        ir_drop_GS_method_mode = GS_method.batchBybatch
    
    if IRdropMode==ir_drop_mode.scn:
        print("now the working dir is {}.".format(os.getcwd()))
        ir_drop_scn_model_name = "./model/SCNron1k_roff1e6_32x32_293_5w.pt"
        scn_ch_size = 32
        scn_layer_num = 7
        scn_model = SCN(scn_layer_num, scn_ch_size, [arrRowSize, arrColSize])
        model_paras_load = torch.load(ir_drop_scn_model_name)
        scn_model.load_state_dict(model_paras_load)

    # used in gs_method & ir2s method, if enable, tow iteration result less than threshold will stop
    if IRdropMode in [ir_drop_mode.accurate_GS_math, ir_drop_mode.ir2s]:
        ir_drop_less_than_thresholdBreak_enable = False
        ir_drop_less_than_threshold = 1e-7

    ##### don't change the below #####################
    # IR drop simulation (gs_method) using cuda to speed up
    cpp_ir_drop_accelerate_flag = True

    print("now the working dir is {}.".format(os.getcwd()))
    if os.getcwd()[-5:]!="point":
        os.chdir("../")
    cpp_ir_drop_accelerate = load(name = "cpp_ir_drop_accelerate", 
                                    # extra_cuda_cflags=["-arch=sm_70"],
                                    sources = ["./fixedPoint/SplitArrCuda/split_arr_matmul.cu"], verbose = True)
    
    # IR drop补偿相关
    IRdropCompOU = False
    IRdropCompensator = ir_drop_compensate_scheme.no_compensation
    if IRdropCompensator == ir_drop_compensate_scheme.icon:
        IRdropCompOU = True
    if IRdropMode == ir_drop_mode.no_ir_drop:
        IRdropCompensator = ir_drop_compensate_scheme.no_compensation
    if IRdropCompensator in [ir_drop_compensate_scheme.memCMOS]:
        IRdropMode = ir_drop_mode.other

    # BNN相关
    useBNN = False
    use_bnn_origin_eval = False
    bnn_first_layer_bit_width = 8
    if useBNN == True:
        # check params
        assert(ConstForSplitArray.dac_bit_width == 1)
        assert(ConstForSplitArray.weight_bit_width == 1)
        assert(cellBits == 1)
        assert((ConstForSplitArray.input_bit_width==1 and inputVmode==input_v_mode.neg_pulse) or
               (ConstForSplitArray.input_bit_width==2 and inputVmode==input_v_mode.complement_code))

        assert((ConstForSplitArray.input_bit_width==1 and inputVmode==input_v_mode.neg_pulse) or (ConstForSplitArray.input_bit_width==2 and inputVmode==input_v_mode.complement_code))
        # assert(inputVmode == input_v_mode.neg_pulse)
    
    if nonlinear_cell_enable:
        if not cpp_ir_drop_accelerate_flag:
            raise Exception("nonlinear cell now is only supported in cuda code")        

    # 科学计算相关
    scientific_computing_pim_enable = False
    defaultInputMode = inputMode.PN
    if scientific_computing_pim_enable == True:
        assert(ConstForSplitArray.dac_bit_width == 1)
        assert(cellBits == 1)
        assert(defaultArrMode == phyArrMode.Ref)
        assert(defaultInputMode == inputMode.PN)

    # 配置参数输出
    print_all_params = False
    if print_all_params:
        print("For simplicity, we directly print the params code:")
        with open("./fixedPoint/nn/commonConst.py", "r") as f:
            print(f.read())

class memCMOS_method():
    C1 = phyArrParams.cellMinConduct
    C2 = phyArrParams.cellMinConduct

