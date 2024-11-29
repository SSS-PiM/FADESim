import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
import math
from scipy.sparse import csr_matrix
from .ir_drop_compensate import icon_compensator, mirrored_compensator, memCMOS_preOU_compensator, memCMOS_postOU_compensator
from .commonConst import phyArrMode, phyArrParams, TensorType, ir_drop_mode, input_v_mode, ir_drop_compensate_scheme
from .fixedPointArithmetic import quantization_tensor, binarize_tensor, to_float
# from .ir_drop_solve import no_ir_drop_cal, ir_drop_ir2s_fastsolve, ir_drop_accurate_math, ir_drop_accurate_GSMethod, \
#     ir_drop_OU
from .ir_drop_solve import ir_drop_em_fastsolve, no_ir_drop_cal, no_ir_drop_cal_ou, ir_drop_ir2s_fastsolve, ir_drop_accurate_math, ir_drop_accurate_GSMethod, ir_drop_OU, ir_drop_scn_model
from .conductance_variation import add_variation
from .commonConst import variations, debug, inputMode, ConstForSplitArray as cfs

# input [bsize, insize]
# w [insize, outsize]
#outputBits now is not used
def splitArr_matmul(input: Tensor, w: Tensor, inputBits: int, dacBits: int, adcBits: int, \
        outputBits: int, weightBits: int, arrType: phyArrMode, float_type: torch.dtype, bnn_first_layer_flag: bool = False) -> Tensor:
    with torch.no_grad():
        bsize = input.size(0)
        insize = w.size(0)
        outsize = w.size(1)
        minCellC = phyArrParams.cellMinConduct
        inputV = phyArrParams.inputMaxVoltage/(2**dacBits-1) # inputMaxVoltage is divided by numbers of input level and get inputV
        output = w.new_zeros((bsize, outsize))
        cellBits = phyArrParams.cellBits

        if not phyArrParams.useBNN or bnn_first_layer_flag:
            fixed_point_input, fixed_point_input_params = quantization_tensor(input, inputBits, TensorType.Normal)
            s1 = fixed_point_input_params[0].item()
        else:
            fixed_point_input, fixed_point_input_params = binarize_tensor(input, TensorType.Normal)
            s1 = fixed_point_input_params[0].item()

        if phyArrParams.inputVmode==input_v_mode.neg_pulse:
            neg_index = fixed_point_input<0
            neg_data_abs = fixed_point_input.abs()[neg_index]
        elif dacBits!=1:  #如果是补码模式，但是 dac不是1bit，这样是不能计算的，所以报错
            raise Exception("[Error] Complement code input mode, but dac is not 1 bit. dacbits = {}".format(dacBits))

        if arrType == phyArrMode.PN:
            if not phyArrParams.useBNN:
                fixed_point_w, fixed_point_w_params = quantization_tensor(w, weightBits, TensorType.PN) 
                s2 = fixed_point_w_params[0].item()
            else:
                fixed_point_w, fixed_point_w_params = binarize_tensor(w, TensorType.PN) 
                s2 = fixed_point_w_params[0].item()

            fixed_point_w_pos = fixed_point_w.clone()
            fixed_point_w_pos[fixed_point_w_pos<0] = 0
            fixed_point_w_neg = fixed_point_w.clone()
            fixed_point_w_neg[fixed_point_w_neg>0] = 0
            fixed_point_w_neg.abs_()

            mask_cell = (1 << cellBits)-1
            mask_in = (1<< dacBits) - 1
            for i in range(0, inputBits, dacBits):

                in_b = fixed_point_input.__rshift__(i).bitwise_and(mask_in)
                if phyArrParams.inputVmode==input_v_mode.neg_pulse:
                    in_b[neg_index] = -(neg_data_abs.__rshift__(i).bitwise_and(mask_in))

                if insize%phyArrParams.arrRowSize!=0: #补全输入电压到 物理阵列行数的倍数
                    temp_size = phyArrParams.arrRowSize-insize%phyArrParams.arrRowSize
                    in_b = torch.cat((in_b, in_b.new_zeros(bsize, temp_size)), 1)

                in_b = in_b.to(float_type).mul(inputV)
                in_b = in_b.reshape(bsize, -1, phyArrParams.arrRowSize)

                for j in range(0, weightBits, cellBits):
                    # 将权值矩阵中数据的各个比特拆分到不同阵列上
                    g_b_pos = fixed_point_w_pos.__rshift__(j).bitwise_and(mask_cell)
                    g_b_neg = fixed_point_w_neg.__rshift__(j).bitwise_and(mask_cell)

                    true_insize = insize
                    true_outsize = outsize

                    # 阵列行补全到物理阵列大小
                    if insize%phyArrParams.arrRowSize!=0:
                        temp_size = phyArrParams.arrRowSize-insize%phyArrParams.arrRowSize
                        g_b_pos = torch.cat((g_b_pos, g_b_pos.new_zeros(temp_size, outsize)), 0)
                        g_b_neg = torch.cat((g_b_neg, g_b_neg.new_zeros(temp_size, outsize)), 0)
                        true_insize += temp_size
                    
                    # 阵列列补全到物理阵列大小
                    if outsize%phyArrParams.arrColSize!=0:
                        temp_size = phyArrParams.arrColSize-outsize%phyArrParams.arrColSize
                        g_b_pos = torch.cat((g_b_pos, g_b_pos.new_zeros(true_insize, temp_size)), 1)
                        g_b_neg = torch.cat((g_b_neg, g_b_neg.new_zeros(true_insize, temp_size)), 1)
                        true_outsize += temp_size

                    numX = true_insize // phyArrParams.arrRowSize
                    numY = true_outsize // phyArrParams.arrColSize
                    
                    g_b_pos = g_b_pos.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)
                    g_b_neg = g_b_neg.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)

                    # g_b_pos && g_b_neg are spilt weight cell conductance with conductance variation
                    # g_b_pos_ && g_b_neg_ left for compensation usage
                    g_b_pos_novar = g_b_pos.reshape(-1, phyArrParams.arrRowSize, true_outsize)
                    g_b_neg_novar = g_b_neg.reshape(-1, phyArrParams.arrRowSize, true_outsize)
                    
                    g_b_pos = add_variation(g_b_pos_novar, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_network)
                    g_b_neg = add_variation(g_b_neg_novar, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_network)


                    # in_b [bsize, numX, arrRowSize]
                    # g_b_pos && g_b_neg [numX, arrRowSize, true_outsize], (true_outsize = numY * arrColSize > outsize)
                    if phyArrParams.useOUSize==False: # do not use OU size to cal, the entrie array is used.
                        if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop:
                            out_pos = no_ir_drop_cal(in_b, g_b_pos)
                            out_neg = no_ir_drop_cal(in_b, g_b_neg)
                        elif phyArrParams.IRdropMode==ir_drop_mode.ir2s:
                            if phyArrParams.cpp_ir_drop_accelerate_flag:
                                g_b_pos = g_b_pos.contiguous()
                                g_b_neg = g_b_neg.contiguous()
                                out_pos =  phyArrParams.cpp_ir_drop_accelerate.ir2s(in_b, g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, numX, numY, phyArrParams.ir_drop_iter_time, phyArrParams.r_wire, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params, False)
                                out_neg =  phyArrParams.cpp_ir_drop_accelerate.ir2s(in_b, g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, numX, numY, phyArrParams.ir_drop_iter_time, phyArrParams.r_wire, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params, False)
                            else:
                                out_pos = ir_drop_ir2s_fastsolve(in_b, g_b_pos, bsize, numX, numY)
                                out_neg = ir_drop_ir2s_fastsolve(in_b, g_b_neg, bsize, numX, numY)
                        elif phyArrParams.IRdropMode==ir_drop_mode.accurate_GS_math:
                            if phyArrParams.cpp_ir_drop_accelerate_flag:
                                # row_size and col_size should be >= 2
                                g_b_pos = g_b_pos.contiguous()
                                g_b_neg = g_b_neg.contiguous()
                                out_pos =  phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(in_b, g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, numX, numY, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params, False)
                                out_neg =  phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(in_b, g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, numX, numY, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params, False)
                            else:
                                out_pos = ir_drop_accurate_GSMethod(in_b, g_b_pos, bsize, numX, numY)
                                out_neg = ir_drop_accurate_GSMethod(in_b, g_b_neg, bsize, numX, numY)
                        elif phyArrParams.IRdropMode==ir_drop_mode.em:
                            out_pos = ir_drop_em_fastsolve(in_b, g_b_pos, bsize, numX, numY)
                            out_neg = ir_drop_em_fastsolve(in_b, g_b_neg, bsize, numX, numY)
                        else:
                            out_pos = ir_drop_accurate_math(in_b, g_b_pos, bsize, numX, numY)
                            out_neg = ir_drop_accurate_math(in_b, g_b_neg, bsize, numX, numY)

                        #为了消除高阻态本身的电流影响, 不去除以maxSumI，而是除以deltaConductance算出来的adcUseI
                        out_pos = out_pos.div(phyArrParams.adcUseI).mul(2**adcBits-1).round().mul(phyArrParams.arrRowSize*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
                        out_neg = out_neg.div(phyArrParams.adcUseI).mul(2**adcBits-1).round().mul(phyArrParams.arrRowSize*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
                        out_pos = out_pos.sum(0 if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop else 1)[:, 0:outsize] # out is [bsize, outsize]
                        out_neg = out_neg.sum(0 if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop else 1)[:, 0:outsize] # out is [bsize, outsize]
                    else: # use OU size to cal 
                        # reshape data for OU compensation usage
                        inv_ = in_b.reshape(bsize, numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0]) # [bsize, numX, #OURow, OURowSize]
                        g_block_pos_novar = g_b_pos_novar.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1]) # [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
                        g_block_neg_novar = g_b_neg_novar.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1]) # [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]

                        # if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.memCMOSOU:
                            #! memCMOS pre-processing compensation: boost input voltage
                            # in_b_pos = memCMOS_preOU_compensator(inv_, g_block_pos_novar, r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                            # in_b_neg = memCMOS_preOU_compensator(inv_, g_block_neg_novar, r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                            # in_b_pos = in_b_pos.reshape()
                            # in_b_neg = in_b_neg.reshape()
                            # pass

                        if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop:
                            # out [numX*ou_row_num, bsize, true_outsize]
                            out_pos = no_ir_drop_cal_ou(in_b, g_b_pos, phyArrParams.OUSize, bsize, true_outsize, numX)
                            out_neg = no_ir_drop_cal_ou(in_b, g_b_neg, phyArrParams.OUSize, bsize, true_outsize, numX)
                        # out [bsize, numX*num_ou_X, true_outsize]
                        elif phyArrParams.IRdropMode==ir_drop_mode.ir2s:
                            if phyArrParams.cpp_ir_drop_accelerate_flag==False:
                                raise Exception("[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate to True")
                            else:
                                g_b_pos = g_b_pos.contiguous()
                                g_b_neg = g_b_neg.contiguous()
                                out_pos =  phyArrParams.cpp_ir_drop_accelerate.ir2s_ou(in_b, g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, phyArrParams.r_wire, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params)
                                out_neg =  phyArrParams.cpp_ir_drop_accelerate.ir2s_ou(in_b, g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, phyArrParams.r_wire, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params)
                        # use gs method
                        # in_b [bsize, numX, row_size]
                        # g_b_pos && g_b_neg [numX, arrRowSize, true_outsize], (true_outsize = numY * arrColSize > outsize)
                        elif phyArrParams.IRdropMode==ir_drop_mode.accurate_GS_math:
                            #in_b is [bsize, numX, row_size]
                            if phyArrParams.cpp_ir_drop_accelerate_flag==False:
                                # raise Exception("[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate to True")
                                out_pos = ir_drop_OU(in_b, g_b_pos, bsize, numX, numY, arrType)
                                out_neg = ir_drop_OU(in_b, g_b_neg, bsize, numX, numY, arrType)
                            else:
                                # reassemble input voltages and cell conductance in mem
                                # in_b = in_b.contiguous()
                                g_b_pos = g_b_pos.contiguous()
                                g_b_neg = g_b_neg.contiguous()
                                out_pos = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b, g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params)
                                out_neg = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b, g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params)
                        else:
                            # raise Exception("[Error] OU Size unsupported ir drop mode")
                            # memCMOSOU每个OU的输入电压都不相同，无法使用上面几种方法，因而写在这里
                            if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.memCMOSOU:
                                pass
                                # in_b_pos []
                                # in_b_neg []
                                # out_pos []
                                # out_neg []
                                # in_b_pos = in_b_pos.contiguous()
                                # in_b_neg = in_b_neg.contiguous()
                                # g_b_pos = g_b_pos.contiguous()
                                # g_b_neg = g_b_neg.contiguous()
                                # out_pos = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b_pos, g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold)
                                # out_neg = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b_neg, g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold)

                        # if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.memCMOSOU:
                            #! memCMOS post-processing compensation: boost output voltage(current)
                            # processing at OU granularity, compensate at OU granularity
                            # TODO: rewrite memCMOSOU scheme & make it simpler
                            # out_pos = memCMOS_postOU_compensator(output_currents, x_[:, ii, :, jj, :].squeeze(1).squeeze(2), bsize,numX, numY, r=ii, c=jj)
                            # out_neg = memCMOS_postOU_compensator(output_currents, x_[:, ii, :, jj, :].squeeze(1).squeeze(2), bsize,numX, numY, r=ii, c=jj)
                            # raise Exception("MemCMOS compensator not implemented")

                        out_pos = out_pos.div(phyArrParams.adcOUuseI).mul(2**adcBits-1).round().mul(phyArrParams.OUSize[0]*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
                        out_neg = out_neg.div(phyArrParams.adcOUuseI).mul(2**adcBits-1).round().mul(phyArrParams.OUSize[0]*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
                        
                        # 过ADC后进行补偿
                        if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.icon:
                            # out [bsize, numX*num_ou_X, true_outsize]
                            # reshape到OU粒度
                            # print("I'm here")
                            out_pos = out_pos.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                            out_neg = out_neg.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                            # output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
                            # g_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
                            # compensated_output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
                            # TODO: rewrite icon scheme & make it simpler
                            out_pos = icon_compensator(out_pos, inv_, g_block_pos_novar, bsize, numX, numY, r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                            out_neg = icon_compensator(out_neg, inv_, g_block_neg_novar, bsize, numX, numY, r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                            # reshape回去
                            out_pos = out_pos.reshape(bsize, numX*phyArrParams.arrOUnum[0], -1)
                            out_neg = out_neg.reshape(bsize, numX*phyArrParams.arrOUnum[0], -1)
                        # mirrored 需要先计算出转置后阵列输出电流，与正常计算输出电流的平均，之后才可以进行补偿
                        elif phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.mirrored:
                            # out [bsize, numX*num_ou_X, true_outsize]
                            # reshape到OU粒度
                            out_pos = out_pos.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                            out_neg = out_neg.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                            # 对输入数据进行转置
                            mirrored_in_b = in_b.flip([2])
                            mirrored_g_b_pos_novar = g_b_pos_novar.reshape(numX, phyArrParams.arrRowSize, numY, phyArrParams.arrColSize)\
                                                        .flip([1, 3])\
                                                        .reshape(-1, phyArrParams.arrRowSize, true_outsize)
                            mirrored_g_b_neg_novar = g_b_neg_novar.reshape(numX, phyArrParams.arrRowSize, numY, phyArrParams.arrColSize)\
                                                        .flip([1, 3])\
                                                        .reshape(-1, phyArrParams.arrRowSize, true_outsize)
                            
                            mirrored_g_b_pos = add_variation(mirrored_g_b_pos_novar, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_mirrored)
                            mirrored_g_b_neg = add_variation(mirrored_g_b_neg_novar, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_mirrored)
                            
                            # mirrored_out_pos = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(mirrored_in_b, mirrored_g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold)
                            # mirrored_out_neg = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(mirrored_in_b, mirrored_g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold)
                            mirrored_out_pos = ir_drop_OU(mirrored_in_b, mirrored_g_b_pos, bsize, numX, numY, arrType)
                            mirrored_out_neg = ir_drop_OU(mirrored_in_b, mirrored_g_b_neg, bsize, numX, numY, arrType)

                            # reshape到OU粒度
                            mirrored_out_pos = mirrored_out_pos.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                            mirrored_out_neg = mirrored_out_neg.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])

                            # 对输出数据进行转置
                            mirrored_out_pos = mirrored_out_pos.flip([2, 4, 5])
                            mirrored_out_neg = mirrored_out_neg.flip([2, 4, 5])

                            mirrored_out_pos = mirrored_out_pos.div(phyArrParams.adcOUuseI).mul(2**adcBits-1).round().mul(phyArrParams.OUSize[0]*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
                            mirrored_out_neg = mirrored_out_neg.div(phyArrParams.adcOUuseI).mul(2**adcBits-1).round().mul(phyArrParams.OUSize[0]*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))

                            out_pos_ = out_pos + mirrored_out_pos
                            out_neg_ = out_neg + mirrored_out_neg
                            
                            # TODO: rewrite mirrored scheme & make it simpler
                            out_pos_ = mirrored_compensator(out_pos_, inv_, g_block_pos_novar, bsize, numX, numY, r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                            out_neg_ = mirrored_compensator(out_neg_, inv_, g_block_neg_novar, bsize, numX, numY, r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)

                            out_pos = out_pos_ / 2
                            out_neg = out_neg_ / 2

                            # reshape回去
                            out_pos = out_pos.reshape(bsize, -1, true_outsize)
                            out_neg = out_neg.reshape(bsize, -1, true_outsize)
                        
                        # print(out_pos)
                        # print(out_neg)
                        # exit(0)

                        out_pos = out_pos.sum(0 if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop else 1)[:, 0:outsize] # out is [bsize, outsize]
                        out_neg = out_neg.sum(0 if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop else 1)[:, 0:outsize] # out is [bsize, outsize]
                        
                    if i!=inputBits-1 or phyArrParams.inputVmode!=input_v_mode.complement_code:
                        output = output.add((out_pos-out_neg).mul(2**(i+j+s1+s2)))
                    else:
                        output = output.sub((out_pos-out_neg).mul(2**(i+j+s1+s2)))
        elif arrType==phyArrMode.Ref:

            if not phyArrParams.useBNN:
                fixed_point_w, fixed_point_w_params = quantization_tensor(w, weightBits, TensorType.Normal) 
                s2 = fixed_point_w_params[0].item()
            else:
                weightBits = 2
                fixed_point_w, fixed_point_w_params = binarize_tensor(w, TensorType.Normal) 
                s2 = fixed_point_w_params[0].item()

            plus = 2**(weightBits-1)
            maximum = 2**(weightBits)-1
            fixed_point_w = fixed_point_w.add(plus)   #ref column mode, add bias.
            fixed_point_w[fixed_point_w>maximum] = maximum  # if max than 2*plus-1, reset to 2*plus-1
 
            mask_cell = (1 << cellBits)-1
            mask_in = (1 << dacBits) - 1
            for i in range(0, inputBits, dacBits):
                in_b = fixed_point_input.__rshift__(i).bitwise_and(mask_in)
                if phyArrParams.inputVmode==input_v_mode.neg_pulse:
                    in_b[neg_index] = -(neg_data_abs.__rshift__(i).bitwise_and(mask_in))

                if insize%phyArrParams.arrRowSize!=0: #补全输入电压到 物理阵列行数的倍数
                    temp_size = phyArrParams.arrRowSize-insize%phyArrParams.arrRowSize
                    in_b = torch.cat((in_b, in_b.new_zeros(bsize, temp_size)), 1)

                in_b = in_b.to(float_type).mul(inputV)
                in_b = in_b.reshape(bsize, -1, phyArrParams.arrRowSize)

                for j in range(0, weightBits, cellBits):
                    g_b = fixed_point_w.__rshift__(j).bitwise_and(mask_cell)

                    true_insize = insize
                    true_outsize = outsize

                    # 阵列行补全到物理阵列大小
                    if insize%phyArrParams.arrRowSize!=0:
                        temp_size = phyArrParams.arrRowSize-insize%phyArrParams.arrRowSize
                        g_b = torch.cat((g_b, g_b.new_zeros(temp_size, outsize)), 0)
                        true_insize += temp_size
                    
                    # 阵列列补全到物理阵列大小
                    if outsize%phyArrParams.arrColSize!=0:
                        temp_size = phyArrParams.arrColSize-outsize%phyArrParams.arrColSize
                        g_b = torch.cat((g_b, g_b.new_zeros(true_insize, temp_size)), 1)
                        true_outsize += temp_size

                    numX = true_insize//phyArrParams.arrRowSize
                    numY = true_outsize//phyArrParams.arrColSize

                    if phyArrParams.useOUSize==False or phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop:
                        # insert additional ref column in each crx array
                        temp_g_b = g_b.reshape(true_insize, numY, -1)
                        additional_col = g_b.new_zeros(true_insize, numY, 2)
                        additional_col[..., -1] = 1
                        g_b = torch.cat((temp_g_b, additional_col), 2)
                        #now g_b is [true_insize, numY, arrColSize+2]
                        g_b = g_b.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)
                        true_outsize += 2*numY                    
                    else:
                        # insert additional ref column in each crx array
                        temp_g_b = g_b.reshape(true_insize, numY, -1)
                        additional_col = g_b.new_zeros(true_insize, numY, phyArrParams.OUSize[1])
                        additional_col[..., -phyArrParams.OUSize[1]+1] = 1
                        g_b = torch.cat((temp_g_b, additional_col), 2)
                        #now g_b is [true_insize, numY, arrColSize+ousize[1]]
                        g_b = g_b.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)
                        g_b[..., -phyArrParams.OUSize[1]+2:] = 1e-12 # set a very high resistance
                        true_outsize += numY*phyArrParams.OUSize[1]

                    g_b_novar = g_b.reshape(numX, phyArrParams.arrRowSize, true_outsize)
                    #in_b is [bsize, numX, arrRowSize]
                    g_b = add_variation(g_b_novar, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_network)

                    if phyArrParams.useOUSize==False:
                        if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop:
                            out = no_ir_drop_cal(in_b, g_b)  # out is [numX, bsize, true_outsize]
                            out = out.reshape(numX, bsize, numY, -1)
                        elif phyArrParams.IRdropMode==ir_drop_mode.ir2s:
                            if phyArrParams.cpp_ir_drop_accelerate_flag:
                                g_b = g_b.contiguous()
                                out =  phyArrParams.cpp_ir_drop_accelerate.ir2s(in_b, g_b, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize+2, numX, numY, phyArrParams.ir_drop_iter_time, phyArrParams.r_wire, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params, False)
                            else:
                                out = ir_drop_ir2s_fastsolve(in_b, g_b, bsize, numX, numY, arrColSize=phyArrParams.arrColSize+2)
                            out = out.reshape(bsize, numX, numY, -1)
                        elif phyArrParams.IRdropMode==ir_drop_mode.accurate_GS_math:
                            if phyArrParams.cpp_ir_drop_accelerate_flag:
                                g_b = g_b.contiguous()
                                out =  phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(in_b, g_b, bsize, phyArrParams.arrRowSize, 2+phyArrParams.arrColSize, numX, numY, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params, False)
                            else:
                                out = ir_drop_accurate_GSMethod(in_b, g_b, bsize, numX, numY, arrColSize=phyArrParams.arrColSize+2)
                        elif phyArrParams.IRdropMode==ir_drop_mode.em:
                            out = ir_drop_em_fastsolve(in_b, g_b, bsize, numX, numY, arrColSize=phyArrParams.arrColSize+2)
                        else:                            
                            raise Exception("[Error] ref col mode not support ir drop accurate math")

                        out = out.div(phyArrParams.adcUseI).mul(2**adcBits-1).round()
                        zeros_col = out[..., phyArrParams.arrColSize:phyArrParams.arrColSize+1]
                        out = out - zeros_col
                        if j==0:
                            ones_col =  out[..., phyArrParams.arrColSize+1:phyArrParams.arrColSize+2]
                            out = out - ones_col.mul(plus).round()
                        out = out[..., 0:phyArrParams.arrColSize].sum(0 if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop else 1).reshape(bsize, -1)[:, 0:outsize]
                        out = out.mul(phyArrParams.arrRowSize*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
                    else: # ou mode
                        zero_index, one_index = -2, -1
                        if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop:
                            out = no_ir_drop_cal_ou(in_b, g_b, phyArrParams.OUSize, bsize, true_outsize, numX)
                            out = out.reshape(-1, bsize, numY, phyArrParams.arrColSize+2)
                        elif phyArrParams.IRdropMode==ir_drop_mode.ir2s:
                            if phyArrParams.cpp_ir_drop_accelerate_flag==False:
                                raise Exception("[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate_flag to True")
                            g_b = g_b.contiguous()
                            out =  phyArrParams.cpp_ir_drop_accelerate.ir2s_ou(in_b, g_b, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize+phyArrParams.OUSize[1], phyArrParams.OUSize[0], phyArrParams.OUSize[1], numX, numY, phyArrParams.ir_drop_iter_time, phyArrParams.r_wire, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params)
                            zero_index, one_index = -phyArrParams.OUSize[1], -phyArrParams.OUSize[1]+1
                            out = out.reshape(bsize, -1, numY, phyArrParams.arrColSize+phyArrParams.OUSize[1])
                        elif phyArrParams.IRdropMode==ir_drop_mode.accurate_GS_math:
                            if phyArrParams.cpp_ir_drop_accelerate_flag==False:
                                raise Exception("[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate to True")
                            g_b = g_b.contiguous()
                            out =  phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b, g_b, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize+phyArrParams.OUSize[1], phyArrParams.OUSize[0], phyArrParams.OUSize[1], numX, numY, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold, phyArrParams.nonlinear_cell_enable, phyArrParams.nonlinear_params)
                            zero_index, one_index = -phyArrParams.OUSize[1], -phyArrParams.OUSize[1]+1
                            out = out.reshape(bsize, -1, numY, phyArrParams.arrColSize+phyArrParams.OUSize[1])
                        
                            
                        out = out.div(phyArrParams.adcOUuseI).mul(2**adcBits-1).round()
                        zeros_col = out[..., zero_index:zero_index+1]
                        out = out - zeros_col
                        if j==0:
                            if one_index+1==0:
                                ones_col =  out[..., one_index:]
                            else:
                                ones_col =  out[..., one_index:one_index+1]
                            out = out - ones_col.mul(plus).round()
                        out = out[..., 0:phyArrParams.arrColSize].sum(0 if phyArrParams.IRdropMode==ir_drop_mode.no_ir_drop else 1).reshape(bsize, -1)[:, 0:outsize]
                        out = out.mul(phyArrParams.OUSize[0]*(2**dacBits-1)*(2**phyArrParams.cellBits-1)/(2**adcBits-1))
             
                    
                    if i!=inputBits-1 or phyArrParams.inputVmode!=input_v_mode.complement_code:
                        output = output.add(out.mul(2**(i+j+s1+s2)))
                    else:
                        output = output.sub(out.mul(2**(i+j+s1+s2)))

        return output

# 按照比特拆分后的矩阵向量乘法
# 此处不再进行比特拆分，只把矩阵划分到不同阵列，并计算VMM
# 此处实现多种不同阵列级乘法操作、阵列/单元级非理想因素，以及相应的补偿方法
# in_b [bsize, insize]
# g_b_pos [insize, outsize]
# g_b_neg [insize, outsize] or None
def matmul(in_b: Tensor, g_b_pos: Tensor, g_b_neg, j, inputBits: int, dacBits: int, adcBits: int, \
        outputBits: int, weightBits: int, arrType: phyArrMode, float_type: torch.dtype) -> Tensor:

    def IRdrop_matmul(in_b: Tensor, g_b_pos: Tensor, g_b_neg, arrType: phyArrMode):
        # IR drop计算
        if arrType == phyArrMode.PN:
            # in_b [bsize, numX, arrRowSize]
            # g_b_pos && g_b_neg [numX, arrRowSize, true_outsize], (true_outsize = numY * arrColSize > outsize)
            if phyArrParams.useOUSize == False:  # do not use OU size to cal, the entrie array is used.
                if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
                    out_pos = no_ir_drop_cal(in_b, g_b_pos)
                    out_neg = no_ir_drop_cal(in_b, g_b_neg)
                elif phyArrParams.IRdropMode == ir_drop_mode.ir2s:
                    if phyArrParams.cpp_ir_drop_accelerate_flag:
                        in_b = in_b.contiguous()
                        g_b_pos = g_b_pos.contiguous()
                        g_b_neg = g_b_neg.contiguous()
                        out_pos = phyArrParams.cpp_ir_drop_accelerate.ir2s(in_b, g_b_pos, bsize,
                                                                           phyArrParams.arrRowSize,
                                                                           phyArrParams.arrColSize, numX, numY,
                                                                           phyArrParams.ir_drop_iter_time,
                                                                           phyArrParams.r_wire,
                                                                           phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                           phyArrParams.ir_drop_less_than_threshold, 
                                                                           phyArrParams.nonlinear_cell_enable,
                                                                           phyArrParams.nonlinear_params,
                                                                           False)
                        out_neg = phyArrParams.cpp_ir_drop_accelerate.ir2s(in_b, g_b_neg, bsize,
                                                                           phyArrParams.arrRowSize,
                                                                           phyArrParams.arrColSize, numX, numY,
                                                                           phyArrParams.ir_drop_iter_time,
                                                                           phyArrParams.r_wire,
                                                                           phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                           phyArrParams.ir_drop_less_than_threshold,
                                                                           phyArrParams.nonlinear_cell_enable,
                                                                           phyArrParams.nonlinear_params,
                                                                           False)
                    else:
                        out_pos = ir_drop_ir2s_fastsolve(in_b, g_b_pos, bsize, numX, numY)
                        out_neg = ir_drop_ir2s_fastsolve(in_b, g_b_neg, bsize, numX, numY)
                elif phyArrParams.IRdropMode == ir_drop_mode.accurate_GS_math:
                    if phyArrParams.cpp_ir_drop_accelerate_flag:
                        # row_size and col_size should be >= 2
                        in_b = in_b.contiguous()
                        g_b_pos = g_b_pos.contiguous()
                        g_b_neg = g_b_neg.contiguous()
                        out_pos = phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(in_b, g_b_pos, bsize,
                                                                                 phyArrParams.arrRowSize,
                                                                                 phyArrParams.arrColSize, numX,
                                                                                 numY,
                                                                                 phyArrParams.ir_drop_iter_time,
                                                                                 1 / phyArrParams.r_wire,
                                                                                 phyArrParams.ir_drop_GS_method_beta_scale,
                                                                                 phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                                 phyArrParams.ir_drop_less_than_threshold,
                                                                                 phyArrParams.nonlinear_cell_enable,
                                                                                 phyArrParams.nonlinear_params,
                                                                                 False)
                        out_neg = phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(in_b, g_b_neg, bsize,
                                                                                 phyArrParams.arrRowSize,
                                                                                 phyArrParams.arrColSize, numX,
                                                                                 numY,
                                                                                 phyArrParams.ir_drop_iter_time,
                                                                                 1 / phyArrParams.r_wire,
                                                                                 phyArrParams.ir_drop_GS_method_beta_scale,
                                                                                 phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                                 phyArrParams.ir_drop_less_than_threshold,
                                                                                 phyArrParams.nonlinear_cell_enable, 
                                                                                 phyArrParams.nonlinear_params, 
                                                                                 False)
                    else:
                        out_pos = ir_drop_accurate_GSMethod(in_b, g_b_pos, bsize, numX, numY)
                        out_neg = ir_drop_accurate_GSMethod(in_b, g_b_neg, bsize, numX, numY)
                elif phyArrParams.IRdropMode == ir_drop_mode.em:
                    out_pos = ir_drop_em_fastsolve(in_b, g_b_pos, bsize, numX, numY)
                    out_neg = ir_drop_em_fastsolve(in_b, g_b_neg, bsize, numX, numY)
                elif phyArrParams.IRdropMode == ir_drop_mode.scn:
                    out_pos = ir_drop_scn_model(in_b, g_b_pos, bsize, numX, numY)
                    out_neg = ir_drop_scn_model(in_b, g_b_neg, bsize, numX, numY)
                else:
                    out_pos = ir_drop_accurate_math(in_b, g_b_pos, bsize, numX, numY)
                    out_neg = ir_drop_accurate_math(in_b, g_b_neg, bsize, numX, numY)
            else:  # use OU size to cal
                if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
                    # out [numX*ou_row_num, bsize, true_outsize]
                    out_pos = no_ir_drop_cal_ou(in_b, g_b_pos, phyArrParams.OUSize, bsize, true_outsize, numX)
                    out_neg = no_ir_drop_cal_ou(in_b, g_b_neg, phyArrParams.OUSize, bsize, true_outsize, numX)
                # out [bsize, numX*num_ou_X, true_outsize]
                elif phyArrParams.IRdropMode == ir_drop_mode.ir2s:
                    if phyArrParams.cpp_ir_drop_accelerate_flag == False:
                        raise Exception(
                            "[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate to True")
                    else:
                        in_b = in_b.contiguous()
                        g_b_pos = g_b_pos.contiguous()
                        g_b_neg = g_b_neg.contiguous()
                        out_pos = phyArrParams.cpp_ir_drop_accelerate.ir2s_ou(in_b, g_b_pos, bsize,
                                                                              phyArrParams.arrRowSize,
                                                                              phyArrParams.arrColSize,
                                                                              phyArrParams.OUSize[0],
                                                                              phyArrParams.OUSize[1],
                                                                              true_insize // phyArrParams.arrRowSize,
                                                                              true_outsize // phyArrParams.arrColSize,
                                                                              phyArrParams.ir_drop_iter_time,
                                                                              phyArrParams.r_wire,
                                                                              phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                              phyArrParams.ir_drop_less_than_threshold,
                                                                              phyArrParams.nonlinear_cell_enable, 
                                                                              phyArrParams.nonlinear_params)
                        out_neg = phyArrParams.cpp_ir_drop_accelerate.ir2s_ou(in_b, g_b_neg, bsize,
                                                                              phyArrParams.arrRowSize,
                                                                              phyArrParams.arrColSize,
                                                                              phyArrParams.OUSize[0],
                                                                              phyArrParams.OUSize[1],
                                                                              true_insize // phyArrParams.arrRowSize,
                                                                              true_outsize // phyArrParams.arrColSize,
                                                                              phyArrParams.ir_drop_iter_time,
                                                                              phyArrParams.r_wire,
                                                                              phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                              phyArrParams.ir_drop_less_than_threshold,
                                                                              phyArrParams.nonlinear_cell_enable, 
                                                                              phyArrParams.nonlinear_params)
                # use gs method
                # in_b [bsize, numX, row_size]
                # g_b_pos && g_b_neg [numX, arrRowSize, true_outsize], (true_outsize = numY * arrColSize > outsize)
                elif phyArrParams.IRdropMode == ir_drop_mode.accurate_GS_math:
                    # in_b is [bsize, numX, row_size]
                    if phyArrParams.cpp_ir_drop_accelerate_flag == False:
                        # raise Exception("[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate to True")
                        out_pos = ir_drop_OU(in_b, g_b_pos, bsize, numX, numY, arrType)
                        out_neg = ir_drop_OU(in_b, g_b_neg, bsize, numX, numY, arrType)
                    else:
                        # reassemble input voltages and cell conductance in mem
                        in_b = in_b.contiguous()
                        g_b_pos = g_b_pos.contiguous()
                        g_b_neg = g_b_neg.contiguous()
                        out_pos = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b, g_b_pos, bsize,
                                                                               phyArrParams.arrRowSize,
                                                                               phyArrParams.arrColSize,
                                                                               phyArrParams.OUSize[0],
                                                                               phyArrParams.OUSize[1],
                                                                               true_insize // phyArrParams.arrRowSize,
                                                                               true_outsize // phyArrParams.arrColSize,
                                                                               phyArrParams.ir_drop_iter_time,
                                                                               1 / phyArrParams.r_wire,
                                                                               phyArrParams.ir_drop_GS_method_beta_scale,
                                                                               phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                               phyArrParams.ir_drop_less_than_threshold,
                                                                               phyArrParams.nonlinear_cell_enable, 
                                                                               phyArrParams.nonlinear_params)
                        out_neg = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b, g_b_neg, bsize,
                                                                               phyArrParams.arrRowSize,
                                                                               phyArrParams.arrColSize,
                                                                               phyArrParams.OUSize[0],
                                                                               phyArrParams.OUSize[1],
                                                                               true_insize // phyArrParams.arrRowSize,
                                                                               true_outsize // phyArrParams.arrColSize,
                                                                               phyArrParams.ir_drop_iter_time,
                                                                               1 / phyArrParams.r_wire,
                                                                               phyArrParams.ir_drop_GS_method_beta_scale,
                                                                               phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                               phyArrParams.ir_drop_less_than_threshold,
                                                                               phyArrParams.nonlinear_cell_enable,
                                                                               phyArrParams.nonlinear_params)
                else:
                    # raise Exception("[Error] OU Size unsupported ir drop mode")
                    # memCMOSOU每个OU的输入电压都不相同，无法使用上面几种方法，因而写在这里
                    if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.memCMOS:
                        pass
            return out_pos, out_neg
        else:
            g_b = g_b_pos
            if phyArrParams.useOUSize == False:
                if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
                    out = no_ir_drop_cal(in_b, g_b)  # out is [numX, bsize, true_outsize]
                elif phyArrParams.IRdropMode == ir_drop_mode.ir2s:
                    if phyArrParams.cpp_ir_drop_accelerate_flag:
                        in_b = in_b.contiguous()
                        g_b = g_b.contiguous()
                        out = phyArrParams.cpp_ir_drop_accelerate.ir2s(in_b, g_b, bsize,
                                                                       phyArrParams.arrRowSize,
                                                                       phyArrParams.arrColSize + 2, numX, numY,
                                                                       phyArrParams.ir_drop_iter_time,
                                                                       phyArrParams.r_wire,
                                                                       phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                       phyArrParams.ir_drop_less_than_threshold,
                                                                       phyArrParams.nonlinear_cell_enable, 
                                                                       phyArrParams.nonlinear_params, 
                                                                       False)
                    else:
                        out = ir_drop_ir2s_fastsolve(in_b, g_b, bsize, numX, numY,
                                                     arrColSize=phyArrParams.arrColSize + 2)
                elif phyArrParams.IRdropMode == ir_drop_mode.accurate_GS_math:
                    if phyArrParams.cpp_ir_drop_accelerate_flag:
                        in_b = in_b.contiguous()
                        g_b = g_b.contiguous()
                        out = phyArrParams.cpp_ir_drop_accelerate.ir_drop_gs(in_b, g_b, bsize,
                                                                             phyArrParams.arrRowSize,
                                                                             2 + phyArrParams.arrColSize, numX,
                                                                             numY,
                                                                             phyArrParams.ir_drop_iter_time,
                                                                             1 / phyArrParams.r_wire,
                                                                             phyArrParams.ir_drop_GS_method_beta_scale,
                                                                             phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                             phyArrParams.ir_drop_less_than_threshold,
                                                                             phyArrParams.nonlinear_cell_enable, 
                                                                             phyArrParams.nonlinear_params, 
                                                                             False)
                    else:
                        out = ir_drop_accurate_GSMethod(in_b, g_b, bsize, numX, numY,
                                                        arrColSize=phyArrParams.arrColSize + 2)
                elif phyArrParams.IRdropMode == ir_drop_mode.em:
                        out = ir_drop_em_fastsolve(in_b, g_b, bsize, numX, numY, arrColSize=phyArrParams.arrColSize+2)
                elif phyArrParams.IRdropMode == ir_drop_mode.scn:
                        out = ir_drop_scn_model(in_b, g_b, bsize, numX, numY, arrColSize=phyArrParams.arrColSize+2)
                else:
                    raise Exception("[Error] ref col mode not support ir drop accurate math")
            else:  # ou mode
                if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
                    out = no_ir_drop_cal_ou(in_b, g_b, phyArrParams.OUSize, bsize, true_outsize, numX)
                elif phyArrParams.IRdropMode == ir_drop_mode.ir2s:
                    if phyArrParams.cpp_ir_drop_accelerate_flag == False:
                        raise Exception(
                            "[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate_flag to True")
                    else:
                        in_b = in_b.contiguous()
                        g_b = g_b.contiguous()
                        out = phyArrParams.cpp_ir_drop_accelerate.ir2s_ou(in_b, g_b, bsize, phyArrParams.arrRowSize,
                                                                          phyArrParams.arrColSize +
                                                                          phyArrParams.OUSize[1],
                                                                          phyArrParams.OUSize[0],
                                                                          phyArrParams.OUSize[1], numX, numY,
                                                                          phyArrParams.ir_drop_iter_time,
                                                                          phyArrParams.r_wire,
                                                                          phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                          phyArrParams.ir_drop_less_than_threshold,
                                                                          phyArrParams.nonlinear_cell_enable, 
                                                                          phyArrParams.nonlinear_params)
                elif phyArrParams.IRdropMode == ir_drop_mode.accurate_GS_math:
                    if phyArrParams.cpp_ir_drop_accelerate_flag == False:
                        out = ir_drop_OU(in_b, g_b, bsize, numX, numY, arrType)
                        # raise Exception(
                        #     "[Error] Now OU mode is only realized in cpp cuda code. You should set cpp_ir_drop_accelerate to True")
                    else:
                        in_b = in_b.contiguous()
                        g_b = g_b.contiguous()
                        out = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(in_b, g_b, bsize,
                                                                           phyArrParams.arrRowSize,
                                                                           phyArrParams.arrColSize +
                                                                           phyArrParams.OUSize[1],
                                                                           phyArrParams.OUSize[0],
                                                                           phyArrParams.OUSize[1], numX, numY,
                                                                           phyArrParams.ir_drop_iter_time,
                                                                           1 / phyArrParams.r_wire,
                                                                           phyArrParams.ir_drop_GS_method_beta_scale,
                                                                           phyArrParams.ir_drop_less_than_thresholdBreak_enable,
                                                                           phyArrParams.ir_drop_less_than_threshold,
                                                                           phyArrParams.nonlinear_cell_enable, 
                                                                           phyArrParams.nonlinear_params)
            return out, None

    bsize = in_b.size(0)
    insize = g_b_pos.size(0)
    outsize = g_b_pos.size(1)
    minCellC = phyArrParams.cellMinConduct
    inputV = phyArrParams.inputMaxVoltage / (2 ** dacBits - 1)  # inputMaxVoltage is divided by numbers of input level and get inputV

    true_insize = insize
    true_outsize = outsize

    # 补全输入电压到 物理阵列行数的倍数
    # in_b [bsize, true_insize]
    if insize % phyArrParams.arrRowSize != 0:
        temp_size = phyArrParams.arrRowSize - insize % phyArrParams.arrRowSize
        in_b = torch.cat((in_b, in_b.new_zeros(bsize, temp_size)), 1)

    # 阵列行补全到物理阵列大小
    if insize % phyArrParams.arrRowSize != 0:
        temp_size = phyArrParams.arrRowSize - insize % phyArrParams.arrRowSize
        g_b_pos = torch.cat((g_b_pos, g_b_pos.new_zeros(temp_size, outsize)), 0)
        if arrType == phyArrMode.PN:
            # 参考列里面没有g_b_neg
            g_b_neg = torch.cat((g_b_neg, g_b_neg.new_zeros(temp_size, outsize)), 0)
        true_insize += temp_size

    # 阵列列补全到物理阵列大小
    if outsize % phyArrParams.arrColSize != 0:
        temp_size = phyArrParams.arrColSize - outsize % phyArrParams.arrColSize
        g_b_pos = torch.cat((g_b_pos, g_b_pos.new_zeros(true_insize, temp_size)), 1)
        if arrType == phyArrMode.PN:
            g_b_neg = torch.cat((g_b_neg, g_b_neg.new_zeros(true_insize, temp_size)), 1)
        true_outsize += temp_size

    numX = true_insize // phyArrParams.arrRowSize
    numY = true_outsize // phyArrParams.arrColSize

    # reserved for compensation usage
    input_pattern = in_b.clone()
    if arrType == phyArrMode.PN:
        data_pos_pattern = g_b_pos.clone()
        data_neg_pattern = g_b_neg.clone()
    else:
        data_pattern = g_b_pos.clone()

    # 输入数据变换
    # in_b [bsize, numX, arrRowSize]
    in_b = in_b.to(float_type).mul(inputV)
    in_b = in_b.reshape(bsize, -1, phyArrParams.arrRowSize)

    # array变换以及加参考列
    if arrType == phyArrMode.PN:
        g_b_pos = g_b_pos.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)
        g_b_neg = g_b_neg.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)

        g_b_pos = g_b_pos.reshape(-1, phyArrParams.arrRowSize, true_outsize)
        g_b_neg = g_b_neg.reshape(-1, phyArrParams.arrRowSize, true_outsize)

        g_b_pos = add_variation(g_b_pos, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_network)
        g_b_neg = add_variation(g_b_neg, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_network)
    else:
        plus = 2 ** (weightBits - 1)
        g_b = g_b_pos

        if phyArrParams.useOUSize == False or phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
            # insert additional ref column in each crx array
            temp_g_b = g_b.reshape(true_insize, numY, -1)
            additional_col = g_b.new_zeros(true_insize, numY, 2)
            additional_col[..., -1] = 1
            g_b = torch.cat((temp_g_b, additional_col), 2)
            # now g_b is [true_insize, numY, arrColSize+2]
            g_b = g_b.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)
            true_outsize += 2 * numY
        else:
            # insert additional ref column in each crx array
            temp_g_b = g_b.reshape(true_insize, numY, -1)
            additional_col = g_b.new_zeros(true_insize, numY, phyArrParams.OUSize[1])
            additional_col[..., -phyArrParams.OUSize[1] + 1] = 1
            g_b = torch.cat((temp_g_b, additional_col), 2)
            # now g_b is [true_insize, numY, arrColSize+ousize[1]]
            g_b = g_b.to(float_type).mul(phyArrParams.cellDeltaConduct).add(minCellC)
            g_b[..., -phyArrParams.OUSize[
                1] + 2:] = 1e-20  # set a very high resistance (all these cells are not supposed to appear)
            true_outsize += numY * phyArrParams.OUSize[1]

        g_b = g_b.reshape(numX, phyArrParams.arrRowSize, true_outsize)
        # in_b is [bsize, numX, arrRowSize]
        g_b = add_variation(g_b, sigma=phyArrParams.sigma, rand_gen=variations.rand_gen_network)

    # pre IR drop compensation, eg: input voltage modification, cell conductance modification
    # if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.memCMOS:
    #     if arrType == phyArrMode.PN:
    #         if phyArrParams.IRdropCompOU == False:
    #             pass
    #         else:
    #             in_b_pos = memCMOS_preOU_compensator(in_b, g_b_pos_novar, bsize, numX, numY)
    #             in_b_neg = memCMOS_preOU_compensator(in_b, g_b_neg_novar, bsize, numX, numY)
    #     else:
    #         if phyArrParams.IRdropCompOU == False:
    #             pass
    #         else:
    #             in_b = memCMOS_preOU_compensator(in_b, g_b_novar, bsize, numX, numY)

    # if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.irdrop_deembedding:
        # if arrType == phyArrMode.PN:
        #     if phyArrParams.IRdropCompOU == False:
        #         pass
        #     else:
        #         pass
        # else:
        #     if phyArrParams.IRdropCompOU == False:
        #         pass
        #     else:
        #         pass

    # IR drop matmul
    if arrType == phyArrMode.PN:
        out_pos, out_neg = IRdrop_matmul(in_b, g_b_pos, g_b_neg, arrType)
    else:
        out, _ = IRdrop_matmul(in_b, g_b, None, arrType)

    # post IR drop compensation, eg: output current modification
    if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.memCMOS:
        # ! memCMOS post-processing compensation: boost output voltage(current)
        # processing at OU granularity, compensate at OU granularity
        # TODO: rewrite memCMOSOU scheme & make it simpler
        # out_pos = memCMOS_postOU_compensator(output_currents, x_[:, ii, :, jj, :].squeeze(1).squeeze(2), bsize,numX, numY, r=ii, c=jj)
        # out_neg = memCMOS_postOU_compensator(output_currents, x_[:, ii, :, jj, :].squeeze(1).squeeze(2), bsize,numX, numY, r=ii, c=jj)
        raise Exception("MemCMOS compensator not implemented")

    # ADC转换
    if arrType == phyArrMode.PN:
        if phyArrParams.useOUSize == False:
            # 为了消除高阻态本身的电流影响, 不去除以maxSumI，而是除以deltaConductance算出来的adcUseI
            out_pos = out_pos.div(phyArrParams.adcUseI).mul(2 ** adcBits - 1).round()
            out_neg = out_neg.div(phyArrParams.adcUseI).mul(2 ** adcBits - 1).round()

        else:
            out_pos = out_pos.div(phyArrParams.adcOUuseI).mul(2 ** adcBits - 1).round()
            out_neg = out_neg.div(phyArrParams.adcOUuseI).mul(2 ** adcBits - 1).round()
    else:
        if phyArrParams.useOUSize == False:
            out = out.div(phyArrParams.adcUseI).mul(2 ** adcBits - 1).round()
        else:
            out = out.div(phyArrParams.adcOUuseI).mul(2 ** adcBits - 1).round()

    # post ADC compensation, eg: output result modification
    if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.icon:
        if phyArrParams.useOUSize and phyArrParams.IRdropCompOU:
            # POUCOU
            if arrType == phyArrMode.PN:
                # out [bsize, numX*num_ou_X, true_outsize]
                # reshape到OU粒度
                out_pos = out_pos.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                          phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                out_neg = out_neg.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                          phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
                # output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
                # input_pattern [bsize, true_insize]
                # data_pos_pattern && data_neg_pattern [true_insize, true_outsize]
                # compensated_output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
                # TODO: rewrite icon scheme & make it simpler
                out_pos = icon_compensator(out_pos, input_pattern, data_pos_pattern, bsize, numX, numY, arrType,
                                           r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                out_neg = icon_compensator(out_neg, input_pattern, data_neg_pattern, bsize, numX, numY, arrType,
                                           r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                # reshape回去
                out_pos = out_pos.reshape(bsize, numX * phyArrParams.arrOUnum[0], -1)
                out_neg = out_neg.reshape(bsize, numX * phyArrParams.arrOUnum[0], -1)
            else:
                # out [bsize, numX*num_ou_X, true_outsize]
                # reshape到OU粒度
                out = out.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                          phyArrParams.arrOUnum[1]+1, phyArrParams.OUSize[1])
                # output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
                # input_pattern [bsize, true_insize]
                # data_pattern [true_insize, true_outsize]
                # compensated_output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
                # TODO: rewrite icon scheme & make it simpler
                out = icon_compensator(out, input_pattern, data_pattern, bsize, numX, numY, arrType,
                                           r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
                # reshape回去
                out = out.reshape(bsize, numX * phyArrParams.arrOUnum[0], -1)
        else:
            raise Exception("ICON does not support other processing / compensation granularity")

    # 若要对整个阵列/OU列累加结果进行补偿，需要先对输出结果以OU为粒度累加一次

    # mirrored 需要先计算出转置后阵列输出电流，与正常计算输出电流的平均，之后才可以进行补偿
    '''
    if phyArrParams.IRdropCompensator == ir_drop_compensate_scheme.mirrored:
        # out [bsize, numX*num_ou_X, true_outsize]
        # reshape到OU粒度
        out_pos = out_pos.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                  phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
        out_neg = out_neg.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                  phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])
        # 对输入数据进行转置
        mirrored_in_b = in_b.flip([2])
        mirrored_g_b_pos_novar = g_b_pos_novar.reshape(numX, phyArrParams.arrRowSize, numY,
                                                       phyArrParams.arrColSize) \
            .flip([1, 3]) \
            .reshape(-1, phyArrParams.arrRowSize, true_outsize)
        mirrored_g_b_neg_novar = g_b_neg_novar.reshape(numX, phyArrParams.arrRowSize, numY,
                                                       phyArrParams.arrColSize) \
            .flip([1, 3]) \
            .reshape(-1, phyArrParams.arrRowSize, true_outsize)

        mirrored_g_b_pos = add_variation(mirrored_g_b_pos_novar, sigma=phyArrParams.sigma,
                                         rand_gen=variations.rand_gen_mirrored)
        mirrored_g_b_neg = add_variation(mirrored_g_b_neg_novar, sigma=phyArrParams.sigma,
                                         rand_gen=variations.rand_gen_mirrored)

        # mirrored_out_pos = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(mirrored_in_b, mirrored_g_b_pos, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold)
        # mirrored_out_neg = phyArrParams.cpp_ir_drop_accelerate.ir_gs_ou(mirrored_in_b, mirrored_g_b_neg, bsize, phyArrParams.arrRowSize, phyArrParams.arrColSize, phyArrParams.OUSize[0], phyArrParams.OUSize[1], true_insize//phyArrParams.arrRowSize, true_outsize//phyArrParams.arrColSize, phyArrParams.ir_drop_iter_time, 1/phyArrParams.r_wire, phyArrParams.ir_drop_GS_method_beta_scale, phyArrParams.ir_drop_less_than_thresholdBreak_enable, phyArrParams.ir_drop_less_than_threshold)
        # mirrored_out_pos = ir_drop_OU(mirrored_in_b, mirrored_g_b_pos, bsize, numX, numY)
        # mirrored_out_neg = ir_drop_OU(mirrored_in_b, mirrored_g_b_neg, bsize, numX, numY)
        mirrored_out_pos, mirrored_out_neg = IRdrop_matmul(mirrored_in_b, mirrored_g_b_pos, mirrored_g_b_neg, arrType)

        # reshape到OU粒度
        mirrored_out_pos = mirrored_out_pos.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                                    phyArrParams.arrOUnum[1],
                                                    phyArrParams.OUSize[1])
        mirrored_out_neg = mirrored_out_neg.reshape(bsize, numX, phyArrParams.arrOUnum[0], numY,
                                                    phyArrParams.arrOUnum[1],
                                                    phyArrParams.OUSize[1])

        # 对输出数据进行转置
        mirrored_out_pos = mirrored_out_pos.flip([2, 4, 5])
        mirrored_out_neg = mirrored_out_neg.flip([2, 4, 5])

        mirrored_out_pos = mirrored_out_pos.div(phyArrParams.adcOUuseI).mul(
            2 ** adcBits - 1).round().mul(
            phyArrParams.OUSize[0] * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                    2 ** adcBits - 1))
        mirrored_out_neg = mirrored_out_neg.div(phyArrParams.adcOUuseI).mul(
            2 ** adcBits - 1).round().mul(
            phyArrParams.OUSize[0] * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                    2 ** adcBits - 1))

        out_pos_ = out_pos + mirrored_out_pos
        out_neg_ = out_neg + mirrored_out_neg

        # TODO: rewrite mirrored scheme & make it simpler
        out_pos_ = mirrored_compensator(out_pos_, inv_, g_block_pos_novar, bsize, numX, numY,
                                        r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)
        out_neg_ = mirrored_compensator(out_neg_, inv_, g_block_neg_novar, bsize, numX, numY,
                                        r_wire=phyArrParams.r_wire, r_load=phyArrParams.r_load)

        out_pos = out_pos_ / 2
        out_neg = out_neg_ / 2

        # reshape回去
        out_pos = out_pos.reshape(bsize, -1, true_outsize)
        out_neg = out_neg.reshape(bsize, -1, true_outsize)
    '''

    # 后处理，包括减掉参考列等
    if arrType == phyArrMode.PN:
        if phyArrParams.useOUSize == False:
            out_pos = out_pos.sum(0 if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop else 1)[:,
                      0:outsize].mul(
                phyArrParams.arrRowSize * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                        2 ** adcBits - 1))  # out is [bsize, outsize]
            out_neg = out_neg.sum(0 if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop else 1)[:,
                      0:outsize].mul(
                phyArrParams.arrRowSize * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                        2 ** adcBits - 1))  # out is [bsize, outsize]
        else:
            out_pos = out_pos.sum(0 if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop else 1)[:,
                      0:outsize].mul(
                phyArrParams.OUSize[0] * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                        2 ** adcBits - 1))  # out is [bsize, outsize]
            out_neg = out_neg.sum(0 if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop else 1)[:,
                      0:outsize].mul(
                phyArrParams.OUSize[0] * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                        2 ** adcBits - 1))  # out is [bsize, outsize]
        out = out_pos - out_neg
    else:
        if phyArrParams.useOUSize == False:
            if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
                out = out.reshape(numX, bsize, numY, -1)
            else:
                out = out.reshape(bsize, numX, numY, -1)
            zeros_col = out[..., phyArrParams.arrColSize:phyArrParams.arrColSize + 1]
            out = out - zeros_col
            if j == 0:
                # 只有最高位需要剪掉参考列，其余不需要
                ones_col = out[..., phyArrParams.arrColSize + 1:phyArrParams.arrColSize + 2]
                out = out - ones_col.mul(plus).round()
            out = out[..., 0:phyArrParams.arrColSize].sum(
                0 if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop else 1).reshape(bsize, -1)[:,
                  0:outsize]
            out = out.mul(
                phyArrParams.arrRowSize * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                        2 ** adcBits - 1))
        else:
            if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop:
                zero_index, one_index = -2, -1
                out = out.reshape(-1, bsize, numY, phyArrParams.arrColSize + 2)
            else:
                zero_index, one_index = -phyArrParams.OUSize[1], -phyArrParams.OUSize[1] + 1
                out = out.reshape(bsize, -1, numY, phyArrParams.arrColSize + phyArrParams.OUSize[1])
            zeros_col = out[..., zero_index:zero_index + 1]
            out = out - zeros_col
            if j == 0:
                if one_index + 1 == 0:
                    ones_col = out[..., one_index:]
                else:
                    ones_col = out[..., one_index:one_index + 1]
                out = out - ones_col.mul(plus).round()
            out = out[..., 0:phyArrParams.arrColSize].sum(
                0 if phyArrParams.IRdropMode == ir_drop_mode.no_ir_drop else 1).reshape(bsize, -1)[:,
                  0:outsize]
            out = out.mul(phyArrParams.OUSize[0] * (2 ** dacBits - 1) * (2 ** phyArrParams.cellBits - 1) / (
                    2 ** adcBits - 1))

    return out

def float64_fifo(data: ndarray, dim: int):
    nonzero_pos = data != 0

    mant = np.bitwise_and(data, (1 << 52) - 1)
    mant[nonzero_pos] = np.bitwise_or(mant[nonzero_pos], 1 << 52)  # add hidden leading 1 if data is not 0

    expo = np.bitwise_and(data.__rshift__(52), (1 << 11) - 1)

    nonzero_expo = expo[nonzero_pos]
    expo_max = nonzero_expo.max().item()
    expo_min = nonzero_expo.min().item()
    expo_range = expo_max - expo_min

    out_len = expo_range + 52 + 1

    out = np.zeros((dim, out_len))
    start_idx = expo.copy()
    start_idx[nonzero_pos] = expo_max - start_idx[nonzero_pos]
    data_input = mant.copy()

    for i in range(52 + 1):
        out[np.arange(dim), np.squeeze(start_idx) + i] = np.squeeze(np.bitwise_and(data_input, (1 << 52)).__rshift__(52))
        data_input = data_input.__lshift__(1)

    return out, out_len, expo_max - 1023

# splitArr matmul for scientific computing
def splitArr_matmul_f(bias: ndarray, matrix_info: dict, dim: int, adcBits: int, arrType: phyArrMode, device) -> Tensor:
    # dacBits = 1, inputBits = ???, weightBits = ???, outputBits = ???, adcBits = ???, cellBits = 1
    dacBits = 1
    cellBits = 1
    # inputBits = 1

    input_dim = output_dim = dim # square matrix
    row_size = phyArrParams.arrRowSize
    col_size = phyArrParams.arrColSize

    output = np.zeros(output_dim)
    # 比较简单的矩阵/向量划分策略
    # 由于矩阵以及bias向量较为稀疏，因此将bias以及matrix分块进行计算，当前分块大小为arrRowSize*1
    #     | b1 |        | M11 0   M12 0   0   M13 |      | b1*M11+b2*M31+b3*M61 |
    #     | 0  |        | 0   M21 0   0   0   0   |      | 0                    |
    # b = | b2 |, M.T = | M31 0   M32 M33 0   0   |, x = | B1*M12+b2*M32        |
    #     | 0  |        | 0   0   M41 M42 0   M43 |      | b2*M33+b3*M62        |
    #     | 0  |        | 0   0   0   0   M51 0   |      | 0                    |
    #     | b3 |        | M61 0   0   M62 0   M63 |      | B1*M13+b2*M43+b3*M63 |
    # 其中M以dict形式存储，{xbr: {c: ndarray}}, 例如M41 -> {3: {2: M41}}
    # TODO: 优化矩阵向量乘法计算性能 + debug
    for keys1 in matrix_info.keys():
        current_input_dim = min((keys1+1)*row_size, input_dim) - keys1*row_size
        bias_block = bias[keys1*row_size:keys1*row_size+current_input_dim]
        if (bias_block > 0).any():
            # 存在正数输入
            bias_pos = bias_block.copy()
            bias_pos[bias_pos < 0] = 0
            pos_input, pos_input_len, bias_pos_expo_max = float64_fifo(bias_pos, current_input_dim)

            if debug.detailed:
                print(bias_pos.view('float64'))
                print(pos_input)
                print(pos_input_len)
                print(bias_pos_expo_max)
            pos_input = torch.transpose(Tensor(pos_input).to(device), 0, 1)
        else:
            pos_input_len = 0
        if (bias_block < 0).any():
            # 存在负数输入
            bias_neg = bias_block.copy()
            bias_neg[bias_neg > 0] = 0

            neg_input, neg_input_len, bias_neg_expo_max = float64_fifo(bias_neg, current_input_dim)

            if debug.detailed:
                print(bias_neg.view('float64'))
                print(neg_input)
                print(neg_input_len)
                print(bias_neg_expo_max)
            neg_input = torch.transpose(Tensor(neg_input).to(device), 0, 1)
        else:
            neg_input_len = 0
        if pos_input_len != 0 or neg_input_len != 0:
            for keys2 in matrix_info[keys1]:
                pos_array, pos_array_len, matrix_pos_expo_max, neg_array, neg_array_len, matrix_neg_expo_max = matrix_info[keys1][keys2]
                if debug.detailed:
                    print(pos_array)
                    print(pos_array_len)
                    print(matrix_pos_expo_max)
                    print(neg_array)
                    print(neg_array_len)
                    print(matrix_neg_expo_max)

                pos_input_lshift = (2.0**(-np.arange(pos_input_len))).reshape(-1, 1)
                neg_input_lshift = (2.0**(-np.arange(neg_input_len))).reshape(-1, 1)
                pos_matrix_lshift = 2.0**(-np.arange(pos_array_len))
                neg_matrix_lshift = 2.0**(-np.arange(neg_array_len))
                if pos_input_len != 0 and pos_array_len != 0:
                    if bias_pos_expo_max + matrix_pos_expo_max > 100:
                        print(bias_pos, pos_array)
                    out = matmul(pos_input, pos_array, None, 1, cfs.input_bit_width, dacBits,
                                   adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                                   float_type=torch.float64)
                    # out = matmul_f(pos_input, pos_array, pos_input_len, dacBits,
                    #                adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                    #                float_type=torch.float64)
                    out = out.cpu().numpy().astype(np.float64)
                    out = out * pos_input_lshift * pos_matrix_lshift
                    output[keys2] += out.sum() * (2.0 ** (bias_pos_expo_max + matrix_pos_expo_max))
                if neg_input_len != 0 and pos_array_len != 0:
                    out = matmul(neg_input, pos_array, None, 1, cfs.input_bit_width, dacBits,
                                 adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                                 float_type=torch.float64)
                    # out = matmul_f(neg_input, pos_array, pos_input_len, dacBits,
                    #                adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                    #                float_type=torch.float64)
                    out = out.cpu().numpy().astype(np.float64)
                    out = out * neg_input_lshift * pos_matrix_lshift
                    output[keys2] += -1 * out.sum() * (2.0 ** (bias_neg_expo_max + matrix_pos_expo_max))
                if pos_input_len != 0 and neg_array_len != 0:
                    out = matmul(pos_input, neg_array, None, 1, cfs.input_bit_width, dacBits,
                                 adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                                 float_type=torch.float64)
                    # out = matmul_f(pos_input, neg_array, pos_input_len, dacBits,
                    #                adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                    #                float_type=torch.float64)
                    out = out.cpu().numpy().astype(np.float64)
                    out = out * pos_input_lshift * neg_matrix_lshift
                    output[keys2] += -1 * out.sum() * (2.0 ** (bias_pos_expo_max + matrix_neg_expo_max))
                if neg_input_len != 0 and neg_array_len != 0:
                    out = matmul(neg_input, neg_array, None, 1, cfs.input_bit_width, dacBits,
                                 adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                                 float_type=torch.float64)
                    # out = matmul_f(neg_input, neg_array, pos_input_len, dacBits,
                    #                adcBits, outputBits=cfs.output_bit_width, weightBits=pos_array_len, arrType=arrType,
                    #                float_type=torch.float64)
                    out = out.cpu().numpy().astype(np.float64)
                    out = out * neg_input_lshift * neg_matrix_lshift
                    output[keys2] += out.sum() * (2.0 ** (bias_neg_expo_max + matrix_neg_expo_max))

    return Tensor(output).to(device)

# splitArr matmul for neural network forwarding
def splitArr_matmul_nn(input: Tensor, w: Tensor, inputBits: int, dacBits: int, adcBits: int, \
                    outputBits: int, weightBits: int, arrType: phyArrMode, float_type: torch.dtype,
                    bnn_first_layer_flag: bool = False) -> Tensor:
    with torch.no_grad():
        bsize = input.size(0)
        outsize = w.size(1)
        output = w.new_zeros((bsize, outsize))
        cellBits = phyArrParams.cellBits

        if not phyArrParams.useBNN or bnn_first_layer_flag:
            fixed_point_input, fixed_point_input_params = quantization_tensor(input, inputBits, TensorType.Normal)
            s1 = fixed_point_input_params[0].item()
        else:
            fixed_point_input, fixed_point_input_params = binarize_tensor(input, TensorType.Normal)
            s1 = fixed_point_input_params[0].item()

        if phyArrParams.inputVmode == input_v_mode.neg_pulse:
            neg_index = fixed_point_input < 0
            neg_data_abs = fixed_point_input.abs()[neg_index]
        elif dacBits != 1:  # 如果是补码模式，但是 dac不是1bit，这样是不能计算的，所以报错
            raise Exception("[Error] Complement code input mode, but dac is not 1 bit. dacbits = {}".format(dacBits))

        if arrType == phyArrMode.PN:
            if not phyArrParams.useBNN:
                fixed_point_w, fixed_point_w_params = quantization_tensor(w, weightBits, TensorType.PN)
                s2 = fixed_point_w_params[0].item()
            else:
                fixed_point_w, fixed_point_w_params = binarize_tensor(w, TensorType.PN)
                s2 = fixed_point_w_params[0].item()

            fixed_point_w_pos = fixed_point_w.clone()
            fixed_point_w_pos[fixed_point_w_pos < 0] = 0
            fixed_point_w_neg = fixed_point_w.clone()
            fixed_point_w_neg[fixed_point_w_neg > 0] = 0
            fixed_point_w_neg.abs_()

            mask_cell = (1 << cellBits) - 1
            mask_in = (1 << dacBits) - 1
            for i in range(0, inputBits, dacBits):

                in_b = fixed_point_input.__rshift__(i).bitwise_and(mask_in)
                if phyArrParams.inputVmode == input_v_mode.neg_pulse:
                    in_b[neg_index] = -(neg_data_abs.__rshift__(i).bitwise_and(mask_in))

                # if insize % phyArrParams.arrRowSize != 0:  # 补全输入电压到 物理阵列行数的倍数
                #     temp_size = phyArrParams.arrRowSize - insize % phyArrParams.arrRowSize
                #     in_b = torch.cat((in_b, in_b.new_zeros(bsize, temp_size)), 1)

                # in_b = in_b.to(float_type).mul(inputV)
                # in_b = in_b.reshape(bsize, -1, phyArrParams.arrRowSize)

                for j in range(0, weightBits, cellBits):
                    # 将权值矩阵中数据的各个比特拆分到不同阵列上
                    g_b_pos = fixed_point_w_pos.__rshift__(j).bitwise_and(mask_cell)
                    g_b_neg = fixed_point_w_neg.__rshift__(j).bitwise_and(mask_cell)

                    out = matmul(in_b, g_b_pos, g_b_neg, None, inputBits, dacBits, adcBits, outputBits, weightBits, arrType, float_type)

                    # if i != inputBits - 1 or phyArrParams.inputVmode != input_v_mode.complement_code:
                    #     output = output.add((out_pos - out_neg).mul(2 ** (i + j + s1 + s2)))
                    # else:
                    #     output = output.sub((out_pos - out_neg).mul(2 ** (i + j + s1 + s2)))

                    if i != inputBits - 1 or phyArrParams.inputVmode != input_v_mode.complement_code:
                        output = output.add(out.mul(2 ** (i + j + s1 + s2)))
                    else:
                        output = output.sub(out.mul(2 ** (i + j + s1 + s2)))
        elif arrType == phyArrMode.Ref:

            if not phyArrParams.useBNN:
                fixed_point_w, fixed_point_w_params = quantization_tensor(w, weightBits, TensorType.Normal)
                s2 = fixed_point_w_params[0].item()
            else:
                weightBits = 2
                fixed_point_w, fixed_point_w_params = binarize_tensor(w, TensorType.Normal)
                s2 = fixed_point_w_params[0].item()

            plus = 2 ** (weightBits - 1)
            maximum = 2 ** (weightBits) - 1
            fixed_point_w = fixed_point_w.add(plus)  # ref column mode, add bias.
            fixed_point_w[fixed_point_w > maximum] = maximum  # if max than 2*plus-1, reset to 2*plus-1

            mask_cell = (1 << cellBits) - 1
            mask_in = (1 << dacBits) - 1
            for i in range(0, inputBits, dacBits):
                in_b = fixed_point_input.__rshift__(i).bitwise_and(mask_in)
                if phyArrParams.inputVmode == input_v_mode.neg_pulse:
                    in_b[neg_index] = -(neg_data_abs.__rshift__(i).bitwise_and(mask_in))

                for j in range(0, weightBits, cellBits):
                    g_b = fixed_point_w.__rshift__(j).bitwise_and(mask_cell)

                    out = matmul(in_b, g_b, None, j, inputBits, dacBits, adcBits, outputBits, weightBits, arrType, float_type)

                    if i != inputBits - 1 or phyArrParams.inputVmode != input_v_mode.complement_code:
                        output = output.add(out.mul(2 ** (i + j + s1 + s2)))
                    else:
                        output = output.sub(out.mul(2 ** (i + j + s1 + s2)))

        return output
