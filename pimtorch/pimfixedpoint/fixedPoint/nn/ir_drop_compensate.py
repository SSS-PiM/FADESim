import math
import torch
from torch import Tensor
from ctypes import sizeof
from .conductance_variation import add_variation
from .commonConst import variations, phyArrParams, GS_method, ir_drop_compensate_scheme, ConstForSplitArray as cfs, debug, memCMOS_method, phyArrMode

# OUadcUseI = phyArrParams.OURowSize*(phyArrParams.cellMaxConduct-phyArrParams.cellMinConduct)*phyArrParams.inputMaxVoltage

# output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
# inv [bsize, numX, #OURow, OURowSize]
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
def icon_compensator(output: Tensor, in_b: Tensor, G: Tensor, bsize: int, numX: int, numY: int, arrType,
                        r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load) -> Tensor:
    alpha = 1.0
    arrOUnum = phyArrParams.arrOUnum[1] if arrType == phyArrMode.PN else phyArrParams.arrOUnum[1] + 1
    # [#OUCol]
    r_left = Tensor([(phyArrParams.OUSize[1] * c + 1) * r_wire for c in range(arrOUnum)]).to(output.device)
    # [#OURow]
    r_down = Tensor([(phyArrParams.arrRowSize - (r + 1) * phyArrParams.OUSize[0] + 1) * r_wire for r in range(phyArrParams.arrOUnum[0])]).to(output.device)

    # 目前只支持补码输入方式 & dacbits = 1
    # [bsize, numX, #OURow]
    k = (inv/phyArrParams.inputMaxVoltage).sum(3)
    # [numX, #OURow, numY, #OUCol, OUColSize]
    G_block_ourow_summed = G_block.sum(2)

    delta = (output.sum(5) * r_left).unsqueeze(-1).repeat(1, 1, 1, 1, 1, phyArrParams.OUSize[1]) # [bsize, numX, #OURow, numY, #OUCol, OUColSize]
    delta = delta + output * (k * r_down).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, arrOUnum, phyArrParams.OUSize[1]) # [bsize, numX, numY, OUColSize] + [bsize, numX, numY, OUColSize] * [bsize, numX, numY, OUColSize]
    delta = delta * G_block_ourow_summed / phyArrParams.OUSize[0] # [bsize, numX, #OURow, numY, #OUCol, OUColSize] * [numX, #OURow, numY, #OUCol, OUColSize]

    return output + delta

# TODO: implement mirrored_compensator for entire array
# output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
# inv [bsize, numX, #OURow, OURowSize]
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
def mirrored_compensator(output: Tensor, inv: Tensor, G_block: Tensor, bsize: int, numX: int, numY: int,
                            r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load) -> Tensor:
    # TODO: debugging mirrored
    alpha = 1.0

    k = inv.sum(3)

    delta = 2 * phyArrParams.arrRowSize * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY,
                                                                                             phyArrParams.arrOUnum[1],
                                                                                             phyArrParams.OUSize[
                                                                                                 1]) * G_block.mul(
        G_block).sum(2) / phyArrParams.OUSize[0] * r_wire

    return output + delta

# output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
# inv [bsize, numX, #OURow, OURowSize]
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_output [bsize, numX, #OURow, numY, #OUCol, OUColSize]
def mirrored_OU_compensator(output: Tensor, inv: Tensor, G_block: Tensor, bsize: int, numX: int, numY: int,
                        r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load) -> Tensor:
    # TODO: debugging mirrored
    alpha = 1.0

    k = inv.sum(3)

    delta = 2 * phyArrParams.arrRowSize * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1]) * G_block.mul(G_block).sum(2) / phyArrParams.OUSize[0] * r_wire
    
    return output + delta

# inv [bsize, numX, #OURow, OURowSize]
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_inv [bsize, numX, #OURow, numY, #OUCol, OURowSize]
def memCMOS_pre_compensator(inv: Tensor, G_block: Tensor, bsize: int, numX: int, numY: int,
                        r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load):
    # TODO: debugging memCMOS_preOU_compensator
    # Vin_i' = Vin_i * (1 + r_left * (li/(Ron+r_down) + (OURowSize-li)/(Roff+r_down)))
    alpha = 1.0

    # [#OUCol]
    r_left = Tensor([(phyArrParams.OUSize[1] * c + 1) * r_wire for c in range(phyArrParams.arrOUnum[1])]).to(inv.device)
    # [#OURow]
    r_down = Tensor([(phyArrParams.arrRowSize - (r + 1) * phyArrParams.OUSize[0] + 1) * r_wire for r in range(phyArrParams.arrOUnum[0])]).to(inv.device)

    G_block = G_block.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])

    gamma = 1 + r_left.unsqueeze(-1).repeat(1, phyArrParams.OUSize[1]) * (1.0 / (1.0 / G_block + r_down.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, phyArrParams.OUSize[0], numY, phyArrParams.arrOUnum[0], phyArrParams.OUSize[1]))).sum(5) # [numX, OURowSize, numY]
    Gr = (gamma - 1) * memCMOS_method.C1

    if (Gr > phyArrParams.cellMaxConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance > {0}, will set cell to {0}".format(phyArrParams.cellMaxConduct))
        #########################################
        Gr[Gr > phyArrParams.cellMaxConduct] = phyArrParams.cellMaxConduct
    if (Gr < phyArrParams.cellMinConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance < {0}, will set cell to {0}".format(phyArrParams.cellMinConduct))
        #########################################
        Gr[Gr < phyArrParams.cellMinConduct] = phyArrParams.cellMinConduct

    Gr_varied = add_variation(Gr, sigma=0, rand_gen=variations.rand_gen_memCMOS)
    gamma = Gr_varied / phyArrParams.cellMinConduct + 1

    compensated_inv = torch.transpose(torch.transpose(inv.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, phyArrParams.arrOUnum[1]) * gamma, 2, 3), 3, 4)
    return compensated_inv # [bsize, numX, #OURow, numY, #OUCol, OURowSize]

# output_currents {batch_size, numX, #OURow, numY, #OUCol, OUColSize}
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_currents [bsize, numX, numY, OUColSize]
def memCMOS_post_compensator(output: Tensor, G_block: Tensor, bsize: int, numX: int, numY: int, r: int, c: int,
                            r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load):
    # TODO: debugging memCMOS_postOU_compensator
    # Iout_j' = Iout_j * (1 + r_down * (kj/(Ron+r_left) + (OUColSize-kj)/(Roff+r_left)))
    alpha = 1.0

    # [#OUCol]
    r_left = Tensor([(phyArrParams.OUSize[1] * c + 1) * r_wire for c in range(phyArrParams.arrOUnum[1])]).to(output.device)
    # [#OURow]
    r_down = Tensor([(phyArrParams.arrRowSize - (r + 1) * phyArrParams.OUSize[0] + 1) * r_wire for r in range(phyArrParams.arrOUnum[0])]).to(output.device)

    G_block = G_block.reshape(numX, phyArrParams.OUSize[0], numY, phyArrParams.OUSize[1])

    gamma = 1 + r_down * (1.0 / (1.0 / G_block + r_left)).sum(1) # [numX, numY, OUColSize]
    Gr = (gamma - 1) * memCMOS_method.C2

    if (Gr > phyArrParams.cellMaxConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance > {0}, will set cell to {0}".format(
                phyArrParams.cellMaxConduct))
        #########################################
        Gr[Gr > phyArrParams.cellMaxConduct] = phyArrParams.cellMaxConduct
    if (Gr < phyArrParams.cellMinConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance < {0}, will set cell to {0}".format(
                phyArrParams.cellMinConduct))
        #########################################
        Gr[Gr < phyArrParams.cellMinConduct] = phyArrParams.cellMinConduct

    Gr_varied = add_variation(Gr, sigma=0, rand_gen=variations.rand_gen_memCMOS)
    gamma = Gr_varied / phyArrParams.cellMinConduct + 1

    compensated_currents = output * gamma
    return compensated_currents # [bsize, numX, numY, OUColSize]

# inv [bsize, numX, #OURow, OURowSize]
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_inv [bsize, numX, #OURow, numY, #OUCol, OURowSize]
def memCMOS_preOU_compensator(inv: Tensor, G_block: Tensor, bsize: int, numX: int, numY: int,
                        r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load):
    # TODO: debugging memCMOS_preOU_compensator
    # Vin_i' = Vin_i * (1 + r_left * (li/(Ron+r_down) + (OURowSize-li)/(Roff+r_down)))
    alpha = 1.0

    # [#OUCol]
    r_left = Tensor([(phyArrParams.OUSize[1] * c + 1) * r_wire for c in range(phyArrParams.arrOUnum[1])]).to(inv.device)
    # [#OURow]
    r_down = Tensor([(phyArrParams.arrRowSize - (r + 1) * phyArrParams.OUSize[0] + 1) * r_wire for r in range(phyArrParams.arrOUnum[0])]).to(inv.device)

    G_block = G_block.reshape(numX, phyArrParams.arrOUnum[0], phyArrParams.OUSize[0], numY, phyArrParams.arrOUnum[1], phyArrParams.OUSize[1])

    gamma = 1 + r_left.unsqueeze(-1).repeat(1, phyArrParams.OUSize[1]) * (1.0 / (1.0 / G_block + r_down.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, phyArrParams.OUSize[0], numY, phyArrParams.arrOUnum[0], phyArrParams.OUSize[1]))).sum(5) # [numX, OURowSize, numY]
    Gr = (gamma - 1) * memCMOS_method.C1

    if (Gr > phyArrParams.cellMaxConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance > {0}, will set cell to {0}".format(phyArrParams.cellMaxConduct))
        #########################################
        Gr[Gr > phyArrParams.cellMaxConduct] = phyArrParams.cellMaxConduct
    if (Gr < phyArrParams.cellMinConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance < {0}, will set cell to {0}".format(phyArrParams.cellMinConduct))
        #########################################
        Gr[Gr < phyArrParams.cellMinConduct] = phyArrParams.cellMinConduct

    Gr_varied = add_variation(Gr, sigma=0, rand_gen=variations.rand_gen_memCMOS)
    gamma = Gr_varied / phyArrParams.cellMinConduct + 1

    compensated_inv = torch.transpose(torch.transpose(inv.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, numY, phyArrParams.arrOUnum[1]) * gamma, 2, 3), 3, 4)
    return compensated_inv # [bsize, numX, #OURow, numY, #OUCol, OURowSize]

# output_currents {batch_size, numX, #OURow, numY, #OUCol, OUColSize}
# G_block [numX, #OURow, OURowSize, numY, #OUCol, OUColSize]
# return compensated_currents [bsize, numX, numY, OUColSize]
def memCMOS_postOU_compensator(output: Tensor, G_block: Tensor, bsize: int, numX: int, numY: int, r: int, c: int,
                            r_wire: float = phyArrParams.r_wire, r_load: float = phyArrParams.r_load):
    # TODO: debugging memCMOS_postOU_compensator
    # Iout_j' = Iout_j * (1 + r_down * (kj/(Ron+r_left) + (OUColSize-kj)/(Roff+r_left)))
    alpha = 1.0

    # [#OUCol]
    r_left = Tensor([(phyArrParams.OUSize[1] * c + 1) * r_wire for c in range(phyArrParams.arrOUnum[1])]).to(output.device)
    # [#OURow]
    r_down = Tensor([(phyArrParams.arrRowSize - (r + 1) * phyArrParams.OUSize[0] + 1) * r_wire for r in range(phyArrParams.arrOUnum[0])]).to(output.device)

    G_block = G_block.reshape(numX, phyArrParams.OUSize[0], numY, phyArrParams.OUSize[1])

    gamma = 1 + r_down * (1.0 / (1.0 / G_block + r_left)).sum(1) # [numX, numY, OUColSize]
    Gr = (gamma - 1) * memCMOS_method.C2

    if (Gr > phyArrParams.cellMaxConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance > {0}, will set cell to {0}".format(
                phyArrParams.cellMaxConduct))
        #########################################
        Gr[Gr > phyArrParams.cellMaxConduct] = phyArrParams.cellMaxConduct
    if (Gr < phyArrParams.cellMinConduct).any():
        ######### debugging info ################
        if debug.detailed:
            print("Error calibrating memristor, cell conductance < {0}, will set cell to {0}".format(
                phyArrParams.cellMinConduct))
        #########################################
        Gr[Gr < phyArrParams.cellMinConduct] = phyArrParams.cellMinConduct

    Gr_varied = add_variation(Gr, sigma=0, rand_gen=variations.rand_gen_memCMOS)
    gamma = Gr_varied / phyArrParams.cellMinConduct + 1

    compensated_currents = output * gamma
    return compensated_currents # [bsize, numX, numY, OUColSize]

def IRdrop_deembedding_compensator():
    # TODO: implementing IRdrop_deembedding_compensator
    pass

def IRdrop_deembedding_OU_compensator():
    # TODO: implementing IRdrop_deembedding_compensator
    pass