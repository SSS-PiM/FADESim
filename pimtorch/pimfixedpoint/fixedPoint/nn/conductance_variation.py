import math
import numpy as np
import torch
from torch import Tensor

def add_variation(conductance, sigma=0, rand_gen=None):
    if isinstance(conductance, Tensor):
        # print(conductance.device)
        # R = Rtarget * exp(q), q~N(0, theta^2), so C = Ctarget/exp(q), here simga = theta^2
        return conductance / torch.exp(torch.randn(conductance.shape, generator=rand_gen) * sigma).to(conductance.device) # 1/(1/c * var) = c/var
    elif isinstance(conductance, float) or isinstance(conductance, int):
        return conductance / torch.exp(torch.randn([1], generator=rand_gen) * sigma).to(conductance.device)
    else:
        raise Exception("Conductance variation for data type {} is not implemented.".format(type(conductance)))


def nonlinear_IV(a, b, V):
    return b*np.sinh(a*V)

def nonlinear_IV_tensor(a: Tensor, b: Tensor, V: Tensor):
    return b*(a*V).sinh()

def nonlinear_R(a, b, V):
    return V/nonlinear_IV(a, b, V)

def nonlinear_R_tensor(a: Tensor, b: Tensor, V: Tensor):
    return V/nonlinear_IV_tensor(a, b, V)

def get_nonlinear_params_from_K2(nonlinear_V: float, R: float, k2: float):
    x = (k2+math.sqrt(k2*k2-4))/2
    a = 2*math.log(x)/nonlinear_V
    b = nonlinear_V/R/math.sinh(a*nonlinear_V)
    return a, b

# R = z*R_off + (1-z)*Ron, (Roff-Ron)*z + Ron = R
def get_nonlinear_ratio_z(r, r_min, r_max):
    return (r-r_min)/(r_max-r_min)

def nonlinear_R_tensor(a_on, b_on, a_off, b_off, z, V):
    return (1-z)*nonlinear_R_tensor(a_on, b_on, V)+z*nonlinear_R_tensor(a_off, b_off, V)
    

# the below is not used currently.
# def get_k2_from_G(G: Tensor, g_min, g_max, K2_off, K2_on):
#     return (G-g_min)/(g_max-g_min)*(K2_on-K2_off) + K2_off

# def get_nonlinear_params_from_K2_tensor(nonlinear_V: float, G: Tensor, k2: Tensor):
#     x = (k2+(k2*k2-4).sqrt())/2
#     a = 2*x.log()/nonlinear_V
#     b = nonlinear_V*G/(a*nonlinear_V).sinh()
#     return a, b

# def get_I_from_VG_consider_nonlinear(nonlinear_V, V, G, g_min, g_max, K2_off, K2_on):
#     k2 = get_k2_from_G(G, g_min, g_max, K2_off, K2_on)
#     a, b = get_nonlinear_params_from_K2_tensor(nonlinear_V, G, k2)
#     return nonlinear_IV_tensor(a, b, V)
    
# def get_R_from_VG_consider_nonlinear(nonlinear_V, V, G, g_min, g_max, K2_off, K2_on):
#     k2 = get_k2_from_G(G, g_min, g_max, K2_off, K2_on)
#     a, b = get_nonlinear_params_from_K2_tensor(nonlinear_V, G, k2)
#     return nonlinear_R_tensor(a, b, V)   

# # test_IV(nonlinear_Voltage, cellMinConduct, cellMaxConduct, cellBits, nonlinear_K2_on, nonlinear_K2_off)
# def test_IV(nonlinear_V0, g_min, g_max, cellBits, K2_on, K2_off):
#     g = torch.Tensor(np.linspace(g_min, g_max, 2**cellBits))
#     r0 = get_R_from_VG_consider_nonlinear(nonlinear_V0, nonlinear_V0, g, g_min, g_max, K2_off, K2_on)
#     r1 = get_R_from_VG_consider_nonlinear(nonlinear_V0, nonlinear_V0/2, g, g_min, g_max, K2_off, K2_on)



# torch.manual_seed(1)
# G = torch.Generator()
# G.seed()
# print(G.initial_seed())
# A = torch.ones([3, 4, 5])
# print(add_variation(10, sigma=1))
