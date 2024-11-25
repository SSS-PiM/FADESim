#!/usr/bin/python3
from sys import argv 
import numpy as np
import os
import argparse
import re
import torch
import time
import SCN

def printA(str, logfile):
    print(str)
    os.system(f"echo \"{str}\" > {logfile}")

def printB(str, logfile):
    print(str)
    os.system(f"echo \"{str}\" >> {logfile}")

def compare_with_spice(exec_path, number, args, input, R_state):
    os.system(exec_path+"sim "+"py_generated_config >> log_%s.log"%number)

    if args.scn_en:
        device = torch.device("cuda:" + str(args.cuda_use_num) if args.scn_use_cuda else "cpu")
        type = torch.float64 if args.scn_use_float64 else torch.float32
        #load model#
        scn_model = SCN.SCN(args.scn_layer_num, args.scn_ch_size, [args.M, args.N])
        model_paras_load = torch.load(args.scn_model_load_file)
        T1 = time.time()
        scn_model.load_state_dict(model_paras_load)


        w = torch.tensor(R_state)
        scn_model = scn_model.to(device, dtype=type)
        w = w.to(device, dtype=type)
        input = torch.tensor(input).view(1, args.M).to(device, dtype=type)

        cellNum = len(args.states)
        LRS = states[1]
        HRS = states[0]

        conductanceDelta = (1/LRS - 1/HRS)/(cellNum-1)

        w_e = scn_model(w)
        w_e = w_e.reshape(args.M, args.N).mul(conductanceDelta) + 1/HRS
        I_out = input.matmul(w_e)

        T2 = time.time()
        
        t = T2 - T1

        M = (args.M-3+2)/1 + 1  #  M = (X - K +2 *padding)/stride + 1
        mem_size = args.scn_layer_num*( (args.scn_ch_size**2)*9 + args.scn_ch_size*(M**2) ) * (8 if args.scn_use_float64 else 4)
        mem_size2 = ( args.scn_layer_num*( (args.scn_ch_size**2)*9) + args.scn_ch_size*(M**2) ) * (8 if args.scn_use_float64 else 4)

        os.system("echo \"{} {:.8f} seconds\" >> log_{}.log\n\n".format("scn test time = ", t, number))
        os.system("echo \"{} {} bytes, only one feature map is calculted then memory = {} bytes\" >> log_{}.log\n\n".format("scn test memory = ", mem_size, mem_size2, number))
        I_out_str = "iout = \n" + " ".join(map(str, I_out.view(-1).tolist()))
        os.system("echo \"%s\" > scn.out"%I_out_str)

    if not args.passh:
        os.system("hspice out > hspice.out")
        f = open("hspice.out", 'r')
        alldata = f.read()

        pattern = re.compile(r"total cpu time\s+(\d*\.*\d*)\s+seconds")
        fd = pattern.findall(alldata)
        if len(fd)!=1:
            raise Exception("hspice.out cpu time only one result should be found, now found {}".format(len(fd)))
        t = fd[0]
        os.system("echo \"{} {} seconds\" >> log_{}.log\n\n".format("hspice test time = ", t, number))

        pattern = re.compile(r"peak memory used\s+(\d*\.*\d*)\s+(\w+)") #peak memory used        298.43 megabytes
        fd = pattern.findall(alldata)
        if len(fd)!=1:
            raise Exception("hspice.out mem used only one result should be found, now found {}".format(len(fd)))
        t = " ".join(fd[0])
        os.system("echo \"{} {}\" >> log_{}.log\n\n".format("hspice test memory = ", t, number))
        f.close()
    
    if args.ngspice_en:
        os.system("ngspice -b out > ngspice.out")
        f = open("ngspice.out", 'r')
        alldata = f.read()
        pattern = re.compile(r"Total analysis time \(seconds\) = (\d*\.*\d*)")
        fd = pattern.findall(alldata)
        if len(fd)!=1:
            raise Exception("ngspicce.out cpu time only one result should be found, now found {}".format(len(fd)))
        t = fd[0]
        os.system("echo \"{} {} seconds\" >> log_{}.log\n\n".format("ngspice test time = ", t, number))

        pattern = re.compile(r"Maximum ngspice program size\s+=\s+(\d*\.*\d*)\s+(\w+)") #Maximum ngspice program size =  207.832 MB
        fd = pattern.findall(alldata)
        if len(fd)!=1:
            raise Exception("ngspice.out mem used only one result should be found, now found {}".format(len(fd)))
        t = " ".join(fd[0])
        os.system("echo \"{} {}\" >> log_{}.log\n\n".format("ngspice test memory = ", t, number))
        f.close()


    if args.ir2s_en:
        os.system("echo \"\n---------------------------\nCompared to hspice reuslts\nIR2S fastmode error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py fastmode.out hspice.out >> log_%s.log\n\n"%number)
    
    if args.gs_en:
        os.system("echo \"gs fastmode error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py fastmode_nodebasedGS.out hspice.out >> log_%s.log\n\n"%number)
        
    if args.em_en:
        os.system("echo \"sci china fastmode error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py fastmode_scichina.out hspice.out >> log_%s.log\n\n"%number)

    if args.pbia_en:
        os.system("echo \"pbia fastmode error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py fastmode_pbia.out hspice.out >> log_%s.log\n\n"%number)

    if args.aihwkit_en:
        os.system("echo \"aihwkit fastmode error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py fastmode_aihwkit.out hspice.out >> log_%s.log\n\n"%number)

    if args.neurosim_en:
        os.system("echo \"neurosim fastmode error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py fastmode_neurosim.out hspice.out >> log_%s.log"%number)
    
    if args.scn_en:
        os.system("echo \"scn error=\n\" >> log_%s.log\n\n"%number)
        os.system("./check_iout.py scn.out hspice.out >> log_%s.log"%number)

        
    os.system("echo \"ir drop free result error=\n\" >> log_%s.log\n\n"%number)
    os.system("./check_iout.py ir_drop_free.out hspice.out >> log_%s.log"%number)
   

def generate_config(input, R_state, M, N, r_wire, states, args):
    f = open("py_generated_config", "w")
    f.write("topString .title twoDArray\n")
    f.write("bottomString .end\n")
    f.write("arraySize %d %d\n"%(M, N))
    f.write("selector no\ndc\n")
    f.writelines(["line_resistance %f\n"%r_wire, "setUseLine left -1\n", "setUseLine down -1\n", "setLineV down -1 0\n", "senseBitlineI down -1\n"])

    ideal_Iout = [0]*N
    for i in range(M):
        for j in range(N):
            ideal_Iout[j] += R_state[i][j]*input[i]
    
    for j in range(N):
        ideal_Iout[j] = ideal_Iout[j]/states[1]+(M-ideal_Iout[j])/states[0]
    
    # file = open("ir_drop_free.out", "w")
    # file.write(" iout = \n")
    # file.writelines([str(data)+" " for data in ideal_Iout])

    f.write("setLineV left %d %f\n"%(-1, input[0]))
    for i, data in enumerate(input):
        if data!=input[0]:
            f.write("setLineV left %d %f\n"%(i, data))

    f.write("setCellR -1 -1 %d\n"%R_state[0][0])
    for i in range(len(R_state)):
        for j in range(len(R_state[i])):
            if R_state[i][j]!=R_state[0][0]:
                f.write("setCellR %d %d %d\n"%(i, j, R_state[i][j])) 

    
    strStates = [(" "+str(x)) for x in states]
    strStates[-1]+='\n'
    f.write("useRtypeReRAMCell yes\n")
    f.write("cellRstates %d "%len(states)+" ".join(strStates))
    if args.ir2s_en:
        f.write("fastmode yes %d "%len(states)+" ".join(strStates))
        f.write("fastsolve %d 0 %d %s\n"%(args.ir2s_iter, 0 if not args.ir2s_auto_break else 1, args.ir2s_break_th))
    if args.gs_en:
        f.write("nodebasedGSMethod "+" ".join(args.gs_params))
        f.write("\n")
    if args.em_en:
        f.write(f"ir_scichina {args.r_wire}\n")
    if args.pbia_en:
        f.write("ir_pbia "+" ".join(args.pbia_params))
        f.write("\n")
    if args.ir_free_en:
        f.write("ir_free\n")
        
    # f.write("ir_aihwkit 1 1\n")
    # f.write("ir_scichina %f\n"%r_wire)
    if args.neurosim_en:
        f.write("ir_neurosim\n")
    if args.aihwkit_en:
        f.write("ir_aihwkit "+" ".join(args.aihwkit_params))
        f.write("\n")
        
    if not args.passh:
        f.write("build out\n")
    f.close()

# def change_reram_modva(states):
#     file = open("reram_mod.va", "r")
#     data = file.read()
#     file.close()

#     pattern = re.compile(r"lowState\s*=\s*.*;")
#     data = pattern.sub("lowState = %f;"%states[1], data)

#     pattern = re.compile(r"highState\s*=\s*.*;")
#     data = pattern.sub("highState = %f;"%states[0], data)

#     file = open("reram_mod.va", "w")
#     file.write(data)
#     file.close()



parser = argparse.ArgumentParser()
parser.add_argument("-M", help = "crx row size", type=int, default=128)
parser.add_argument("-N", help = "crx col size", type=int, default=128)
parser.add_argument("-r_wire", help = "wire resistance", type=float, default=2.93)

## if enable different ir drop sim mode ##
parser.add_argument("-ir2s_en", help = "enable ir2s fastmode", action="store_true", default=False)
parser.add_argument("-gs_en", help = "enable gs method fastmode", action="store_true", default=False)
parser.add_argument("-em_en", help = "enable sci china (em) fastmode", action="store_true", default=False)
parser.add_argument("-pbia_en", help = "enable pbia fastmode", action="store_true", default=False)
parser.add_argument("-neurosim_en", help = "enable neurosim fastmode", action="store_true", default=False)
parser.add_argument("-aihwkit_en", help = "enable aihwkit fastmode", action="store_true", default=False)
parser.add_argument("-scn_en", help = "enable scn pred. mode", action="store_true", default=False)
parser.add_argument("-ngspice_en", help = "enable ngspice", action="store_true", default=False)
parser.add_argument("-ir_free_en", help = "enable ngspice", action="store_true", default=False)
## end ###

parser.add_argument("-ir2s_auto_break", help = "enable ir2s fastmode break", action="store_true", default=False)
parser.add_argument("-ir2s_break_th", help = "enable ir2s fastmode break", type=str, default="1e-9")
parser.add_argument("-ir2s_iter", help = "iterative times of ir2s fastmode", type=int, default=6)
parser.add_argument("-gs_params", help = "params of gs fastmode", type=str, nargs="+", default=["500", "1.97", "0", "no", "1e-9"])
parser.add_argument("-pbia_params", help = "params of pbia fastmode", type=str, nargs="+", default=["20", "no", "1e-9"])
parser.add_argument("-aihwkit_params", help = "params of aihwkit fastmode", type=str, nargs="+", default=["0.3", "1"])
parser.add_argument("-scn_layer_num", help = "layer num of scn model", type=int, default=7)
parser.add_argument("-scn_ch_size", help = "channel size of scn model", type=int, default=32)
parser.add_argument("-scn_model_load_file", help = "load scn model", type=str, default="SCNAlox_hfox_128x128_293_5w_checkp_lrs16900.pt")
parser.add_argument("-scn_use_cuda", help = "scn run in gpu", action="store_true", default=False)
parser.add_argument("-cuda_use_num", help = "cuda num", type=int, default=0)
parser.add_argument("-scn_use_float64", help = "scn run with float64", action="store_true", default=False)

parser.add_argument("-states", nargs="+", help = "cell resistance states", type=float, default=[74867, 16.9e3])
parser.add_argument("-num", help = "log number", type=int, default=0)
parser.add_argument("-rd", help = "strat random test rather than worst case", action="store_true", default=False)
parser.add_argument("-rd_input_params", help = "the percent of input 0 and 1", nargs="+", type=float, default=[0.5, 0.5])
parser.add_argument("-rd_state_params", help = "the percent of cell states 0 and 1", nargs="+", type=float, default=[0.5, 0.5])
parser.add_argument("-passh", help = "pass hspice test", action="store_true", default=False)


args = parser.parse_args()

M, N = args.M, args.N
r_wire = args.r_wire
cellStates = len(args.states)
states = args.states
ir2s_time = args.ir2s_iter
log_number = args.num

if args.rd:
    log_file_name = "size%dx%d_r%.2f_%d_rd"%(M, N, r_wire, log_number)
else:
    log_file_name = "size%dx%d_r%.2f_%d"%(M, N, r_wire, log_number)

log_file_name = "log_%s.log"%log_file_name

printA("crx size = (%d, %d), r_wire = %f, cellstates = %d, "%(M, N, r_wire, cellStates), log_file_name)
printB("states = {}".format(states), log_file_name)
printB("rd = {}".format(args.rd), log_file_name)

if args.rd:
    printB("input rd p = {}".format(" ".join(map(str, args.rd_input_params))), log_file_name)
    printB("state rd p = {}".format(" ".join(map(str, args.rd_state_params))), log_file_name)

if cellStates!=2:
    print("Error: now only support 2 states in a cell")
    exit(-1)
if args.gs_en:
    printB("gs params = "+" ".join(args.gs_params), log_file_name)
if args.pbia_en:
    printB("pbia params = "+" ".join(args.pbia_params), log_file_name)
if args.aihwkit_en:
    printB("aihwkit params = "+" ".join(args.aihwkit_params), log_file_name)
if args.scn_en:
    printB("scn parms, layer num = {}, ch_size = {}, load model file = {}, use float64 = {}".format(args.scn_layer_num, args.scn_ch_size, args.scn_model_load_file, "True" if args.scn_use_float64 else "False"), log_file_name)
# change_reram_modva(states)

input = [1]*M
R_state = [[1]*N for _ in range(M)]

#generate worst case ir drop
generate_config(input, R_state, M, N, r_wire, states, args)
exec_path = "../build/"

if not args.rd:
    compare_with_spice(exec_path, "size%dx%d_r%.2f_%d"%(M, N, r_wire, log_number), args, input, R_state)
else:
    input = np.random.rand(M)
    # input = [1]*M
    input[input>args.rd_input_params[0]] = 1
    input[input!=1] = 0
    R_state = np.random.rand(M, N)
    R_state[R_state>args.rd_state_params[0]] = 1
    R_state[R_state!=1] = 0
    generate_config(input, R_state, M, N, r_wire, states, args)
    compare_with_spice(exec_path, "size%dx%d_r%.2f_%d_rd"%(M, N, r_wire, log_number), args, input, R_state)

# for i in range(times-1):
#     #generate random case ir drop
#     input = np.random.rand(M)
#     input[input>0.5] = 1
#     input[input!=1] = 0
#     R_state = np.random.rand(M, N)
#     R_state[R_state>0.5] = 1
#     R_state[R_state!=1] = 0
#     generate_test_config(input, R_state, M, N, r_wire, states, fastsolve_time)
#     compare_with_spice(exec_path, i+1)




