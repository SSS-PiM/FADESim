#!/usr/bin/python3
from sys import argv 
import numpy as np
import os
import argparse
import re

def generate_test_config(input, R_state, M, N, r_wire, states, fastsolve_time, args):
    f = open("py_generated_config", "w")
    f.write("topString .title twoDArray\n")
    f.write("topString .hdl 'reram_mod.va' reram_mod\n")
    f.write("bottomString .tran 1 1\n")
    f.write("bottomString .end\n")
    f.write("arraySize %d %d\n"%(M, N))
    f.write("selector no\n")
    f.writelines(["line_resistance %f\n"%r_wire, "setUseLine left -1\n", "setUseLine down -1\n", "setLineV down -1 0\n"])
    if args.I:
        f.write("senseBitlineI down -1\n")

    if args.V or args.CV or args.DV or args.UV:
        f.write("senseCellV -1 -1\n")

    os.system("echo \"%d %d %f %d %d %d %d %d %f %f\" > gen_table_config"%(M, N, r_wire, 2, 2, 1, fastsolve_time, 2, states[0], states[1]))

    ideal_Iout = [0]*N
    for i in range(M):
        for j in range(N):
            ideal_Iout[j] += R_state[i][j]*input[i]
    
    for j in range(N):
        ideal_Iout[j] = ideal_Iout[j]/states[1]+(M-ideal_Iout[j])/states[0]
    
    file = open("ir_drop_free.out", "w")
    file.write(" iout = \n")
    file.writelines([str(data)+" " for data in ideal_Iout])

    f.write("setLineV left %d %f\n"%(-1, input[0]))
    for i, data in enumerate(input):
        if data!=input[0]:
            f.write("setLineV left %d %f\n"%(i, data))

    f.write("setCellR -1 -1 %d\n"%R_state[0][0])
    for i in range(len(R_state)):
        for j in range(len(R_state[i])):
            if R_state[i][j]!=R_state[0][0]:
                f.write("setCellR %d %d %d\n"%(i, j, R_state[i][j])) 

    f.write("build out\n")
    strStates = [(" "+str(x)) for x in states]
    strStates[-1]+='\n'
    f.write("fastmode yes %d "%len(states)+" ".join(strStates))
    f.write("fastsolve %d 0\n"%(fastsolve_time))
    f.close()

def change_reram_modva(states):
    file = open("reram_mod.va", "r")
    data = file.read()
    file.close()

    pattern = re.compile(r"lowState\s*=\s*.*;")
    data = pattern.sub("lowState = %f;"%states[1], data)

    pattern = re.compile(r"highState\s*=\s*.*;")
    data = pattern.sub("highState = %f;"%states[0], data)

    file = open("reram_mod.va", "w")
    file.write(data)
    file.close()

def test(exec_path, log_name, iter_time, args):
    
    if iter_time==1:
        if args.FFF<=0:
            os.system("hspice out > hspice.out")
        else:
            #这个模式下，我将用fastmode 迭代fff次，来作为正确的结果，生成电压等输出，方便后面进行对比
            file = open("py_generated_config", "r")
            data = file.readlines()
            file.close()
            olddata = data.copy()
            for i, d in enumerate(data):
                row = d.strip().split()
                if row[0]=="fastsolve":
                    data[i] = "fastsolve %d 0\n"%args.FFF
                else:
                    data[i] = data[i]+"\n"
                olddata[i] = olddata[i]+"\n"
            file = open("py_generated_config", "w")
            file.writelines(data)
            file.close()
            os.system(exec_path+"sim "+"py_generated_config")
            os.system("mv fastmode.out fastmode.std.out")

            file = open("py_generated_config", "w")
            file.writelines(olddata)
            file.close()

    os.system(exec_path+"sim "+"py_generated_config > %s"%log_name)

            

    # os.system("echo \"\n---------------------------\nCompared to hspice reuslts\nIR2S fastmode error=\n\" >> log_%s.log\n\n"%log_name)
    # os.system("./check_iout.py fastmode.out hspice.out >> log_%s.log\n\n"%log_name)

def checkI(log_name, number, args):
    if args.FFF<=0:
        os.system("./check_iout.py fastmode.out hspice.out >> %s"%log_name)
    else:
        os.system("./check_iout.py fastmode.out fastmode.std.out >> %s"%log_name)
    file = open(log_name, "r")
    data = file.read()
    file.close()

    if args.avg:
        pattern = re.compile(r"Iout avg relative error rate = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
    else:
        pattern = re.compile(r"Iout max relative error rate = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
    res = pattern.findall(data)
    if len(res)==0:
        print("Error: do not find pattern!")
        exit(-1)
    relative_err = float(res[-1][0])
    if relative_err<args.max_err:
        return True
    return False


def checkV(log_name, number, args, pass_ok):

    if not pass_ok:
        if args.FFF<=0:
            os.system("./check cell >> %s"%log_name)
            os.system("./check cmp >> %s"%log_name)
        else:
            os.system("./check cmp fastmode.out fastmode.std.out >> %s"%log_name)
    file = open(log_name, "r")
    data = file.read()
    file.close()

    if (number&1)!=0:
        if args.avg:
            pattern = re.compile(r"avg relative error in all voltage = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        else:
            pattern = re.compile(r"max relative error in all voltage = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        res = pattern.findall(data)
        if len(res)==0:
            print("Error: do not find pattern!")
            exit(-1)
        relative_err = float(res[-1][0])
        if relative_err<args.max_err:
            return True
        return False
    
    if (number&4)!=0:
        if args.avg:
            pattern = re.compile(r"avg relative cell voltage error = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        else:
            pattern = re.compile(r"max relative cell voltage error = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        res = pattern.findall(data)
        if len(res)==0:
            print("Error: do not find pattern!")
            exit(-1)
        relative_err = float(res[-1][0])
        if relative_err<args.max_err:
            return True
        return False
    
    if (number&8)!=0:
        if args.avg:
            pattern = re.compile(r"avg relative error in wordine voltage = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        else:
            pattern = re.compile(r"max relative error in wordine voltage = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        res = pattern.findall(data)
        if len(res)==0:
            print("Error: do not find pattern!")
            exit(-1)
        relative_err = float(res[-1][0])
        if relative_err<args.max_err:
            return True
        return False

    if (number&16)!=0:
        if args.avg:
            pattern = re.compile(r"avg relative error in bitline voltage = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        else:
            pattern = re.compile(r"max relative error in bitline voltage = ([+-]?[\d]+([\.][\d]+)?([Ee][+-]?[\d]+)?\d*.?\d)%")
        res = pattern.findall(data)
        if len(res)==0:
            print("Error: do not find pattern!")
            exit(-1)
        relative_err = float(res[-1][0])
        if relative_err<args.max_err:
            return True
        return False
    
    print("Error: not support number")
    exit(-1)

parser = argparse.ArgumentParser()

parser.add_argument("-M", help = "crx row size", type=int, default=128)
parser.add_argument("-N", help = "crx column size", type=int, default=128)
parser.add_argument("-r_wire", help = "crx wire resistance", type=float, default=2.93)
parser.add_argument("-max_err", help = "to achieve max relative error%% < max_err%%", type=float, default=0.01)
parser.add_argument("-states", nargs="+", help = "cell resistance states", type=float, default=[1e7, 1e5])
parser.add_argument("-V", help = "max relative error is judged according to all up and down plane voltage", action="store_true")
parser.add_argument("-I", help = "max relative error is judged according to the bitline current", action="store_true")
parser.add_argument("-CV", help = "max relative error is judged according to the the cell (up-down) voltage", action="store_true")
parser.add_argument("-UV", help = "max relative error is judged according to the the up plane voltage", action="store_true")
parser.add_argument("-DV", help = "max relative error is judged according to the the down plane voltage", action="store_true")
parser.add_argument("-avg", help = "use avg relative error <max_err, rather than max relative error", action="store_true")
parser.add_argument("-check", help = "check whether the iter_time is real ok", type=int, default=0)
parser.add_argument("-iter", help = "alread know iter time, only need to check", type=int, default=-1)
parser.add_argument("-FFF", help = "use fastmode x iterations to generate the standord output", type=int, default=0)


args = parser.parse_args()
M, N = args.M, args.N
r_wire = args.r_wire
cellStates = len(args.states)
states = args.states

need = 0
if args.V:
    need+=1
if args.I:
    need+=2
if args.CV:
    need+=4
if args.UV:
    need+=8
if args.DV:
    need+=16

print("crx size = (%d, %d), r_wire = %f, cellstates = %d,"%(M, N, r_wire, cellStates))
print("states = ", states)

if cellStates!=2:
    print("Error: now only support 2 states in a cell")
    exit(-1)

change_reram_modva(states)

input = [1]*M
R_state = [[1]*N for _ in range(M)]

exec_path = "../build/"

iter_time = 1
bk = 0



while args.iter<0:
    generate_test_config(input, R_state, M, N, r_wire, states, iter_time, args)
    log_name = "log_obtain_size%dx%d_r%.2f_%d.log"%(M, N, r_wire, iter_time)
    test(exec_path, log_name, iter_time, args)

    if iter_time==1:
        file = open(log_name+"time", "w")
    else:
        file = open(log_name+"time", "w+")

    pass_ok = False
    if args.V and (bk&1)==0:
        if checkV(log_name, 1, args, pass_ok):
            bk += 1
            file.write("need %d iter to ensure all voltage %s relative err< max_err (%f%%)\n"%(iter_time, "avg" if args.avg else "max", args.max_err))
        pass_ok = True

    if args.I and (bk&2)==0:
        if checkI(log_name, 2, args):
            bk += 2
            file.write("need %d iter to ensure bitline current %s relative err< max_err (%f%%)\n"%(iter_time, "avg" if args.avg else "max", args.max_err))

    if args.CV and (bk&4)==0:
        if checkV(log_name, 4, args, pass_ok):
            bk += 4
            file.write("need %d iter to ensure cell voltage %s relative err< max_err (%f%%)\n"%(iter_time, "avg" if args.avg else "max", args.max_err))
        pass_ok = True

    if args.UV and (bk&8)==0:
        if checkV(log_name, 8, args, pass_ok):
            bk += 8
            file.write("need %d iter to ensure up voltage %s relative err< max_err (%f%%)\n"%(iter_time, "avg" if args.avg else "max", args.max_err))
        pass_ok = True

    if args.DV and (bk&16)==0:
        if checkV(log_name, 16, args, pass_ok):
            bk += 16
            file.write("need %d iter to ensure down voltage %s relative err < max_err (%f%%)\n"%(iter_time, "avg" if args.avg else "max", args.max_err))
        pass_ok = True

    file.close()
    file = open(log_name+"time", "r")
    if file.read()=="":
        os.system("rm %s"%(log_name+"time"))
    if bk==need:
        break
    iter_time += 1



def myRand(in_R, i):
    input_0_percent = (i%10)/10
    R_0_percent = (i%30)/10
    in_R[0] = np.random.rand(args.M)
    in_R[0][in_R[0]<input_0_percent] = 0
    in_R[0][in_R[0]!=0] = 1

    in_R[1] = np.random.rand(args.M, args.N)
    in_R[1][in_R[1]<R_0_percent] = int(0)
    in_R[1][in_R[1]!=0] = int(1)



if args.iter<0:
    args.iter = iter_time

iter_time = args.iter
print("we need %d iter_time.\n"%args.iter)


input = [1]*M
R_state = [[1]*N for _ in range(M)]

if args.check>0:
    print("Get into iter time check function\n")
    for i in range(args.check):
        in_R = [input, R_state]
        myRand(in_R, i)
        print(in_R[1])
        generate_test_config(in_R[0], in_R[1], M, N, r_wire, states, iter_time, args)
        log_name = "log_check_size%dx%d_r%.2f_%d.log"%(M, N, r_wire, iter_time)
        test(exec_path, log_name, 1)
        pass_ok = False
        if args.V:
            if not checkV(log_name, 1, args, pass_ok):
                print("Error, this iter time is not ok")
            pass_ok = True
        if args.CV:
            if not checkV(log_name, 4, args, pass_ok):
                print("Error, this iter time is not ok")
            pass_ok = True
        if args.UV:
            if not checkV(log_name, 8, args, pass_ok):
                print("Error, this iter time is not ok")
            pass_ok = True
        if args.DV:
            if not checkV(log_name, 16, args, pass_ok):
                print("Error, this iter time is not ok")
            pass_ok = True
        if args.I:
            if not checkI(log_name, 2, args):
                print("Error, this iter time is not ok")