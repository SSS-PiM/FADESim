#!/usr/bin/python3
from sys import argv
import re

def get_Iout_data(data_lines):
    ok = False
    for line in data_lines:
        line = line.strip().split()
        if len(line)>1 and line[0]=="iout":
            ok = True
            continue
        if ok:
            ret_data = [float(x) for x in line]
            return ret_data

def get_scale(k):
    if k=='F':
        x = 1e-15
    elif k=='M':
        x = 1e-3
    elif k=='P':
        x = 1e-12
    elif k=='K':
        x = 1e3
    elif k=='N':
        x = 1e-9
    elif k=='X':
        x = 1e6
    elif k=='U':
        x = 1e-6
    elif k=='G':
        x = 1e9
    else:
        x = 1
    return x

f=[]

if len(argv) == 1:
    f.append("fastmode_scichina.out")
    f.append("fastmode.out")
elif len(argv) == 2:
    f.append("fastmode_scichina.out")
    f.append(argv[1])
else:
    f.append(argv[1])
    f.append(argv[2])

if f[0]=="hspice.out":
    f[0], f[1] = f[1], f[0]

print("Compre {} with {}".format(f[0], f[1]))
exact_is_hspice = False
if f[1]=="hspice.out":
    exact_is_hspice = True
check_file = open(f[0], 'r')
exact_file = open(f[1], 'r')

if not exact_is_hspice:
    check_data = check_file.readlines()
    exact_data = exact_file.readlines()

    check_data = get_Iout_data(check_data)
    exact_data = get_Iout_data(exact_data)
    if len(check_data)!=len(exact_data):
        print("error: length not equal")
        exit(1)
else:
    check_data = check_file.readlines()
    check_data = get_Iout_data(check_data)

    exact_data = exact_file.read()
    pattern = re.compile(r"(volt|time)\s+current\s+v\d{1,4}\s+\d\.\d*\s+(\d*(\.){0,1}\d*)([A-Za-z]*)")
    exact_data_tmp = pattern.findall(exact_data)
    exact_data= []
    for line in exact_data_tmp:
        # print(line)
        unit = get_scale(line[-1].upper())
        exact_data.append(float(line[-3])*unit)
    
    if len(check_data)!=len(exact_data):
        print("error: length not equal, len1 = {}, len2={}".format(len(check_data), len(exact_data)))
        exit(1)


diff_data = [0 if abs(x[1])==0 else abs(x[0]-x[1])/abs(x[1]) for x in zip(check_data, exact_data)]

# print(diff_data)
print("Iout max relative error rate = {:f}%".format(max(diff_data)*100))
print("Iout avg relative error rate = {:f}%".format(sum(diff_data)/len(diff_data)*100))
print("################\n\n")

