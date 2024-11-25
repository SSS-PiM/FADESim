#!/usr/bin/python3
from sys import argv 
import numpy as np
import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-S", help = "Interval value for each step ", type=float, default=0.001)
parser.add_argument("-Min", help = "Range minimum", type=float, default=1.7)
parser.add_argument("-Max", help = "Range maximum", type=float, default=1.9999999)
parser.add_argument("-max_iter", help = "max iter number", type=int, default=50000)
parser.add_argument("-delta", help = "iterate until error < delta", type=float, default=1e-9)
parser.add_argument("-fast", help = "use method of tripartition", type=bool, default=True)
args = parser.parse_args()

def test(x):
    file = open("config")
    data = file.readlines()
    newdata = []
    for line in data:
        d = line.strip().split()
        if len(d)==0: continue
        if d[0] != "nodebasedGSMethod":
            newdata.append(line)
        else:
            newdata.append("nodebasedGSMethod {} {} {} {} {}\n".format(args.max_iter, x, 0, "yes", args.delta))   #nodebasedGSMethod 50000 1.97576 0 yes 1e-5
    file.close()
    file = open("config", "w")
    file.writelines(newdata)
    file.close()
    os.system("./sim >>temp_out.out")

def getLastTwoIterTime():
    file = open("temp_out.out")
    data = file.readlines()
    t1, t2= -1, 0
    for line in data:
        d = line.strip().split()
        if (len(d)==0): continue
        if d[0]=="threshould":
            t1 = t2
            t2 = int(d[5])
    file.close()
    return t1, t2



recordfile = open("print_omega_fig.out", "w")
temp = open("temp_out.out", "w")
temp.writelines([])
temp.close()
total_iter_needed = 0

if args.fast==False:
    x = args.Min
    while x < args.Max:
        test(x)
        x += args.S
else:
    l = args.Min
    r = args.Max
    best_omega = 1.0
    iter_time = 5000000
    while abs(r-l)>args.S:
        mid1, mid2 = (r-l)/3+l, (r-l)/3*2+l
        test(mid1)
        test(mid2)
        t1, t2 = getLastTwoIterTime()
        total_iter_needed += (t1+t2)
        if t2>t1:
            r = mid2
        else:
            l = mid1
    test(r)

print("total iter needed to get best w is {}".format(total_iter_needed))
file = open("temp_out.out")
data = file.readlines()
record = []
best_omega = 1.0
iter_time = 500000
for line in data:
    d = line.strip().split()
    if len(d)==0: continue
    if len(d)>8 and d[4]=="GS-method,":
        x = float(d[7]) 
    if d[0]=="nodebase":
        record.append(str(x)+" ")
        record.append(d[5]+" ")
    elif d[0]=="threshould":
        record.append(d[5]+"\n")
        if int(d[5])<iter_time: 
            best_omega = x
            iter_time = int(d[5])
        
recordfile.write("best omega = {}, iter_time = {}\n".format(best_omega, iter_time))
file.close()
recordfile.writelines(record)
recordfile.close()






