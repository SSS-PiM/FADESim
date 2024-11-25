import os
import random
import argparse
import numpy as np
import re

def print_res(gather_res : dict, type : str) -> None:
    for key in gather_res:
        print(key, f", {type} relative error = ")
        value = gather_res[key]
        print(",".join(map(str, value)))
        value.sort()
        vmax = value[-1]
        vmin = value[0]
        ave = np.average(value)
        mid = np.median(value)
        print(f"min = {vmin}, max = {vmax}, mid = {mid}, ave = {ave}\n")

def read_res_from_file(log_file_name : str, gather_res_max : dict, gather_res_ave : dict) -> None:
    file = open(log_file_name, "r")
    all_data = file.read()

    pattern = re.compile(r"((?:gs|sci|pbia|aihwkit|scn|ir)\s+(?:\w+\s+)*)error=[\s\S]{,70}Iout max (?:\w+\s+)*=\s+(\d*\.*\d*)")#\%")
    res = pattern.findall(all_data)

    for line in res:
        if gather_res_max.get(line[0]) == None:
            gather_res_max[line[0]] = []
        gather_res_max[line[0]].append(float(line[-1]))
        # print(line)
    
    pattern = re.compile(r"((?:gs|sci|pbia|aihwkit|scn|ir)\s+(?:\w+\s+)*)error=[\s\S]{,200}Iout avg (?:\w+\s+)*=\s+(\d*\.*\d*)")#\%")
    res = pattern.findall(all_data)

    for line in res:
        if gather_res_ave.get(line[0]) == None:
            gather_res_ave[line[0]] = []
        gather_res_ave[line[0]].append(float(line[-1]))    


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-basetest", help = "two base test", action="store_true", default=False)
    parser.add_argument("-test_num", help = "test num for loop", type=int, default=100)
    parser.add_argument("-addtest", help = "start additional test, if not we will only gather result", action="store_true", default=False)

    args = parser.parse_args()

    if args.basetest:
        os.system("python3 test_diff_fastmode.py -M 128 -N 128 -r_wire 2.93 -gs_en -gs_params \"726\" \"1.97549\" \"0\" \"no\" \"1e-9\" -aihwkit_en -aihwkit_params \"0.467566\" \"1\" -scn_en -scn_model_load_file SCNAlox_hfox_128x128_293_5w_checkp_lrs16900.pt -scn_use_float64 -em_en -ngspice_en -states 74867 16.9e3 -ir_free_en -num 0")

        os.system("python3 test_diff_fastmode.py -M 128 -N 128 -r_wire 2.93 -gs_en -gs_params \"689\" \"1.97513\" \"0\" \"no\" \"1e-9\" -aihwkit_en -aihwkit_params \"0.066467\" \"1\" -scn_en -scn_model_load_file SCNAlox_hfox_128x128_293_5w_checkp_lrs50k.pt -scn_use_float64 -em_en -pbia_en -pbia_params \"100\" \"yes\" \"1e-9\" -ngspice_en -states 221.5e3 50e3 -ir_free_en -num 1")

    gather_res_max = {}
    gather_res_ave = {}
    for num in range(1, args.test_num+1):
        in_p = random.randint(5, 95)/100.0
        s_p = random.randint(5, 95)/100.0
        # in_p, s_p = 1, 1

        if args.addtest:
            print(f"in p = {in_p}, s p = {s_p}")
            os.system("python3 test_diff_fastmode.py -M 128 -N 128 -r_wire 2.93 -gs_en -gs_params \"689\" \"1.97513\" \"0\" \"no\" \"1e-9\" -aihwkit_en -aihwkit_params \"0.066467\" \"1\" -scn_en -scn_model_load_file SCNAlox_hfox_128x128_293_5w_checkp_lrs50k.pt -scn_use_float64 -em_en -pbia_en -pbia_params \"100\" \"yes\" \"1e-9\"  -states 221.5e3 50e3 -ir_free_en -num {} -rd -rd_input_params {} {} -rd_state_params {} {}".format(num, in_p, 1-in_p, s_p, 1-s_p))
        log_file_name = "size%dx%d_r%.2f_%d_rd"%(128, 128, 2.93, num)
        log_file_name = "log_%s.log"%log_file_name
        read_res_from_file(log_file_name, gather_res_max, gather_res_ave)

    print_res(gather_res_max, "max")
    print_res(gather_res_ave, "avg")


    print("\n\n")
    gather_res_max = {}
    gather_res_ave = {}
    for num in range(args.test_num+1, 2*args.test_num+1):
        in_p = random.randint(5, 95)/100.0
        s_p = random.randint(5, 95)/100.0
        # in_p, s_p = 1, 1

        if args.addtest:
            print(f"in p = {in_p}, s p = {s_p}")
            os.system("python3 test_diff_fastmode.py -M 128 -N 128 -r_wire 2.93 -gs_en -gs_params \"726\" \"1.97549\" \"0\" \"no\" \"1e-9\" -aihwkit_en -aihwkit_params \"0.467566\" \"1\" -scn_en -scn_model_load_file SCNAlox_hfox_128x128_293_5w_checkp_lrs16900.pt -scn_use_float64 -em_en  -states 74867 16.9e3 -ir_free_en -num {} -rd -rd_input_params {} {} -rd_state_params {} {}".format(num, in_p, 1-in_p, s_p, 1-s_p))
        log_file_name = "size%dx%d_r%.2f_%d_rd"%(128, 128, 2.93, num)
        log_file_name = "log_%s.log"%log_file_name
        read_res_from_file(log_file_name, gather_res_max, gather_res_ave)
    
    print_res(gather_res_max, "max")
    print_res(gather_res_ave, "avg")