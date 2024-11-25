import os
import random
import argparse
import numpy as np
import re
import sys

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
    parser.add_argument("-test_num", help = "test num for loop", type=int, default=100)
    parser.add_argument("-addtest", help = "start additional test, if not we will only gather result", action="store_true", default=False)

    args = parser.parse_args()


    arr_size = [32, 64, 128]
    r_wire = [1, 2.93, 5, 10]
    Ron = 16.9e3
    Roff = Ron*4.43

    gs_params = {
        "32_1_ron16.9k": ["171", "1.91111"],
        "64_1_ron16.9k": ["336", "1.95093"], 
        "128_1_ron16.9k": ["689", "1.97515"],
        "32_2.93_ron16.9k": ["178", "1.90542"],
        "64_2.93_ron16.9k": ["359",  "1.95211"],
        "128_2.93_ron16.9k": ["726", "1.97549"],
        "32_5_ron16.9k": ["186", "1.90739"],
        "64_5_ron16.9k": ["357", "1.95082"],
        "128_5_ron16.9k": ["754", "1.97513"],
        "32_10_ron16.9k": ["187", "1.90789"],
        "64_10_ron16.9k": ["383", "1.95184"],
        "128_10_ron16.9k": ["785", "1.97513"],
    }
    
    scn_model_names = {
        "32_1_ron16.9k": "SCNAlox_hfox_32x32_1_5w.pt",
        "64_1_ron16.9k":  "SCNAlox_hfox_64x64_1_5w.pt", 
        "128_1_ron16.9k": "SCNAlox_hfox_128x128_1_5w.pt",
        "32_2.93_ron16.9k": "SCNAlox_hfox_32x32_293_5w.pt",
        "64_2.93_ron16.9k": "SCNAlox_hfox_64x64_293_5w.pt",
        "128_2.93_ron16.9k":"SCNAlox_hfox_128x128_293_5w.pt",
        "32_5_ron16.9k": "SCNAlox_hfox_32x32_5_5w.pt",
        "64_5_ron16.9k": "SCNAlox_hfox_64x64_5_5w.pt",
        "128_5_ron16.9k": "SCNAlox_hfox_128x128_5_5w.pt",
        "32_10_ron16.9k": "SCNAlox_hfox_32x32_10_5w.pt",
        "64_10_ron16.9k": "SCNAlox_hfox_64x64_10_5w.pt",
        "128_10_ron16.9k": "SCNAlox_hfox_128x128_10_5w.pt",
    }

    for r in r_wire:
        for size in arr_size:

            if type(r)==int:
                name = f"{size}_{r}_ron16.9k"
            else:
                name = f"{size}_{r:.2f}_ron16.9k"
            gs_para = gs_params[name]
            scn_model_name = "./model/"+scn_model_names[name]
            # print(gs_para)
            # continue

            gather_res_max = {}
            gather_res_ave = {}
    
            for num in range(1, args.test_num+1):
                in_p = random.randint(5, 95)/100.0
                s_p = random.randint(5, 95)/100.0
                # in_p, s_p = 1, 1

                if args.addtest:
                    print(f"in p = {in_p}, s p = {s_p}")
                    os.system(f"python3 test_diff_fastmode.py -M {size} -N {size} -r_wire {r} -gs_en -gs_params {str(2000)} {str(gs_para[1])} \"0\" \"yes\" \"1e-9\" -scn_en -scn_model_load_file {scn_model_name} -scn_use_float64 -em_en -pbia_en -pbia_params \"100\" \"yes\" \"1e-9\"  -states {Roff} {Ron} -ir_free_en -num {num} -rd -rd_input_params {in_p} {1-in_p} -rd_state_params {s_p} {1-s_p}")
                log_file_name = "size%dx%d_r%.2f_%d_rd"%(size, size, r, num)
                log_file_name = "log_%s.log"%log_file_name
                read_res_from_file(log_file_name, gather_res_max, gather_res_ave)

            
            with open(f"logfin_arr{size}_r{r}_ron{Ron}.log", 'w') as f:
                sys.stdout = f
                print_res(gather_res_max, "max")
                print_res(gather_res_ave, "avg")
                sys.stdout = sys.__stdout__
