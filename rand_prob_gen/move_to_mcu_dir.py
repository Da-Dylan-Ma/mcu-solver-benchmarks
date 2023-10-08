"""
File: execute_random_tests.py
Author: Anoushka, Khai
Date: 2023-09-15
Description: A Python script to move random MPC problem generated by prob_data_gen.py to MCU source code directory, also generate C code for OSQP.
"""

import os
import time
path_to_root = os.getcwd()
print(path_to_root)

import sys
from rand_prob_gen.gen_code_osqp import generate_osqp_solver

from rand_prob_gen.gen_mpc_problem import *

# SOLVER = 'TinyMPC'
SOLVER = 'OSQP'

# num_tests = 24
# timings = np.zeros(num_tests)
# each row is the parameters and timing/memory results of a new problem
# columns = nx, nu, Nh, Nsim, timing of each MPC iter, number iterations, tracking error sqrd norm, memory footprint
# dataset = np.zeros((24, 4+200+1+1+1))

for prob_i, prob_dir in enumerate(os.listdir(path_to_root+'/random_problems/')):
    if prob_dir == 'prob_nx_8':  # Select problem instance (directory)
    # if 'Nh' in prob_dir or 'nx' in prob_dir or 'nu' in prob_dir:
        path_to_prob_dir = path_to_root+'/random_problems/'+prob_dir

        # Transfer for the new problem to PlatformIO directories
        if SOLVER == 'OSQP':
            generate_osqp_solver(path_to_prob_dir)  # generate C code here, use OSQP v1
            os.system('cp -R '+path_to_prob_dir+'/generated_osqp_solver/osqp_configure.h '+path_to_root+'/../teensy_osqp_benchmark/lib/osqp/inc/osqp_configure.h')
            os.system('cp -R '+path_to_prob_dir+'/generated_osqp_solver/osqp_data_workspace.h '+path_to_root+'/../teensy_osqp_benchmark/include/osqp_data_workspace.h')
            os.system('cp -R '+path_to_prob_dir+'/generated_osqp_solver/osqp_data_workspace.c '+path_to_root+'/../teensy_osqp_benchmark/src/osqp_data_workspace.c')
            os.system('cp -R '+path_to_prob_dir+'/rand_prob_osqp_params.npz '+path_to_root+'/../teensy_osqp_benchmark/src/rand_prob_osqp_params.npz')
            os.system('cp -R '+path_to_prob_dir+'/rand_prob_osqp_xbar.h '+path_to_root+'/../teensy_osqp_benchmark/lib/osqp/inc/public/rand_prob_osqp_xbar.h')
            print("For OSQP")

        if SOLVER == 'TinyMPC':
            os.system('cp -R '+path_to_prob_dir+'/rand_prob_tinympc_params.hpp '+path_to_root+'/../teensy_tinympc_benchmark/include/problem_data/rand_prob_tinympc_params.hpp')
            os.system('cp -R '+path_to_prob_dir+'/rand_prob_tinympc_xbar.hpp '+path_to_root+'/../teensy_tinympc_benchmark/include/problem_data/rand_prob_tinympc_xbar.hpp')
            os.system('cp -R '+path_to_prob_dir+'/constants.hpp '+path_to_root+'/../teensy_tinympc_benchmark/include/constants.hpp')
            print("For TinyMPC")
        print(prob_dir)
        # Time OSQP
        # os.system('cd /Users/anoushkaalavill/Documents/REx_Lab/mcu-testing/teensy_tinympc_benchmark; pio run --target upload')
        # os.system('cd '+path_to_root+'/teensy_tinympc_benchmark; pio run --target upload')
        # time.sleep(30)

        # with serial.Serial('/dev/cu.usbmodem111513201', 9600, timeout=1) as ser:
        #     while ser.is_open:
        #         line = ser.readline()
        #         line = line.decode('ASCII')
        #         if line:
        #             print(line)
        #             timings[prob_i] = line
        #             break
        
        # print("PRESS BOOT")
        # time.sleep(30)
        
        # print(timings)
        # scipy.io.savemat(path_to_prob_dir+'/timings.mat', {'timings': timings})
