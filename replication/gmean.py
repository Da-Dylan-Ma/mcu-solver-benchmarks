import math
from statistics import geometric_mean

def parse_log_and_calculate_gmean(file_path):
    per_iteration_timings = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    norm = float(parts[0])
                    iterations = int(parts[1])
                    time_us = int(parts[2])

                    if iterations > 0:
                        per_iteration_time = time_us / iterations
                        per_iteration_timings.append(per_iteration_time)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")

    if per_iteration_timings:
        gmean = geometric_mean(per_iteration_timings)
        print(f"Geometric Mean of Per Iteration Timing: {gmean:.4f} us")
    else:
        print("No valid data found in the log file.")

parse_log_and_calculate_gmean('osqp_mpc_1e-4.txt')
