import serial
import os

# Configure serial connection
serial_port = "/dev/ttyACM0"  # Replace with the Teensy's actual port
baud_rate = 9600
output_dir = "teensy_logs"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Open serial connection
ser = serial.Serial(serial_port, baud_rate)
print(f"Listening on {serial_port}...")

# File handles for CSVs
files = {
    "SPARSITY_P": open(os.path.join(output_dir, "P_sparsity.csv"), "w"),
    "SPARSITY_A": open(os.path.join(output_dir, "A_sparsity.csv"), "w"),
    "VECTOR_q": open(os.path.join(output_dir, "q_vector.csv"), "w"),
    "VECTOR_l": open(os.path.join(output_dir, "l_vector.csv"), "w"),
    "VECTOR_u": open(os.path.join(output_dir, "u_vector.csv"), "w"),
    "ITERATION": open(os.path.join(output_dir, "iterations.csv"), "w"),
}

try:
    while True:
        line = ser.readline().decode("utf-8").strip()
        parts = line.split(",")

        if parts[0] == "SPARSITY":
            # Sparsity pattern: SPARSITY,label,iteration,row,col,value
            label = parts[1].replace(" ", "_")
            iteration = parts[2]
            file_key = f"SPARSITY_{label}"
            for row in ser:
                row = row.decode("utf-8").strip()
                if row == "":
                    break  # End of sparsity block
                files[file_key].write(f"{iteration},{row}\n")

        elif parts[0] == "VECTOR":
            # Vectors: VECTOR,label,iteration,value
            label = parts[1]
            iteration = parts[2]
            file_key = f"VECTOR_{label}"
            for value in ser:
                value = value.decode("utf-8").strip()
                if value == "":
                    break  # End of vector block
                files[file_key].write(f"{iteration},{value}\n")

        elif parts[0] == "ITERATION":
            # Iteration details: ITERATION,norm,iterations,time
            files["ITERATION"].write(",".join(parts[1:]) + "\n")

except KeyboardInterrupt:
    print("Stopping logging...")

finally:
    # Close all files
    for f in files.values():
        f.close()
    ser.close()
