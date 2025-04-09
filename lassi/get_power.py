# Used for power consumption
import csv
import time
from datetime import datetime
import subprocess
from threading import Thread
import threading
from queue import Queue
import queue
import pandas as pd
import matplotlib.pyplot as plt
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, NVMLError, nvmlShutdown
from lassi.format_colors import color


def query_gpu_state():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.gr,clocks.sm,power.draw,pstate",
                "--format=csv"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("\n----- GPU State:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(" Failed to query GPU state:")
        print(e.stderr)

        
def report_gpu_memory():
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print("\n----- GPU Memory Status:")
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            used = info.used // (1024 * 1024)
            free = info.free // (1024 * 1024)
            total = info.total // (1024 * 1024)
            print(f" GPU {i}: {used} MiB used, {free} MiB free, {total} MiB total")
        query_gpu_state()
    except NVMLError as error:
        print("Failed to query GPU memory:", error)
    finally:
        try:
            nvmlShutdown()
        except NVMLError as shutdown_error:
            print("Failed to shut down NVML cleanly:", shutdown_error)

            
# Not used in LASSI - Energy Efficiency version
def get_gpu_power():
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        power = float(result.stdout.strip())
        return power
    except Exception as e:
        print(f"Error collecting GPU power data: {e}")
        return None
            

def power_collection_thread(power_queue, stop_event, handle, interval=0.01):
    """Thread function to collect power data from the GPU."""
    start_time = time.time()
    while not stop_event.is_set():
        try:
            current_time = time.time() - start_time
            # Query GPU power usage
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            # power = lassi.getidlepower.get_gpu_power()
            if power is not None:
                power_queue.put((current_time, power))
            time.sleep(interval)
        except Exception as e:
            print(f"Error querying power: {e}")
            break 
        # time.sleep(0.01)


def plot_gpu_power(file_name, start_time, end_time, idle=False):
    data = pd.read_csv(file_name)

    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data['GPU Power (W)'], label='GPU Power')
    
    # Highlight the inference period
    plt.axvspan(float(start_time), float(end_time), alpha=0.2, color='yellow', label='Execution Period')
    
    plt.axvline(x=float(start_time), color='g', linestyle='--', label='Execution Start')
    plt.axvline(x=float(end_time), color='r', linestyle='--', label='Execution End')
    # plt.axvline(x=float(data['Unload model time'][0]), color='b', linestyle='--', label='Model Unloaded')
    
    if idle:
        plt.title("Total GPU Power Usage Over Time")  # for"  {file_name}')
        plt.ylabel('GPU Power (W)')
    else:
        plt.title("Net GPU Power Usage Over Time")
        plt.ylabel('GPU Power (W) - idle power substracted')
    plt.xlabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return True  # data['Inference Duration'][0]


# Function to calculate total energy consumption
def calculate_total_energy(file_name, start_time, end_time):
    data = pd.read_csv(file_name)
    
    # Filter data to only include the inference period
    print("Start time: " + str(round(start_time, 3)) + " -- End time: " + str(round(end_time, 3)))
    code_execution_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()
    
    if len(code_execution_data) > 1:
        time_interval = code_execution_data['Time'].diff().mean()
    else:
        print("Not enough data points to calculate energy consumption.")
        return 0, 0, 0

    # Because we subtract off ave idle power, we could have negative values,
    # which are not physically meaningful. So, we clamp negative net power values to zero.
    code_execution_data['GPU Power (W)'] = code_execution_data['GPU Power (W)'].clip(lower=0)

    # Calculate total energy consumption in joules
    total_energy = (code_execution_data['GPU Power (W)'] * time_interval).sum()
    ave_power = code_execution_data['GPU Power (W)'].mean()
    std_power = code_execution_data['GPU Power (W)'].std()

    # Get baseline energy consumption for same duration
    # print("\nCapturing idle energy consumed for same duration...") 
    #idle_energy, ave_idle_power = idlepower(int(end_time - start_time))
    # print("... Complete.")
    
    return total_energy, ave_power, std_power

    
def display_energy_results(file_name, start_time, end_time, idle):
    plot_gpu_power(file_name, start_time, end_time, idle)
    total_energy_joules, ave_power, std_power = calculate_total_energy(file_name, start_time, end_time)
    
    # Convert energy units
    total_energy_wh = total_energy_joules / 3600  # Convert from Joules to Watt-hours
    total_energy_kwh = total_energy_wh / 1000  # Convert from Watt-hours to Kilowatt-hours
    
    execution_duration = end_time - start_time
    if idle:
        print(f"Total GPU energy consumption for {file_name}:")
    else:
        print(f"Net (idle corrected) GPU energy consumption for {file_name}:")
    print(f"  {total_energy_kwh:.6f} kWh")
    print(f"  {total_energy_wh:.4f} Wh")
    print(f"  {total_energy_joules:.2f} J")
    print("Average GPU power: " + color.BOLD + f"{ave_power:.2f} W +/- {std_power:.2f} " + color.END)
    print(" with sample standard deviation.")
    if idle:
        print("Total Energy During Exe Runtime: " + color.BOLD + f"  {total_energy_joules:.2f} J" + color.END)
    else:
        print("Net Energy During Exe Runtime: " + color.BOLD + f"  {total_energy_joules:.2f} J" + color.END)
    print("Code execution duration: " + color.BOLD + f" {execution_duration:.2f} seconds" + color.END)
    # print("Idle Energy (same duration): " + f"  {idle_energy:.2f} J")
    print("----------------------------------------------------------------------------------------------------------------------------------------")

    metrics_results_string = f"Total Energy During Exe Runtime: {total_energy_joules:.2f} Joules"
    metrics_results_string += f"\nAverage GPU power: {ave_power:.2f} Watts +/- {std_power:.2f}"
    metrics_results_string += f"\nCode execution duration: {execution_duration:.2f} seconds" 
   
    return total_energy_joules, ave_power, std_power, metrics_results_string

    
# Function to read CSV and plot the GPU power usage
def power_collect_data(power_queue, code_source, target_lang, start_exe_time, end_exe_time, ave_idle_power, idle):
    # now = datetime.now()
    # dt_string = now.strftime("%m-%d-%Y_t%H_%M_%S")
    
    # Collect power data
    power_data = []
    while not power_queue.empty():
        power_data.append(power_queue.get())
    
    # Subtract off idle power:
    # a list of tuples (timestamp, power):
    net_power_data = [
        (t, p - ave_idle_power) for (t, p) in power_data
    ]
    
    # Write results to CSV
    if code_source == "" and target_lang == "":
        code_source = "idle"
    filename = f"{code_source}_{target_lang}_gpu_net_power_results.csv"  # _{dt_string}
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'GPU Power (W)'])
        for time_elapsed, power in net_power_data:
            writer.writerow([f"{time_elapsed:.2f}", power])
    
    # print(f"\nMODEL TRIAL: {experiment}")
    # print(f"Inference time: {inference_duration:.2f} seconds")
    print(f"GPU power data has been written to {filename}")

    total_energy_joules, ave_power, std_power, metrics_results_string = display_energy_results(filename, start_exe_time, end_exe_time, idle)
    
    return total_energy_joules, ave_power, std_power, metrics_results_string

    
def idlepower(duration):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
    
    start_power_time = time.time()
    power_queue = queue.Queue()
    stop_power_event = threading.Event()
    
    # Start power collection thread
    power_thread = Thread(target=power_collection_thread, args=(power_queue, stop_power_event, handle))
    power_thread.start()
    # time.sleep(2)))
         
    # Continue collecting power data for a short time after inference
    print("Duration for idle time measurement: " + str(round(duration, 2)))
    # end_power_time = time.time()
    time.sleep(duration)
    end_power_time = time.time()
    # while (end_power_time - start_power_time) < duration:
        # time.sleep(0.001)
    #    end_power_time = time.time()
        #print("... idle duration increment: " + str(end_power_time - start_power_time))
    stop_power_event.set()
    power_thread.join()
    # print("... idle duration increment: " + str(end_power_time - start_power_time))
    
    # We don't need the metrics_results_string for idle metrics.
    total_energy_joules, ave_power, std_power, metrics_results_string = power_collect_data(power_queue, "", "", 0, end_power_time - start_power_time, ave_idle_power=0, idle=True)

    # Shutdown NVML
    nvmlShutdown()

    return round(total_energy_joules, 2), round(ave_power, 2), round(std_power, 2)
