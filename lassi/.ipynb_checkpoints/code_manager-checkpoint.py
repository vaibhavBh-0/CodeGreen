import os
import subprocess
import time
import shlex  # for splitting command line arguments
from tqdm import tqdm  # for progress bar feature
# Used for power consumption
import csv
from threading import Thread
import threading
from queue import Queue
import queue
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
import lassi.get_power
from lassi.format_colors import color


def build_script_code(folder, code_build_script):
    print("Build script - current dir: " + str(os.getcwd()))
    # Execute the build script command
    build_script_file = str(os.getcwd()) + "/" + folder + "/altis/" + code_build_script + ".sh"
    subprocess.run(['chmod', '+x', build_script_file], check=True)
    result = subprocess.run([build_script_file], capture_output=True, text=True)

    # Check if the compilation was successful
    return_result = ""
    if result.returncode == 0:
        print("\n" + color.BOLD + "Compilation of code successful!" + color.END)
    else:
        # If there was an error, print the error message
        return_result = result.stderr
        print("\n" + color.BOLD + color.RED + "***** Compilation failed." + color.END)
        print(" Error message: ")
        print(return_result)

    return return_result


def compile_code(source_file, lang, code_compiler, code_compiler_kwds):
       
    # Split the compile keywords into a list of arguments
    compile_kwds_list = shlex.split(code_compiler_kwds)

    if lang == "OMP":
        # The desired name of the compiled program
        output_file = source_file.split('.cpp')[0]
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
    
    elif lang == "CUDA":
        # The desired name of the compiled program
        output_file = source_file.split('.cu')[0]
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
        
    print("Compile command: " + str(compile_command))
    
    # Execute the compilation command
    result = subprocess.run(compile_command, capture_output=True, text=True)

    # Check if the compilation was successful
    return_result = ""
    if result.returncode == 0:
        print("\n" + color.BOLD + "Compilation of code successful!\n" + color.END)
    else:
        # If there was an error, print the error message
        return_result = result.stderr
        print("\n" + color.BOLD + color.RED + "***** Compilation failed.\n" + color.END)
        print(" Error message: ")
        print(return_result)
        
    return return_result, output_file


def execute_code(executable_file, exe_input_parameters, power_measure, code_source, target_lang, build_script, wait=10):

    # Wait 20 seconds for the system to settle down
    # print("...WAITING.")
    # time.sleep(20)
    # print("...DONE WAITING
    if power_measure:
        print("\n----- >>> System check BEFORE exe code...")
        lassi.get_power.report_gpu_memory()

    duration = 0
    start_power_time = 0
    total_energy_joules = 0
    ave_power = 0
    std_power = 0
    metrics_results_string = ""

    # Simulating a progress bar while sleeping for 5 seconds
    for _ in tqdm(range(wait), desc="Waiting for the system to settle down for " + str(wait) + " seconds..."):
        time.sleep(1)

    return_code = ""
    
    # Capture current directory to switch back after code execution
    initial_dir = os.getcwd()

    # Extract the compiled program to execute
    directory_path = os.path.dirname(executable_file)
    executable = os.path.basename(executable_file)

    # Change the working directory to the subdirectory of the executable
    os.chdir(directory_path)

    # Ensure the binary has execute permissions
    subprocess.run(['chmod', '+x', executable])

    # Build execute command
    if build_script:
        execute_command = ['./' + executable] + exe_input_parameters[0].split(' ')
    else:
        execute_command = ['./' + executable] + exe_input_parameters
    print("Working directory: " + directory_path)
    print("Execute command: " + str(execute_command) + "\n")

    # Execute the binary object with input parameters, if provided

    if power_measure:
        try:
            nvmlInit()
        except NVMLError as err:
            print(f"Failed to initialize NVML: {err}")
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
        # Collect idle power average value before exe
        idle_power_before = []
        for _ in range(200):  # 2 seconds @ 0.01s intervals
            p = nvmlDeviceGetPowerUsage(handle) / 1000.0
            idle_power_before.append(p)
            time.sleep(0.01)
        ave_idle_before = sum(idle_power_before) / len(idle_power_before)
     
    # Ensure the file is open and available before executing
    if os.path.exists(executable):
        try:
            # Open the file to ensure it is accessible
            with open(executable, 'rb') as code_file:
                # If power measurement is enabled, set it up
                if power_measure:
                    start_power_time = time.time()
                    power_queue = queue.Queue()
                    stop_power_event = threading.Event()
    
                    # Start power collection thread
                    power_thread = Thread(target=lassi.get_power.power_collection_thread, args=(power_queue, stop_power_event, handle))
                    power_thread.start()
                    time.sleep(15)
    
                # Start the process
                start_exe_time = time.time() - start_power_time
                repeat = 1
                stdout_lines = []
                stderr_lines = []
                for i in range(0, repeat):
                    process = subprocess.Popen(execute_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    exception_error = ""
                    
                    # Print the output in real-time
                    print("\n----- < BEGIN: STANDARD OUTPUT FROM CODE EXECUTION > -----\n")
                    try:
                        # Read stdout in a separate thread or in a non-blocking way
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            print(line, end='')  # Print each line as it is received
                            stdout_lines.append(line)
                        process.stdout.close()
                
                        # Read stderr in a separate thread or in a non-blocking way
                        while True:
                            line = process.stderr.readline()
                            if not line:
                                break
                            stderr_lines.append(line)
                        process.stderr.close()
                    
                        # Wait for the process to complete and get the return code
                        return_code = process.wait()
                        
                        if return_code != 0:
                            error_output = ''.join(stderr_lines)
                            print("Program Execution Error:", error_output)
    
                    except Exception as e:
                        print("An error occurred while executing the program: ", e)
                        exception_error = str(e)

                    print("\n----- < END: Standard Output > -----\n")

                total_energy_joules = 0
                ave_power = 0
                # Only continue power measurements if no exe error
                if power_measure and return_code == 0:            
                    end_exe_time = time.time() - start_power_time
                    duration = end_exe_time - start_exe_time
                    # Continue collecting power data for a short time after inference
                    time.sleep(15)
                    end_power_time = time.time()            
                    # while (end_power_time - start_power_time) < 10:
                        # time.sleep(0.01)
                    #    end_power_time = time.time()
                    stop_power_event.set()
                    power_thread.join()
                    # Collect idle power average value after exe
                    idle_power_after = []
                    for _ in range(200):  # 2 seconds @ 0.01s intervals
                        p = nvmlDeviceGetPowerUsage(handle) / 1000.0
                        idle_power_after.append(p)
                        time.sleep(0.01)
                    ave_idle_after = sum(idle_power_after) / len(idle_power_after)

                    ave_idle_power = (ave_idle_before + ave_idle_after) / 2
        
                    total_energy_joules, ave_power, std_power, metrics_results_string = lassi.get_power.power_collect_data(
                        power_queue, 
                        code_source, 
                        target_lang, 
                        start_exe_time, 
                        end_exe_time,
                        ave_idle_power,
                        idle=False
                    )

                    # Shutdown NVML
                    try:
                        nvmlShutdown()
                    except NVMLError as err:
                        print(f"Failed to shutdown NVML: {err}")
        
        except IOError as e:
            print(f"Pipeline error | An I/O error occurred: {e.strerror}")
        except Exception as e:
            print(f"Pipeline error | An error occurred: {e}")
    else:
        print(f"Pipeline error | The file {executable} does not exist.")
    
    
    # Check if the program ran successfully
    return_result = "0"  # default OK
    if return_code == 0:
        print("\n" + color.BOLD + "Program executed successfully!" + color.END)
    else:
        # If there was an error during execution, print the error message
        return_result = "Program exited with return code: " + str(return_code) + " "
        if return_code == -11:
            return_result += "Segmentation fault detected."
        stderr_output = ''.join(stderr_lines)
        if exception_error != "":
            return_result += "\nException Error: " + exception_error
        if stderr_output:
            return_result += "\nStandard Error: " + stderr_output
        print("\n" + color.BOLD + color.RED + "***** Execution failed." + color.END)
        print("Error message: ")
        print(return_result)
    
    os.chdir(initial_dir)

    stdout_output = ''.join(stdout_lines)

    if power_measure:
        print("\n----- >>> System check AFTER exe code...")
        lassi.get_power.report_gpu_memory()

    # Convert to native Python float:
    total_energy_joules = float(total_energy_joules)

    return return_result, stdout_output, total_energy_joules, ave_power, std_power, duration, metrics_results_string