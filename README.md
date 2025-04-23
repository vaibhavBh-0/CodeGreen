![CodeGreenJamLogo](/images/CodeGreen_logo.png)

# About the Jam

Large energy demands in high-performance computing (HPC) systems have become a major concern in terms of stress on current power infrastructure and environmental impact from carbon emission [(Silva et al. 2024)](https://doi.org/10.1016/j.rser.2023.114019). For example, current exascale systems require between 24 and 39 MW of power [(Top 500)](http://www.top500.org/).  Further, the hyperscale tech companies are seeking facilities that can support gigawatts of power, introducing challenges in expanding existing transmission networks [(Miller 2024)](https://www.datacenterfrontier.com/hyperscale/article/55021675/the-gigawatt-data-center-campus-is-coming).

While research in developing more energy efficient *hardware* is critical, there remain ample opportunities to *write performant parallel code* that requires less energy consumption.

Energy-aware programming is a learning gap that can become an expert skill and career differentiator for tomorrow's software engineers.

The **Code Green Jam** is your opportunity to experimentally learn how to write energy-efficient CUDA C++ code. This event does not bring you the answers--you are provided with power measurement tools, access to GPU compute, real code examples, and learning resources to try to figure things out. The goal is for you to walk away with an increased appreciation for energy-efficient parallel programming and a foundation to continue learning and practicing more. 

## GPU Compute 

For the Code Green Jam, you are provided access during the event to the NCSA's [Delta](https://docs.ncsa.illinois.edu/systems/delta/en/latest/) GPU compute environment.

## Test code and Evaluation tools

The power evaluation tools provided is a sub-set of the **LASSI** framework (**L***LM-based* **A***utomated*
**S***elf-correcting pipeline for generating parallel* **S***c***I***entific codes*), a current research project in the [SPEAR Lab at UIC](https://spear.lab.uic.edu/). This project explores how to harness large language models to autonomously generate and refactor existing scientific parallel codes either for translating to an altnerate programming language or to improve energy efficiency--the same goal of the Code Green Jam. The first paper on code tranlsation is linked below and the latest work on energy efficiency was recently submitted for a double-blind review in mid-April 2025, so is not yet available for sharing.

The test codes provided are the same evaluated in this recent research. These are located in the ```test_codes``` folder of this repo. Each parallel code application is included in a subfolder with its folder name being the same as the application name.

You are also provided with a separate folder, ```lassi_solutions```, to see how the automated LASSI pipeline refactored the same code you will try out during the Code Green Jam. Along with this *possible* solution is a written comparison between the original source code and the refactored code, as evaluated by an LLM-as-a-Judge agent. This solution provided by the LASSI pipeline may not be optimal and requires additional human-in-the-loop evaluation to determine its validity--this is where you come in as a participant of the Code Green Jam!


## Related Learning Documents

Become familiar with the folloing reference documents to better inform your approach to refactoring paralle code to consume less energy:

1. C++ programming Guide by NVIDIA:
[PDF](resources/NVIDIA_CUDA_Cpp_Programming_Guide_v12-8.pdf) or online at 
https://docs.nvidia.com/cuda/cuda-c-programming-guide/


2. CUDA C++ Best Practices Guide by NVIDIA:
[PDF](resources/NVIDIA_CUDA_Cpp_Best_Practices_Guide_v12-8.pdf) or online at 
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

3. Sample CUDA code exercises are avaiable in this repo: ```CodeGreen/resources/CUDA_exercises/```


## Related research: LASSI 

Original LASSI paper:
https://ieeexplore.ieee.org/document/10740822 

M. T. Dearing, Y. Tao, X. Wu, Z. Lan and V. Taylor, "LASSI: An LLM-Based Automated Self-Correcting Pipeline for Translating Parallel Scientific Codes," 2024 IEEE International Conference on Cluster Computing Workshops (CLUSTER Workshops), Kobe, Japan, 2024, pp. 136-143, doi: 10.1109/CLUSTERWorkshops61563.2024.00029.


# Getting Starting

All registered participants of the Code Green Jam have been given access to the NCSA Delta GPU cluster. Pre-registration and account setup with this system should have been completed *before* the start of the Jam event.

## Step 1: Remote in to Detla GPU

Follow Steps 5 and 6 in the [*Code Green Jam - Getting Started Guide*](resources/Code_Green_Setup_Instructions.pdf) to log in to the Detla GPU cluster and initiate a Jupyter Lab notebook.

NOTE: When starting your Jupyter Lab instance, please select the following partition: ```gpuA40x4-interactive```

## Step 2: Clone the Code Green repo

Starting at the NCSA's Delta Open OnDemand page where you launched your Jupyter Lab instance, open a separate terminal from the top menu bar options Cluster >> Delta Shell Access.

![Deltashell](/images/Delta_Shell_Access.png)

This will open an interactive terminal in your web browser and bring you to a command line prompt with your username. From here you should clone the Code Green repository:

```git clone https://github.com/SPEAR-UIC/CodeGreen.git```

```cd CodeGreen```

## Step 3: Create a local Python virtual environemt

You should do the following steps in the same terminal opened in Step 2.

From inside the folder you cloned the CodeGreen repo, create a local Python virtual environment, e.g., called ".env", and install the dependencies:

```python3 -m venv .env```

```source .env/bin/activate```

```pip install --upgrade pip```

```pip install -r requirements.txt```

## Step 4: Test the Jacobi example

As a first step to practice with the power profile tool, you should execute the cells in the ```lassi_code_powerprofile.ipynb``` notebook.

When you are trying to refactor any one of the provided code samples, you will type in the application names (the same as the subfolders under ```test_codes```) and enter in the file name of the CUDA C++ source code into the notebook cell shown below. These file names are all of the format "appname"-cuda_main.cu.

![InputCodeFile](/images/LASSI_powerprofile_input_filefolder.png)

By default, the ```jacobi``` application is included in this cell for you to practice with. Also, this folder already contains an example *refactored* code for you to test alongside the original source code, which is provided as the ```refactorcode_filename``` value seen in the above screenshot.

Then, execute the subsequent two cells in the Jupyter notebook. These will compile and execute the source code and the refactored example code. The source code will compile and execute successfully. If your refactored code does not compile or execute, then you will need to debug your code, save, and try again. 

During execution of each code (the source code and the refactored code), the power profile will be measured and the energy consumption calculated. The results of these measurements will be presented to you for comparison. 

The goal is for the refactored code to **(1)** output the same functional computation as the source code while **(2)** using less power or consumpting less total energy during the runtime.

## Step 5: It's Your Turn! - Refactor Code for Energy Efficiency

Your goal is to start with any one of the source codes provided in the ```test_codes``` folder. Copy the "appname"-cuda_main.cu file as a new file in the *same* folder with a new name of your choice, e.g., "appname"-my_ee_refactor.cu. 

Edit your copy of the code as you see fit to make it more energy efficient. There are many strategies to try to achieve this goal--and you should learn about them starting with the Related Learning Documents below. 

When you are ready to test your code, return to the ```lassi_code_powerprofile.ipynb``` and replace the ```refactorcode_filename``` variable with your refactored code file name. Re-run the cells in the Jupyter notebook to see how your changes worked!

*Side note*: If you plan on doing code dev or testing outside of the provided Jupyter notebook environment and work directly from the terminal, then you will be responsible for being sufficiently familiar and finding your way during the Code Green Jam event. At a minimum, you will need to load the CUDA module with  ```module load cuda``` before compiling code.