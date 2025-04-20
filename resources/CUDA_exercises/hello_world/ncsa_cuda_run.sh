#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=CUDAExample # Set the job name to "CUDAExample"
#SBATCH --time=00:30:00        # Set the wall clock limit to 30 minutes
#SBATCH --ntasks=1             # Request 1 task (core)
#SBATCH --mem=10G              # Request 10GB per node
#SBATCH --partition=gpuA40x4        # Request the GPU partition/queue
#SBATCH --gres=gpu:a40:4      # Request a node with 1 A40 GPU's
#SBATCH --account=bbvi-delta-gpu 

##OPTIONAL JOB SPECIFICATIONS

#SBATCH --mail-type=ALL        # Send email on all job events
#SBATCH --output=stdout.%j     # Send stdout to “stdout.[jobID]"
#SBATCH --error=stderr.%j      # Send stderr to “stderr.[jobID]"
##SBATCH --mail-user=email     # Send all emails to email

# load required module(s)
module load cuda

# run your program
./a.out
