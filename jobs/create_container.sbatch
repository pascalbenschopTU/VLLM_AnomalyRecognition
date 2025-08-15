#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=3:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=16   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=32GB               # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead        
#SBATCH --mail-type=END         # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=logs/cc/create_container_%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/cc/create_container_%j.err    # Set name of error log. %j is the Slurm jobId

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

#Remaining job commands go below here. For example, to run a Matlab script named "matlab_script.m", uncomment:
#module use /opt/insy/modulefiles # Use DAIC INSY software collection
#module load matlab/R2020b        # Load Matlab 2020b version
#srun matlab < matlab_script.m # Computations should be started with 'srun'.

# apptainer build --disable-cache apptainer/videollama_cuda122.sif apptainer/hf_models_cuda122.def
# apptainer build --disable-cache apptainer/yolo_cuda122.sif apptainer/bias_pred_cuda122.def

# apptainer build --disable-cache apptainer/sentence_transform.sif apptainer/sentence_transform.def
apptainer build --disable-cache apptainer/dp2.sif apptainer/dp22.def


# apptainer build --disable-cache apptainer/videollama_cuda122.sif apptainer/videollama_cuda122.def
