#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=4:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=8GB               # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END         # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=logs/test_slurm_%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/test_slurm_%j.err    # Set name of error log. %j is the Slurm jobId
#SBATCH --gres=gpu


/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

#Remaining job commands go below here. For example, to run a Matlab script named "matlab_script.m", uncomment:
#module use /opt/insy/modulefiles # Use DAIC INSY software collection
#module load matlab/R2020b        # Load Matlab 2020b version
#srun matlab < matlab_script.m # Computations should be started with 'srun'.

# apptainer exec --nv apptainer/dp2.sif bash -c "cd datasets_AR/deep_privacy2 && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE && python anonymize.py configs/anonymizers/face.py -i ../RWF2000/ --output_path ../RWF2000_face_gan"

apptainer exec --nv apptainer/dp2.sif bash -c "cd datasets_AR/deep_privacy2 && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE && python anonymize.py configs/anonymizers/FB_cse.py -i ../RWF2000/ --output_path ../RWF2000_body_gan"