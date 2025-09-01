#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=4:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=4   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=16GB               # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --gres=gpu:l40:1        
#SBATCH --mail-type=END         # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=logs/nvila/out/nvila_slurm_%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/nvila/err/nvila_slurm_%j.err    # Set name of error log. %j is the Slurm jobId

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

#Remaining job commands go below here. For example, to run a Matlab script named "matlab_script.m", uncomment:
#module use /opt/insy/modulefiles # Use DAIC INSY software collection
#module load matlab/R2020b        # Load Matlab 2020b version
#srun matlab < matlab_script.m # Computations should be started with 'srun'.


# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --video_root ../datasets_AR/UCF_videos_temporal/ --eval_json logs/UCF_Crime/NVILA-8B-Video/eval_20250521-095450.json"
# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --video_root ../datasets_AR/UCF_videos_temporal/ --eval_json logs/UCF_Crime/NVILA-8B-Video/eval_20250523-103659.json"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name UCF_Crime --video_root ../../datasets_AR/UCF_videos_temporal/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name UCF_Crime_Privacy --video_root ../../datasets_AR/UCF_videos_privacy/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name RWF2000 --video_root ../../datasets_AR/RWF2000/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name RWF2000_Depth --video_root ../../datasets_AR/RWF2000_depth/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name UCF_Crime --video_root ../../datasets_AR/UCF_videos_temporal/ --experiment Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name UCF_Crime_Unguided --video_root ../../datasets_AR/UCF_videos_temporal/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name XD_Violence --video_root ../../datasets_AR/XD_Violence/videos/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name RWF2000_Masked --video_root ../../datasets_AR/RWF2000_masked2/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name RWF2000_Privacy --video_root ../../datasets_AR/RWF2000_privacy --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name RWF2000_Privacy_face_gan --video_root ../../datasets_AR/RWF2000_face_gan --experiment No_Few_Shot"

# apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python apply_vila.py --dataset_name RWF2000_Privacy_body_gan --video_root ../../datasets_AR/RWF2000_body_gan --experiment No_Few_Shot"

apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python full_predict_vila.py --dataset_name Anomaly         --video_root ../../datasets_AR/synthetic_dataset/anomaly_videos         --experiment merged"
apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python full_predict_vila.py --dataset_name Spatial         --video_root ../../datasets_AR/synthetic_dataset/spatial_videos         --experiment merged"
apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python full_predict_vila.py --dataset_name Spatial_Colored --video_root ../../datasets_AR/synthetic_dataset/spatial_colored_videos --experiment merged"
apptainer exec --nv apptainer/nvila_cuda122.sif bash -c "cd models/VILA && python full_predict_vila.py --dataset_name Following      --video_root ../../datasets_AR/synthetic_dataset/follow_videos        --experiment merged"
