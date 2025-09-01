#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours) medium is longer
#SBATCH --time=4:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=16GB               # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --gres=gpu:a40:1        
#SBATCH --mail-type=ALL         # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=logs/gemma/out/gemma_slurm_%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/gemma/err/gemma_slurm_%j.err    # Set name of error log. %j is the Slurm jobId

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working

#Remaining job commands go below here. For example, to run a Matlab script named "matlab_script.m", uncomment:
#module use /opt/insy/modulefiles # Use DAIC INSY software collection
#module load matlab/R2020b        # Load Matlab 2020b version
#srun matlab < matlab_script.m # Computations should be started with 'srun'.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name UCF_Crime_Privacy --video_root ../datasets_AR/UCF_videos_privacy/ --experiment No_Few_Shot"
# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma-3n --dataset_name UCF_Crime --video_root ../datasets_AR/UCF_videos_temporal/ --experiment Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name UCF_Crime --video_root ../datasets_AR/UCF_videos_temporal/ --experiment Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name UCF_Crime_Unguided --video_root ../datasets_AR/UCF_videos_temporal/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name UCF_Crime --video_root ../datasets_AR/UCF_videos_temporal/ --experiment No_Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name RWF2000_Privacy --video_root ../datasets_AR/RWF2000_privacy --experiment No_Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name RWF2000_Privacy_face_gan --video_root ../datasets_AR/RWF2000_face_gan --experiment No_Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name RWF2000_Privacy_body_gan --video_root ../datasets_AR/RWF2000_body_gan --experiment No_Few_Shot"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name UCF_Crime --video_root ../datasets_AR/UCF_videos_temporal/ --experiment No_Few_Shot --eval_json logs/UCF_Crime/No_Few_Shot/Gemma3-4B/eval_20250809-112209.json"

# apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models&& python apply_videollm.py --model_type gemma --dataset_name UCF_Crime --video_root ../datasets_AR/UCF_videos_temporal/ --experiment Few_Shot --eval_json logs/UCF_Crime/Few_Shot/Gemma3-4B/eval_20250810-110352.json"

apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models && python apply_videollm.py --model_type gemma --dataset_name Anomaly         --video_root ../datasets_AR/synthetic_dataset/anomaly_videos         --experiment merged"
apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models && python apply_videollm.py --model_type gemma --dataset_name Spatial         --video_root ../datasets_AR/synthetic_dataset/spatial_videos         --experiment merged"
apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models && python apply_videollm.py --model_type gemma --dataset_name Spatial_Colored --video_root ../datasets_AR/synthetic_dataset/spatial_colored_videos --experiment merged"
apptainer exec --nv apptainer/hf_models_final3_cuda122.sif bash -c "cd models && python apply_videollm.py --model_type gemma --dataset_name Following      --video_root ../datasets_AR/synthetic_dataset/follow_videos        --experiment merged"
