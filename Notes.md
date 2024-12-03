## USE CASES:
1. Editting and masking, e.g. adobe premiere
2. Other research, 3D reconstr. segementing out background
3. Animal Behavior
4. Robotics

## TO DO 8/28 -->
- Set up Git Repo
- Try my own videos
- Try on YT videos, maybe trying Edited videos? Cut videos? Virtual Videos (eg slideshow)? Rotating or Zooming videos?
- Videos where objects instantly change from first frame

- Sort IMGS by 0 leading
- Changes in Lighting
- Long form jump cut video of tech review


## TO DO 9/2 -->
1. Perm select frames to add to long term
2. Try back Prop on jump editted videos 

## TO DO 9/30 -->
Make videos of the success and failure cases

## tmux --> srun --> python
srun --job-name=“atekkey” --account=bdnb-delta-gpu --time=1:00:00 --partition=gpuA100x4 --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=4 --mem=140g --exclusive --no-kill --gpu-bind=closest --pty /bin/bash -i

watch -n 5 nvidia-smi

top in terminal
free -h ,, Memory

Created new envirn, anaconda

-------------------------------------------------------------

# Get data thru wget
wget "URLHERE"
tar -xvf FNAMEHERE.tar

# Get data from Gdrive:
pip install gdown
gdown FILEID

## RUN
sbatch slurm_batch.sh

# VIEW
squeue --job JOBID


# Imgs to video
ffmpeg -framerate 4 -i %d.png -c:v mpeg4 -r 30 phone_out.mp4

# Get thru Youtube
yt-dlp -f mp4 https://www.youtube.com/watch?v=Y46mvjmMhHM



### 1917 MOVIE
# Test
Time Expended for 538 frames: 202.6713 seconds
# For BackProp over all Misses
Time Expended for 538 frames: 222.4424 seconds
# Backprop: (1.1) * time
# Worst case: O(2n)


python /work/hdd/bdnb/atekkey/sam2/tools/vos_inference.py --base_video_dir "/work/hdd/bdnb/vpurushotham/datasets/LVOS/valid/JPEGImages" --input_mask_dir "/work/hdd/bdnb/vpurushotham/datasets/LVOS/valid/Annotations" --output_mask_dir "/work/hdd/bdnb/atekkey/sam2/notebooks/results/SAM2-Reg2"
