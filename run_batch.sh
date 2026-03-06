#!/usr/bin/bash

# Parameters
#SBATCH --account=drf@a100
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=log_recon
#SBATCH --ntasks=1
#SBATCH --distribution=block:block
#SBATCH --error=%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=%A_%a_0_log.out
#SBATCH --partition=a100
#SBATCH --qos=normal@a100
#SBATCH --signal=USR2@120
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --wckey=submitit
#SBATCH --array=0-1

#SBATCH -L fs_store,fs_work

#SBATCH --cpus-per-task=32


set -x
cd $WORK/Codes/wcrr-noncartesian-3d-mri

module purge
module load cuda/13
source $WORK/Environments/bench/bin/activate
export WANDB_MODE=offline
cit=0
group=1
folder=tune_again 
for method in drunet #ncpdnet wv tv wcrr 
do
	for vid in 0 1
	do
		if [ $((ctr/group)) -eq $SLURM_ARRAY_TASK_ID ]
		then
			python prospective_tuning.py --root $SCRATCH/DATA/Benchmark_Networks --simulation 0 --method $method --volume_id $vid  --compress_coil -1 --folder $folder --init sense 
		fi
		ctr=$((ctr+1))
	done
done
wait


