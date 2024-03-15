#!/bin/bash
#SBATCH -o ./out-t.out # STDOUT
#SBATCH -e ./err-t.err # STDERR
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node 2
#SBATCH --mem 256GB
#SBATCH -N 1
#SBATCH -p fvl
#SBATCH -t 2-00:00:00
#SBATCH -q high
#SBATCH -J coda

sh experiments/cifar-100.sh
# sh experiments/imagenet-r.sh
# sh experiments/domainnet.sh
