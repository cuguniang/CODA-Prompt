#!/bin/bash
#SBATCH -o ./out-test.out # STDOUT
#SBATCH -e ./err-t.err # STDERR
#SBATCH --cpus-per-task 2
#SBATCH --gpus-per-node 2
#SBATCH --mem 128GB
#SBATCH -N 1
#SBATCH -p fvl
#SBATCH -t 2-00:00:00
#SBATCH -q medium
#SBATCH -J coda

sh experiments/cifar-100.sh
# sh experiments/imagenet-r.sh
# sh experiments/domainnet.sh
