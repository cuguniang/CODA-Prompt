# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory
OUTDIR=/share/ckpt/yhzhou/coda/${DATASET}-10-task/0315-rd-lr0.01

# hard coded inputs
GPUID='0 1'
CONFIG=configs/cifar-100_prompt.yaml
CONFIG_FT=configs/cifar-100_ft.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# my-prompt
#
# prompt parameter args:
#    arg 1 = prompt location [input/attention]
#   x arg 2 = deep prompt [1/0]
#   x arg 3 = shared [1/0] only when deep work
#    arg 4 = prompt length per task

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name MyPrompt \
    --prompt_param 1 \
    --log_dir ${OUTDIR}/my-p

rm -rf ${OUTDIR}/my-p/models

# CODA-P

# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --log_dir ${OUTDIR}/coda-p

# rm -rf ${OUTDIR}/coda-p/models

# DualPrompt

# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt

# rm -rf ${OUTDIR}/dual-prompt/models

# L2P++

# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p++

# rm -rf ${OUTDIR}/l2p++/models


# FT++
# python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type default --learner_name FinetunePlus \
#     --log_dir ${OUTDIR}/ft++

# FT
# python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type default --learner_name Nor`ma`lNN \
#     --log_dir ${OUTDIR}/ft

# Offline
# python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type default --learner_name NormalNN --upper_bound_flag \
#     --log_dir ${OUTDIR}/offline-upper_bound
