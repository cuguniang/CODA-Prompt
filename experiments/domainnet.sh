# bash experiments/domainnet.sh
# experiment settings
DATASET=DomainNet
N_CLASS=345

# save directory
OUTDIR=/share/ckpt/yhzhou/coda/${DATASET}/10-task-panda

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/domainnet_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# my-prompt
#
# prompt parameter args:
#    arg 1 = prompt location [input/attention]
#    arg 2 = deep prompt [True/False]
#    arg 3 = shared [True/False] only when deep work
#    arg 4 = prompt length per task

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name MyPrompt \
    --prompt_param input 1 \
    --log_dir ${OUTDIR}/my-p


# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.0 \
#     --log_dir ${OUTDIR}/coda-p

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 5 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p++