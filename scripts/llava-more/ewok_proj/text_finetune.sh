#!/bin/bash

# source activate more
# cd local/path

# export PYTHONPATH=.
# export WANDB_ENTITY=project_entity
# export WANDB_PROJECT=project_name
# export WANDB_MODE=offline

# IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
# export OMP_NUM_THREADS=1

# echo "CPUs: $SLURM_CPUS_PER_TASK"
# echo "GPUs: $SLURM_GPUS_PER_NODE"
# echo "MASTER ADDR: ${MASTER_ADDR}"
# echo "MASTER PORT: ${MASTER_PORT}"

epochs=1
model_path=meta-llama/Llama-3.1-8B-Instruct # this variable indicate the path of the used language model
# images_path=local/path
data_train_path=/home/alongon/data/ewok/llava_instruct_80k.json
train_text_path=/home/alongon/data/ewok/llava_instruct_80k_text.json
output_dir=/home/alongon/model_weights/ewok/llama3.1_8B_instruct_textonly
# vision_tower=local/path
# mm_projector_path=local/path

# job_name="your/job/name"
# nnodes=<number_of_nodes>
# echo "job name: $job_name"
export TOKENIZER_PATH=$model_path

# CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun \
--nnodes=1 --nproc-per-node=1 \
./src/llava/train/train_mem.py \
--model_name_or_path $model_path \
--model_architecture llama_3_1 \
--version llama_3_1 \
--llm_pad_token pad \
--data_path $data_train_path \
--text_path $train_text_path \
--group_by_modality_length True \
--bf16 True \
--optim adamw_8bit \
--output_dir $output_dir \
--num_train_epochs $epochs \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 128 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 100 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 0 \
--lazy_preprocess True \
--report_to none