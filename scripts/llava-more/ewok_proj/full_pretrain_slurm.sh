#!/bin/bash

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export OMP_NUM_THREADS=1

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"

epochs=1
model_path=meta-llama/Llama-3.1-8B-Instruct # this variable indicate the path of the used language model
images_path=  #  root folder of where images reside (after extraction from images.zip)
data_train_path=  #  path to chat.json
vision_tower=openai/clip-vit-large-patch14-336

job_name="your/job/name"
nnodes=1
echo "job name: $job_name"
export TOKENIZER_PATH=$model_path

torchrun \
--nnodes=$nnodes --nproc-per-node=1 --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$job_name --rdzv-backend=c10d \
./src/llava/train/train_mem.py \
--model_name_or_path $model_path \
--model_architecture llama_3_1 \
--version llama_3_1 \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--mm_projector_type mlp2x_gelu \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir \
--num_train_epochs $epochs \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 128 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 100 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to none