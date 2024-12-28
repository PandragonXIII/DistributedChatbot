cd ./dependency/LLaMA-Factory
# DISTRIBUTED_ARGS="
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
#   "
DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes 1 \
    "

MODEL_PATH="E:\research\models\qwen2.5-1.5b-instruct"
OUTPUT_PATH="FT-output/qwen2.5-1.5b-instruct"
DATASET="webnovel_cn_subset"

torchrun $DISTRIBUTED_ARGS dependency/LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 1024 \
    --save_steps 500 \
    --plot_loss \
    --num_train_epochs 1 \
    --bf16