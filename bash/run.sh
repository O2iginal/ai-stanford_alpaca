cd "$(dirname "$0")"/..

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=4869 train.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B" \
    --data_path ./alpaca_data_1k.json \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --tf32 False # --tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7