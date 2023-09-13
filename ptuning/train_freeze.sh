LR=1e-4
RUN_TIME=$(date "+%Y-%m-%d-%H:%M:%S")
NUM_GPUS=2

CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /home/qianq/data/text2text_train/train_final_add.json \
    --validation_file /home/qianq/data/text2text_train/train_final_add.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/qianq/model/chatglm2-6b \
    --output_dir output/train-freeze-$LR-$RUN_TIME \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --unfreeze_layers 1
    # --quantization_bit 4

