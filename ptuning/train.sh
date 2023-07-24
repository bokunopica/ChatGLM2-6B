PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=2
RUN_TIME=$(date "+%Y-%m-%d %H:%M:%S")

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /home/qianq/data/text2text_train/train_final_add.json \
    --validation_file /home/qianq/data/text2text_train/train_final_add.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/qianq/model/chatglm2-6b \
    --output_dir output/train-$PRE_SEQ_LEN-$LR-$RUN_TIME \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 5000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4

