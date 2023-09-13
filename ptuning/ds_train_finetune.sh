
LR=1e-4
NUM_GPUS=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


deepspeed --num_gpus=$NUM_GPUS --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /home/qianq/data/text2text_train/train_final_add.json \
    --validation_file /home/qianq/data/text2text_train/train_final_add.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/qianq/model/chatglm2-6b \
    --output_dir output/train-finetune-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16

