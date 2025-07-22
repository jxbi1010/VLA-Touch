export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export NCCL_SOCKET_IFNAME=eno1
#export NCCL_IB_DISABLE=1

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt_ckpt/rdt-finetune-1b-mango-rgb"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/home/allenbi/PycharmProjects24/cutlass"

export WANDB_PROJECT="RDT_finetune_mahjong"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

#deepspeed --hostfile=hostfile.txt launch main.py \

accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=4 \
    --sample_batch_size=4 \
    --max_train_steps=40000 \
    --checkpointing_period=1000 \
    --sample_period=1000 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --gradient_accumulation_steps=4 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=wandb \
    --use_8bit_adam \
    --resume_from_checkpoint="./checkpoints/rdt_ckpt/rdt-finetune-1b-mango-rgb/checkpoint-20000"
    #    --pretrained_model_name_or_path="./checkpoints/rdt_ckpt/rdt-finetune-1b-watercup-rgb/checkpoint-20000"