#!/bin/bash

# Default values
NAME=navdp_1node_lr
MODEL=navdp_1node
IL_OVERRIDES='{}'  # JSON override, e.g. '{"lr_scheduler_type":"cosine","lr_eta_min":1e-6}'

# IL_OVERRIDES examples:
#   '{"lr_scheduler_type":"linear","lr_end_factor":0.5,"lr_total_iters":10000}'
#   '{"lr_scheduler_type":"linear","lr_end_factor":0.1,"lr_total_iters":30000}'  # 더 길게 decay (30k, 10배 감소)
#   '{"lr_scheduler_type":"cosine","lr_eta_min":1e-6}'                           # cosine (전체 step에 걸쳐 1e-6까지)
#   '{"lr":5e-5,"lr_scheduler_type":"cosine","lr_eta_min":1e-7}'                 # lr 자체 + cosine

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --il-overrides)
            IL_OVERRIDES="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set GPU devices and NUM_GPUS
case $MODEL in
    "rdp")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "cma")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "cma_plus")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "seq2seq")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "seq2seq_plus")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "navdp")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "navdp_1node")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "navdp_1node_lr")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    *)
        echo "Error: Unsupported model type: $MODEL"
        exit 1
        ;;
esac

# Check if NUM_GPUS is set
if [[ -z $NUM_GPUS ]]; then
    echo "Error: NUM_GPUS is not set"
    exit 1
fi


export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if model is navdp to use torchrun, otherwise use python
if [[ "$MODEL" == "navdp" || "$MODEL" == "navdp_1node" || "$MODEL" == "navdp_1node_lr" ]]; then
    echo "Using torchrun to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    echo "IL overrides: $IL_OVERRIDES"
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        --il-overrides "$IL_OVERRIDES"
else
    echo "Using python to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    python scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        --il-overrides "$IL_OVERRIDES"
fi
