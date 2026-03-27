#!/bin/bash

# Default values
NAME=navdp_h200_1gpu
MODEL=navdp_h200_1gpu
DEBUG_FLAG=""

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
        --debug)
            DEBUG_FLAG="--debug"
            shift
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
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "cma")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "cma_plus")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "seq2seq")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "seq2seq_plus")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "navdp")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "navdp_h200_1gpu")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
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

if [[ "$MODEL" == "navdp" || "$MODEL" == "navdp_h200_1gpu" ]]; then
    echo "Using torchrun to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        $DEBUG_FLAG
else
    echo "Using python to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    python scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        $DEBUG_FLAG
fi
