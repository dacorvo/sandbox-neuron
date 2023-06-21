#!/bin/sh

set -e

model=distilbert-base-uncased-finetuned-sst-2-english
dataset=sst2
batch_size=1
seq_length=128

for auto_cast in matmul all; do
    for auto_cast_type in bf16 fp16; do
        # Convert the model
        neuron_path="$(pwd)/${model}-${auto_cast}-${auto_cast_type}"
        optimum-cli export neuron -m $model          \
                    --batch_size $batch_size         \
                    --sequence_length $seq_length    \
                    --auto_cast $auto_cast           \
                    --auto_cast_type $auto_cast_type \
                    $neuron_path
        # Run inference
        echo "Inference results for ${auto_cast}-${auto_cast_type}"
        python inference.py --model $neuron_path             \
                            --dataset $dataset               \
                            --input-column sentence          \
                            --metric accuracy                \
                            --batch-size $batch_size         \
                            --seq-length $seq_length
    done
done