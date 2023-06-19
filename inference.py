import os
import argparse
import torch
from optimum.neuron import NeuronModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert-base-uncased-finetuned-sst-2-english',
                        help='The model to use for inference: can be a model ID, or the path to a local directory.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='The static batch size of the Neuron model.')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='The static sequence length of the Neuron model.')
    parser.add_argument('--save-dir', type=str,
                        help='The path to save the Neuron model.')
    args = parser.parse_args()

    if os.path.exists(args.model):
        # This is a model that has already been exported to Neuron, just load it with its tokenizer
        model = NeuronModelForSequenceClassification.from_pretrained(args.model)
        # Sanity checks
        assert args.batch_size == model.config.neuron_batch_size
        assert args.seq_length == model.config.neuron_sequence_length
    else:
        model = NeuronModelForSequenceClassification.from_pretrained(args.model, export=True, batch_size=1, sequence_length=args.seq_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.save_dir:
        model.save_pretrained(os.path.join(args.save_dir))

    # create embeddings for inputs
    inputs = "To be, or not to be: that is the question: Whether it is nobler in the mind" \
             " to suffer the slings and arrows of outrageous fortune, Or to take arms against a sea of troubles"
    embeddings = tokenizer(inputs, return_tensors="pt")
    # convert to tuple for neuron model
    neuron_inputs = tuple(embeddings.values())

    # run prediction
    with torch.no_grad():
        logits = model(*neuron_inputs)[0]
        cls_id = logits.argmax().item()
        print(model.config.id2label[cls_id])


if __name__ == "__main__":
    main()