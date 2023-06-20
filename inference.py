import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import pipeline
from evaluate import evaluator
from optimum.neuron import NeuronModelForSequenceClassification
from optimum.exporters import TasksManager
from transformers import AutoTokenizer, AutoConfig


def read_model_id(model_dir):
    with open(os.path.join(model_dir, 'config.json')) as f:
        return json.load(f)['_name_or_path']


def get_neuron_model_class(model_id):
    # Infer task from the Hub model id
    task = TasksManager.infer_task_from_model(model_id)
    # Get AutoModel class for task
    auto_model_class = TasksManager.get_model_class_for_task(task)
    neuron_classes = [NeuronModelForSequenceClassification]
    for neuron_class in neuron_classes:
        if neuron_class.auto_model_class == auto_model_class:
            return neuron_class, task
    raise ValueError(f"{model_id} is associated to {task} which is not supported for Neuron (yet).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for inference: can be a model ID, or the path to a local directory.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset on which the model should be evaluated.')
    parser.add_argument('--metric', type=str, required=True,
                        help='The metric used to evaluate the model.')
    parser.add_argument('--num-samples', type=int,
                        help='The number of samples to use for evaluation.')
    parser.add_argument('--input-column', type=str, default='sentence',
                        help='The name of the input column in the selected dataset.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='The static batch size of the Neuron model.')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='The static sequence length of the Neuron model.')
    parser.add_argument('--save-dir', type=str,
                        help='The path to save the Neuron model.')
    args = parser.parse_args()

    if os.path.exists(args.model):
        # Extract original model Hub name from the config (Too brittle ?)
        model_id = read_model_id(args.model)
        # Get the corresponding Neuron class
        neuron_model_class, task = get_neuron_model_class(model_id)
        model = neuron_model_class.from_pretrained(args.model)
        # Sanity checks
        assert args.batch_size == model.config.neuron_batch_size
        assert args.seq_length == model.config.neuron_sequence_length
    else:
        # Get the Neuron class for the specified model_id
        neuron_model_class, task = get_neuron_model_class(args.model)
        model = neuron_model_class.from_pretrained(args.model,
                                                   export=True,
                                                   batch_size=args.batch_size,
                                                   sequence_length=args.seq_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.save_dir:
        model.save_pretrained(os.path.join(args.save_dir))

    # Load dataset
    datasets = load_dataset(args.dataset)
    eval_dataset = datasets["validation"] if "validation" in datasets else datasets["test"]
    if args.num_samples:
        eval_dataset = eval_dataset.select(range(args.num_samples))

    # Create evaluation pipeline
    eval_pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_length=args.seq_length,
        padding="max_length",
        truncation=True
    )

    # Instantiate evaluator
    task_evaluator = evaluator(task)
    metric = task_evaluator.compute(model_or_pipeline=eval_pipe,
                                    data=eval_dataset,
                                    metric=args.metric,
                                    input_column=args.input_column,
                                    label_mapping=model.config.label2id)
    print(metric)


if __name__ == "__main__":
    main()