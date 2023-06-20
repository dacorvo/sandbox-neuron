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
            return neuron_class, task, auto_model_class
    raise ValueError(f"{model_id} is associated to {task} which is not supported for Neuron (yet).")


def evaluate_model(model, tokenizer, max_length, task, dataset, input_column, metric):
    # Create evaluation pipeline
    eval_pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

    # Instantiate evaluator
    task_evaluator = evaluator(task)
    return task_evaluator.compute(model_or_pipeline=eval_pipe,
                                  data=dataset,
                                  metric=metric,
                                  input_column=input_column,
                                  label_mapping=model.config.label2id)


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
    parser.add_argument('--input-column', type=str, default='text',
                        help='The name of the input column in the selected dataset.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='The static batch size of the Neuron model.')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='The static sequence length of the Neuron model.')
    parser.add_argument('--save-dir', type=str,
                        help='The path to save the Neuron model.')
    args = parser.parse_args()

    # Load dataset
    datasets = load_dataset(args.dataset)
    eval_dataset = datasets["validation"] if "validation" in datasets else datasets["test"]
    if args.num_samples:
        eval_dataset = eval_dataset.select(range(args.num_samples))

    # Instantiate or load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if os.path.exists(args.model):
        # Extract original model Hub name from the config (Too brittle ?)
        model_id = read_model_id(args.model)
        # Get the corresponding Neuron class
        neuron_model_class, task, _ = get_neuron_model_class(model_id)
        neuron_model = neuron_model_class.from_pretrained(args.model)
        # Sanity checks
        assert args.batch_size == model.config.neuron_batch_size
        assert args.seq_length == model.config.neuron_sequence_length
        model = None
    else:
        # Get the Neuron class for the specified model_id
        neuron_model_class, task, auto_model_class = get_neuron_model_class(args.model)
        # Instantiate the Hub model
        model = auto_model_class.from_pretrained(args.model)
        # Instantiate the Neuron model
        neuron_model = neuron_model_class.from_pretrained(args.model,
                                                          export=True,
                                                          batch_size=args.batch_size,
                                                          sequence_length=args.seq_length)
    if args.save_dir:
        model.save_pretrained(os.path.join(args.save_dir))

    # Evaluate the Neuron model
    neuron_metric = evaluate_model(neuron_model, tokenizer, args.seq_length, task, eval_dataset, args.input_column, args.metric)

    if model is not None:
        # Evaluate also the Hub model
        metric = evaluate_model(model, tokenizer, args.seq_length, task, eval_dataset, args.input_column, args.metric)

    # Display results as a table
    heading = "{:<30}".format('')
    if model is not None:
        heading +="{:^20}".format('Hub')
    heading +="{:^20}".format('Neuron')
    print(heading)
    for key in neuron_metric:
        row = f"{key:<30}"
        if metric is not None:
            row += f"{metric[key]:>20,.4f}"
        row += f"{neuron_metric[key]:>20,.4f}"
        print(row)

if __name__ == "__main__":
    main()