import os
import sys
import time
# import wandb
import numpy as np
from evaluate import load
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, set_seed,
    AutoTokenizer, EvalPrediction, Trainer, TrainingArguments)

# WANDB_API_KEY = "74139a5dfde92426ae83d3e008e3b666476fc20e"
MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
METRIC_NAME = 'accuracy'
OUTPUT_DIR = r'G:\My Drive\ANLP\ex1\models'


def get_preprocess_func(tokenizer):
    def preprocess_function(examples):
        # Tokenize the texts:
        # When max_length is not specified, the default value is the
        # longest input value the model can accept, as required
        result = tokenizer(examples['sentence'], truncation=True)
        return result

    return preprocess_function


def get_compute_metrics_func(metric):
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)[METRIC_NAME]
        return {METRIC_NAME: result}

    return compute_metrics


def find_best_model(all_results):
    accuracies = {}
    best_accuracy = 0
    best_model = best_trainer = best_model_name = None

    for model in MODELS:
        mean_accuracy = np.mean([result[METRIC_NAME] for result in all_results[model]])
        std_accuracy = np.std([result[METRIC_NAME] for result in all_results[model]])
        accuracies[model] = {'mean': mean_accuracy, 'std': std_accuracy}
        if mean_accuracy > best_accuracy:
            best_model_name = model

    for index, result in enumerate(all_results[best_model_name]):
        if result[METRIC_NAME] > best_accuracy:
            best_accuracy = result[METRIC_NAME]
            best_trainer = result['trainer']
            best_model = result['model']

    return accuracies, best_model, best_model_name, best_trainer


def create_files(accuracies, end_time, predictions, start_time, test_dataset, total_train_time):
    # Save Predictions in predictions.txt file:
    with open(os.path.join(OUTPUT_DIR, 'predictions.txt'), 'w') as f:
        for index, prediction in enumerate(predictions):
            f.write(f'{test_dataset[index]["sentence"]}###{prediction}\n')

    # Create res.txt file:
    with open(os.path.join(OUTPUT_DIR, 'res.txt'), 'w') as f:
        for model in MODELS:
            f.write(f'{model},{accuracies[model]["mean"]} +- {accuracies[model]["std"]}\n')
        f.write("----\n")
        f.write(f'train time,{total_train_time}\n')
        f.write(f'predict time,{(end_time - start_time)}\n')


def main():

    # wandb.login()

    # 1. Load arguments:

    num_seeds = int(sys.argv[1])
    train_num_samples = int(sys.argv[2])
    val_num_samples = int(sys.argv[3])
    test_num_samples = int(sys.argv[4])

    # 2. Load dataset:

    raw_datasets = load_dataset('sst2')

    raw_train_dataset = raw_datasets['train'].select(range(train_num_samples)) \
        if train_num_samples > -1 else raw_datasets['train']
    raw_eval_dataset = raw_datasets['validation'].select(range(val_num_samples)) \
        if val_num_samples > -1 else raw_datasets['validation']
    raw_test_dataset = raw_datasets['test'].select(range(test_num_samples)) \
        if test_num_samples > -1 else raw_datasets['test']

    # 3. Load model and tokenizer:

    all_results = {}
    total_train_time = 0

    for model_name in MODELS:

        all_results[model_name] = []

        for seed in range(num_seeds):
            set_seed(seed)

            name = f"{model_name.replace('/', '-')}_{seed}"
            path = os.path.join(OUTPUT_DIR, name)
            config = AutoConfig.from_pretrained(model_name)
            # wandb.init(project="ANLP-ex1", dir=OUTPUT_DIR, config=config, name=name)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 4. Tokenize dataset:

            eval_dataset = raw_eval_dataset.map(get_preprocess_func(tokenizer), batched=True)
            train_dataset = raw_train_dataset.map(get_preprocess_func(tokenizer), batched=True)

            # 5. Define metrics for evaluation:

            metric = load(METRIC_NAME)

            # 6. Train:

            training_args = TrainingArguments(output_dir=path,
                                              # report_to='wandb',
                                              run_name=name,
                                              save_strategy='no')

            # check if model is already trained
            is_trained = os.path.exists(os.path.join(path, 'pytorch_model.bin'))

            model = AutoModelForSequenceClassification.from_pretrained(path if is_trained else model_name,
                                                                       config=config, cache_dir="./cache")

            trainer = Trainer(model=model, args=training_args,
                              train_dataset=train_dataset, eval_dataset=eval_dataset,
                              compute_metrics=get_compute_metrics_func(metric),
                              tokenizer=tokenizer)
            if not is_trained:
                train_result = trainer.train()
                train_time = train_result.metrics['train_runtime']
                with open(os.path.join(path, 'train_time.txt'), 'w') as f:
                    f.write(str(float(train_time)))

                trainer.save_model(output_dir=path)
            else:
                with open(os.path.join(path, 'train_time.txt'), 'r') as f:
                    train_time = float(f.read())

            total_train_time += train_time

            # 7. Evaluate:

            model.eval()
            eval_res = trainer.evaluate(eval_dataset=eval_dataset)

            all_results[model_name].append({METRIC_NAME: eval_res['eval_accuracy'],
                                            'trainer': trainer, 'model': model})

            # wandb.finish()

    # 8. Find best model: (the model with the highest mean accuracy on validation set)

    accuracies, best_model, best_model_name, best_trainer = find_best_model(all_results)

    # 9. Predict:

    best_tokenizer = AutoTokenizer.from_pretrained(best_model_name)
    test_dataset = raw_test_dataset.map(get_preprocess_func(best_tokenizer), batched=True)

    # Removing the `label` columns because it contains -1 and Trainer won't like that.
    test_dataset = test_dataset.remove_columns("label")

    # Run model.eval() before prediction
    best_model.eval()

    start_time = time.time()
    predictions = []
    for i in range(len(test_dataset)):
        sample = test_dataset.select(range(i, i + 1))
        predictions.append(best_trainer.predict(sample).predictions)

    # Another option was to use the tokenizer to tokenize each sample from test_dataset
    # and then use the model to predict its label (prediction = model(**tokenized_sample))

    predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1)
    end_time = time.time()

    create_files(accuracies, end_time, predictions, start_time, test_dataset, total_train_time)


if __name__ == '__main__':
    main()
