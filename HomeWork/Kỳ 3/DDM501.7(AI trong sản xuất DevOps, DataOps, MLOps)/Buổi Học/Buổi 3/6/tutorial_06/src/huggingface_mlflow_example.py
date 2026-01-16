
import os
import mlflow
import mlflow.transformers
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


load_dotenv(dotenv_path=".env")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_metrics(eval_pred):
    """
        Compute metrics for evaluation
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def main():
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("huggingface-sentiment-analysis-tutorial")

    model_name = "distilbert-base-uncased"
    num_labels = 2
    max_length = 128
    batch_size = 32
    num_epochs = 3
    learning_rate = 2e-5

    print(f'Loading dataset...')
    dataset = load_dataset("imdb", split="train[:1000]")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f'Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                num_labels=num_labels)
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=max_length)
    print(f'Tokenizing dataset...')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics
    )
    with mlflow.start_run(run_name="huggingface-sentiment-analysis-tutorial"):
        mlflow.set_tags(
            {
                "owner": "dsteam",
                "algorithm": "huggingface",
                "dataset": "imdb",
                "version": "1",
            }
        )
        mlflow.log_params(
            {
                "model": model_name,
                "num_labels": num_labels,
                "max_length": max_length,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "device": str(device),
                "train_size": len(tokenized_datasets['train']),
                "test_size": len(tokenized_datasets['test']),
                "total_samples": len(dataset['train']) + len(dataset['test']),
            }
        )
        print(f'Starting training...')
        train_result = trainer.train()
        
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        })

        print('Evaluating...')
        eval_result = trainer.evaluate()

        mlflow.log_metrics({
            "eval_loss": eval_result['eval_loss'],
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_f1': eval_result['eval_f1'],
        })
        print('Training results...')
        print(f'Train Loss: {train_result.training_loss:.4f}')
        print(f"Eval Loss: {eval_result['eval_loss']:.4f}")
        print(f"Eval Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"Eval F1: {eval_result['eval_f1']:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        print('Saving model...')
        model_path = "./model"
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer,
            },
            artifact_path="model",
            registered_model_name="distilbert-sentiment"
        )
        print('Testing inference...')
        test_texts = ["I love this movie!", "This is the worst movie I've ever seen."]
        inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        print('Predictions:')
        for text, pred in zip(test_texts, predictions):
            sentiment = "positive" if pred == 1 else "negative"
            print(f'Text: {text}, Sentiment: {sentiment}')


if __name__ == "__main__":
    main()