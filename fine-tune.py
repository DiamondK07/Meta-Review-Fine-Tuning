from transformers import T5ForConditionalGeneration, T5TokenizerFast, Trainer, TrainingArguments, pipeline
import evaluate
from datasets import load_dataset
import zipfile
import os

# Define the function for preprocessing
def preprocess(examples):
    # Function to clean up input and output texts
    preprocess_input = lambda x: " ".join(x.strip().split()) if x else ""
    return {'inputs': [preprocess_input(doc) for doc in examples['Input']],
            'outputs': [preprocess_input(doc) for doc in examples['Output']]}

# Load and preprocess the dataset
dataset = load_dataset('zqz979/meta-review')
dataset = dataset.map(preprocess, batched=True)
dataset = dataset.filter(lambda x: x['inputs'] and x['outputs'])  # Remove rows with empty input/output

# Initialize the tokenizer and model
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Tokenize the dataset
def tokenize_function(examples):
    model_inputs = tokenizer(examples['inputs'], max_length=512, padding='max_length', truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['outputs'], max_length=150, padding='max_length', truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    evaluation_strategy="steps",
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('./results/final_model')