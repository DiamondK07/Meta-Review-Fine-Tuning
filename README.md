# Meta-Review Fine-Tuning Project

This repository contains the code and instructions for fine-tuning the T5 model using a dataset of meta reviews. The objective is to train a model that can summarize paper meta reviews effectively.

## Files Overview

- **fine-tune.py**: The main script for loading, preprocessing, tokenizing, and fine-tuning the T5 model on the dataset.
- **Rouge.py**: Evaluates the fine-tuned model using the ROUGE metric for performance measurement.
- **summarize.py**: Script for using the fine-tuned model to generate summaries of meta reviews.
- **run_ft.sh**: Shell script to run the fine-tuning and summarization pipeline using Conda.
- **requirements.txt**: List of Python dependencies required for running the scripts.
- **logs**: Contains logs from the training and summarization process.

## Procedure

### 1. Set Up the Environment

Ensure that all dependencies are installed:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

The dataset `zqz979/meta-review` from Hugging Face is used. It contains paper meta reviews that are being used for training the T5 model. The dataset is preprocessed to clean and normalize the text inputs.

### 3. Hyperparameter Rationale

- **Batch Size** (`per_device_train_batch_size=8` and `per_device_eval_batch_size=16`): These values were chosen based on GPU memory limitations while ensuring efficient training steps.
- **Epochs** (`num_train_epochs=3`): Three epochs provide a balance between model performance and training time.
- **Learning Rate** (5e-5): Chose a lower learning rate to avoid overfitting
- **Warmup Steps** (`warmup_steps=500`): Slowly increased the learning rate to stabilize training in initial steps.
- **Weight Decay** (0.01): Regularization parameter to prevent overfitting and improve model generalization.

### 4. Prompt Design Rationale

The prompt used for generating summaries is designed to guide the model explicitly:

```
"Summarize the following paper meta review in a concise and informative manner.\n"
"Focus on the key points, strengths, and weaknesses mentioned in the review, and provide an overview that captures "
"the essence of the analysis. Ensure the summary highlights the core contributions and any major critiques.\n\n"
"Meta Review: {meta_review_text}\nSummary:"
```

This prompt ensures that the model understands the structure of the input and focuses on summarizing it effectively by emphasizing strengths, weaknesses, and overall contributions.

### 5. Running the Fine-Tuning Script

To run the fine-tuning process:

```bash
python fine-tune.py
```

### 6. Evaluating the Model

The `Rouge.py` script evaluates the performance of the fine-tuned model using the ROUGE metric:

```bash
python Rouge.py
```

### 7. Summarization Pipeline

To generate summaries using the fine-tuned model, run:

```bash
python summarize.py
```


## Requirements

- transformers
- datasets
- torch
- sentencepiece
- numpy
- pandas
- scikit-learn
- tqdm
- evaluate