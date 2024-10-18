from transformers import T5ForConditionalGeneration, T5TokenizerFast, Trainer, TrainingArguments, pipeline
import evaluate
from datasets import load_dataset
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
# Initialize the summarization pipeline with the fine-tuned model
summarizer = pipeline("summarization", model="./results/final_model", tokenizer=tokenizer, device=0)

# Load the evaluation dataset
eval_dataset = load_dataset('zqz979/meta-review', split='validation')
eval_dataset = eval_dataset.filter(lambda x: x['Input'] is not None and x['Output'] is not None and len(x['Input'].strip()) > 0)

# Generate summaries
generated_summaries = []
for article in eval_dataset['Input']:
    summary = summarizer(article, max_length=150, min_length=40, truncation=True)
    generated_summaries.append(summary[0]['summary_text'])

# Load the ROUGE metric using the evaluate library
rouge = evaluate.load('rouge')

# Compute ROUGE scores
results = rouge.compute(predictions=generated_summaries, references=eval_dataset['Output'])

# Print ROUGE Scores
print("ROUGE Scores:", results)
