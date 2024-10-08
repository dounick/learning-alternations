from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("qing-yao/babylm-baseline")
tokenizer = AutoTokenizer.from_pretrained("src/training/tokenizer")

# Load the evaluation dataset
# Make sure to specify the correct format. Here we're assuming a text dataset.
eval_dataset = load_dataset("text", data_files={"test": "data/corpora/babylm/test.txt"}, split="test")
# Tokenize the evaluation dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Apply the tokenizer to the dataset
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch (or TensorFlow if you prefer)
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./models/babylm-baseline/results",  # Directory to save results
    per_device_eval_batch_size=8,  # Adjust batch size as needed
    evaluation_strategy="steps",  # You can set this to "epoch" if you prefer
    eval_steps=100,  # Evaluate every 100 steps, adjust as needed
    logging_dir='./models/babylm-baseline/logs',  # Directory to store logs
    remove_unused_columns=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,  # Pass the training arguments
    eval_dataset=eval_dataset,  # Pass the evaluation dataset
    tokenizer=tokenizer,  # Pass the tokenizer
)

# Perform evaluation
trainer.evaluate()
