import pandas as pd 
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer

def tokenize_function(example):
    return (example)

df = pd.read_csv("erraticana-offtopic-raw.csv")
df = df[df["Author"] == "bonepriest#6318"]
df = df[df["Content"].str.strip().str.contains(r"\s", regex=True, na=False) == True]

bone_dataset = Dataset.from_pandas(df)
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_dataset = bone_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()