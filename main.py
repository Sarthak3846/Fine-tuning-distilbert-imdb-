from transformers import TFAutoModelForSequenceClassification, AutoTokenizer 
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam 

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("imdb")

def tokenizer_function(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenizer_function, batched = True)

train_dataset = tokenized_dataset["train"].shuffle(seed=42).batch(16)
eval_dataset = tokenized_dataset["test"].batch(16)

model.compile(
    optimizer=Adam(learning_rate = 5e-5),
    loss = model.compute_loss,
    metrics = ["accuracy"]
    )

model.fit(
    train_dataset,
    epochs=3,
    validation_data = eval_dataset,
    verbose=1
)

eval_results = model.evaluate(eval_dataset)
print("Test loss: ", eval_results[0])
print("Test Accuracy: ", eval_results[1])

