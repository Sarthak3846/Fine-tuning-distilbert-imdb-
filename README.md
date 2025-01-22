# Sentiment Analysis with Hugging Face and TensorFlow.
This repository contains code to fine-tune a pre-trained model (distilbert-base-uncased) for sentiment analysis on the IMDb dataset using the Hugging Face transformers library and TensorFlow.
In this project, we fine-tune a pre-trained DistilBERT model on the IMDb dataset for binary sentiment classification (positive/negative sentiment). The model is trained to classify movie reviews as positive or negative based on their text content.

# Libraries Used
Hugging Face Transformers: For loading pre-trained models and tokenizers.
TensorFlow: For model training and evaluation.
Datasets: To load the IMDb dataset.

# how it works 
Load Pre-trained Model: We load a pre-trained distilbert-base-uncased model using the Hugging Face transformers library.
Dataset Preprocessing: We preprocess the IMDb dataset by tokenizing the text data using the pre-trained tokenizer.
Model Compilation: The model is compiled with the Adam optimizer, SparseCategoricalCrossentropy loss, and accuracy metrics.
Fine-Tuning: The model is fine-tuned using the training dataset, and performance is validated using the evaluation dataset.
Evaluation: After training, the model's performance on the test dataset is evaluated and printed.
