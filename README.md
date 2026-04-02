# Fake-News-NN
This project implements a robust pipeline for classifying news articles as real or fake using deep learning in PyTorch. You’ll find two complementary model architectures:
 - Bidirectional LSTM (Bi-LSTM) that leverages both the article headline and full text to capture contextual dependencies in both forward and backward directions.
 - Convolutional Neural Network (CNN) with 1D convolutions and max-pooling to detect n-gram patterns indicative of misinformation.

Key features include:
 - Data Handling: Clean, tokenize, and pad raw news data drawn from the Kaggle Fake News dataset (train.csv), automatically mapping labels and splitting into stratified train/validation sets.
 - Vocabulary & Embeddings: Build a custom word-index vocabulary and initialize an embedding layer (e.g. GloVe or learned embeddings) to transform text into dense vector representations.
 - Hyperparameter Control: Command-line flags for model choice, embedding size, hidden dimensions/number of filters, dropout rate, sequence lengths, batch size, number of epochs, learning rate, and data sampling fraction for quick prototyping.
 - Training Utilities: Real-time batch-level loss reporting, checkpointing each epoch, and easy resume/retraining on full or subsampled data.
- Real-News Augmentation: Optional RSS-based augmentation pulls in public real-news headlines and summaries from Reuters and NPR to reduce fake-biased predictions.
- Threshold Calibration: Training saves an inference threshold so the model only labels an article as fake when the fake probability is strong enough.
 - Inference Server: A lightweight Flask app that loads the best checkpoint, preprocesses incoming JSON payloads (title + text), and returns live predictions (real or fake with confidence score) via a /predict endpoint.

To retrain with the real-news augmentation and calibrated threshold:

```bash
python train.py \
	--model bilstm \
	--data_file data/train.csv \
	--epochs 10 \
	--batch_size 32 \
	--augment_real_web
```
