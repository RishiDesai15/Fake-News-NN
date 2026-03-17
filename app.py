import os
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import FakeNewsDataset, clean_text
from models import BiLSTM

app = Flask(__name__)
CORS(app)

# 1) Use the processed split (label column exists)
TRAIN_CSV     = "train_split.csv"            # <- changed
CHECKPOINT    = "checkpoints/bilstm_e1.pt"   # or your best checkpoint
EMBEDDING_DIM = 100
HIDDEN_DIM    = 128
DROPOUT       = 0.5

# 2) Build vocab from the TRAIN split (which has title,text,label)
dummy_ds = FakeNewsDataset(TRAIN_CSV, build_vocab=True)
vocab    = dummy_ds.vocab
pad_idx  = vocab['<PAD>']

# 3) Load the model
model = BiLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
    pad_idx=pad_idx
)
model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
model.eval()

# 4) Inference helper
def preprocess(title: str, text: str):
    t = clean_text(title)
    x = clean_text(text)
    h_seq = dummy_ds.text_to_seq(t, dummy_ds.max_title_len)
    t_seq = dummy_ds.text_to_seq(x, dummy_ds.max_seq_len)
    return (
      torch.tensor([h_seq], dtype=torch.long),
      torch.tensor([t_seq], dtype=torch.long)
    )

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('.', 'test.html')

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.json or {}
    title = data.get('title', '')
    text  = data.get('text', '')
    h, x  = preprocess(title, text)
    with torch.no_grad():
        logits = model(h, x)
        probs  = F.softmax(logits, dim=1).squeeze().tolist()
    label = "fake" if probs[1] > probs[0] else "real"
    score = probs[1] if label=="fake" else probs[0]
    return jsonify(label=label, score=score)

if __name__ == '__main__':
    # Default to a non-privileged port; override with PORT env var when needed.
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')))
