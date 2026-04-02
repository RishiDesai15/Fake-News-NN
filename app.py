import os
import json
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
CHECKPOINT    = "checkpoints/best_model.pt"
INFERENCE_CFG  = "checkpoints/inference_config.json"
EMBEDDING_DIM = 100
HIDDEN_DIM    = 128
DROPOUT       = 0.5

# 2) Load checkpoint to get expected vocab size
checkpoint_state = torch.load(CHECKPOINT, map_location='cpu')
expected_vocab_size = checkpoint_state['embedding.weight'].shape[0]

# 3) Build vocab from the TRAIN split, then limit to checkpoint size
dummy_ds = FakeNewsDataset(TRAIN_CSV, build_vocab=True, max_vocab_size=expected_vocab_size)
vocab    = dummy_ds.vocab
pad_idx  = vocab['<PAD>']

# 4) Load the model
model = BiLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
    pad_idx=pad_idx
)
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}. Train first to create best_model.pt."
    )
model.load_state_dict(checkpoint_state)
model.eval()

fake_threshold = 0.5
if os.path.exists(INFERENCE_CFG):
    with open(INFERENCE_CFG, 'r', encoding='utf-8') as f:
        config = json.load(f)
    fake_threshold = float(config.get('fake_threshold', fake_threshold))

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
    fake_score = probs[1]
    real_score = probs[0]
    label = "fake" if fake_score >= fake_threshold else "real"
    score = fake_score if label == "fake" else real_score
    return jsonify(label=label, score=score, fake_score=fake_score, real_score=real_score, threshold=fake_threshold)

if __name__ == '__main__':
    # Default to a non-privileged port; override with PORT env var when needed.
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')))
