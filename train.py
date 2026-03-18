import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import FakeNewsDataset
from models import BiLSTM, CNN


def evaluate(model, loader, criterion, device, model_name):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for title, text, label in loader:
            title, text, label = title.to(device), text.to(device), label.to(device)
            out = model(title, text) if model_name == 'bilstm' else model(text)
            loss = criterion(out, label)
            total_loss += loss.item() * label.size(0)

            preds = torch.argmax(out, dim=1)
            y_true.extend(label.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    p_real, r_real, f1_real, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average='binary', zero_division=0
    )

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision_macro': p_macro,
        'recall_macro': r_macro,
        'f1_macro': f1_macro,
        'precision_real': p_real,
        'recall_real': r_real,
        'f1_real': f1_real,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['bilstm','cnn'], required=True)
    p.add_argument('--data_file', type=str, required=True)
    p.add_argument('--sample_frac', type=float, default=1.0,
                   help="Fraction of CSV to sample for quick runs")
    p.add_argument('--embedding_dim', type=int, default=100)
    p.add_argument('--hidden_dim',    type=int, default=128)
    p.add_argument('--num_filters',   type=int, default=128)
    p.add_argument('--kernel_size',   type=int, default=5)
    p.add_argument('--dropout',       type=float, default=0.5)
    p.add_argument('--batch_size',    type=int, default=32)
    p.add_argument('--epochs',        type=int, default=5)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--max_seq_len',   type=int, default=100,
                   help="Max tokens in body")
    p.add_argument('--max_title_len', type=int, default=20,
                   help="Max tokens in title")
    p.add_argument('--checkpoint_dir',type=str, default='checkpoints')
    p.add_argument('--patience',      type=int, default=3,
                   help="Early stopping patience based on val macro-F1")
    args = p.parse_args()

    # 1) Load and sample
    df = pd.read_csv(args.data_file)
    df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
    if 'type' in df.columns:
        df = df.rename(columns={'type':'label'})
    elif 'label' not in df.columns:
        raise KeyError("CSV must contain either a 'type' or 'label' column")

    # 2) Map categories to 0/1
    fake_types = {'fake','bs','satire','junksci','conspiracy','hate'}

    def map_label(v):
        if str(v).strip() in {'0', '1'}:
            return int(v)
        return 1 if str(v).strip().lower() in fake_types else 0

    df['label'] = df['label'].apply(map_label)
    df = df[['title','text','label']].dropna()

    # 3) Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv',   index=False)

    # 4) Datasets
    train_ds = FakeNewsDataset(
        'train_split.csv', build_vocab=True,
        max_seq_len=args.max_seq_len,
        max_title_len=args.max_title_len
    )
    vocab, pad_idx = train_ds.vocab, train_ds.vocab['<PAD>']
    val_ds = FakeNewsDataset(
        'val_split.csv', vocab=vocab,
        max_seq_len=args.max_seq_len,
        max_title_len=args.max_title_len
    )

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=args.batch_size)

    # 5) Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model=='bilstm':
        model = BiLSTM(len(vocab), args.embedding_dim,
                       args.hidden_dim, args.dropout, pad_idx)
    else:
        model = CNN(len(vocab), args.embedding_dim,
                    args.num_filters, args.kernel_size,
                    args.dropout, pad_idx)
    model.to(device)

    class_counts = train_df['label'].value_counts().to_dict()
    total = len(train_df)
    # Inverse-frequency weights reduce bias toward the dominant class.
    w0 = total / (2.0 * class_counts.get(0, 1))
    w1 = total / (2.0 * class_counts.get(1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt  = optim.Adam(model.parameters(), lr=args.learning_rate)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(args.checkpoint_dir, 'best_model.pt')
    best_f1 = -1.0
    epochs_no_improve = 0

    # 6) Train
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        seen = 0

        for i, (title, text, label) in enumerate(tl, 1):
            title, text, label = title.to(device), text.to(device), label.to(device)
            opt.zero_grad()
            out = model(title, text) if args.model=='bilstm' else model(text)
            loss = crit(out, label)
            loss.backward()
            opt.step()

            running_loss += loss.item() * label.size(0)
            seen += label.size(0)
            if i % 50 == 0:
                avg = running_loss / max(seen, 1)
                print(f"[Epoch {epoch}] Batch {i}/{len(tl)}  Loss: {avg:.4f}")
                running_loss = 0.0
                seen = 0

        # Save checkpoint each epoch
        ckpt = os.path.join(args.checkpoint_dir, f"{args.model}_e{epoch}.pt")
        torch.save(model.state_dict(), ckpt)

        val_metrics = evaluate(model, vl, crit, device, args.model)
        print(
            f"[Epoch {epoch}] "
            f"ValLoss: {val_metrics['loss']:.4f}  "
            f"Acc: {val_metrics['accuracy']:.4f}  "
            f"MacroF1: {val_metrics['f1_macro']:.4f}  "
            f"RealRecall: {val_metrics['recall_real']:.4f}"
        )

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"✅ New best checkpoint saved: {best_ckpt} (MacroF1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"⏹ Early stopping at epoch {epoch} "
                    f"(no macro-F1 improvement for {args.patience} epochs)."
                )
                break

    print(f"✅ Training complete. Best model: {best_ckpt} (MacroF1={best_f1:.4f})")

if __name__ == '__main__':
    main()
 
#To Train:  
""" 
 python train.py \
  --model bilstm \
  --data_file data/train.csv \
  --sample_frac 1.0 \
  --max_seq_len 500 \
  --max_title_len 30 \
  --epochs 10 \
  --batch_size 32
"""

# To Run:
"""
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=80
"""

#In another terminal run"
"""
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "title":"City Council Approves New Community Garden",
        "text":"The city council voted unanimously yesterday to fund the construction of a new community garden set to open this spring, offering free workshops and local produce stands."
      }'
"""