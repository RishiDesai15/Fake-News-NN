import re
import pandas as pd
import torch
from torch.utils.data import Dataset

# Basic text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class FakeNewsDataset(Dataset):
    def __init__(self, csv_file, vocab=None,
                 max_seq_len=500, max_title_len=30,
                 build_vocab=False):
        df = pd.read_csv(csv_file)
        df['title'] = df['title'].fillna('').apply(clean_text)
        df['text']  = df['text'].fillna('').apply(clean_text)
        df = df[['title','text','label']].dropna()

        self.max_seq_len   = max_seq_len
        self.max_title_len = max_title_len

        if build_vocab:
            texts = df['text'].tolist() + df['title'].tolist()
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab
        self.pad_idx = self.vocab['<PAD>']

        self.data = []
        for _, row in df.iterrows():
            t_seq = self.text_to_seq(row['text'],  max_seq_len)
            h_seq = self.text_to_seq(row['title'], max_title_len)
            label = int(row['label'])
            self.data.append((torch.tensor(h_seq),
                              torch.tensor(t_seq),
                              torch.tensor(label)))

    def build_vocab(self, texts):
        vocab = {'<PAD>':0, '<UNK>':1}
        for text in texts:
            for w in text.split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        return vocab

    def text_to_seq(self, text, max_len):
        seq = [self.vocab.get(w, self.vocab['<UNK>'])
               for w in text.split()]
        if len(seq) < max_len:
            seq += [self.pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
