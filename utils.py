import re
import csv
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import Iterable, List

import torch
from torch.utils.data import Dataset

# Increase field size limit for large CSV fields
csv.field_size_limit(10**6)

# Basic text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class FakeNewsDataset(Dataset):
    def __init__(self, csv_file, vocab=None,
                 max_seq_len=500, max_title_len=30,
                 build_vocab=False, max_vocab_size=None):
        # Load CSV using basic Python reader to avoid pandas/numpy issues
        rows = []
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('title') and row.get('text') and row.get('label'):
                    try:
                        label = int(row['label'])
                        rows.append({'title': row['title'], 'text': row['text'], 'label': label})
                    except (ValueError, TypeError):
                        continue
        
        self.max_seq_len   = max_seq_len
        self.max_title_len = max_title_len

        if build_vocab:
            texts = [row['text'] for row in rows] + [row['title'] for row in rows]
            self.vocab = self.build_vocab(texts, max_vocab_size=max_vocab_size)
        else:
            self.vocab = vocab
        self.pad_idx = self.vocab['<PAD>']

        self.data = []
        for row in rows:
            t_seq = self.text_to_seq(clean_text(row['text']),  max_seq_len)
            h_seq = self.text_to_seq(clean_text(row['title']), max_title_len)
            label = row['label']
            self.data.append((torch.tensor(h_seq),
                              torch.tensor(t_seq),
                              torch.tensor(label)))

    def build_vocab(self, texts, max_vocab_size=None):
        vocab = {'<PAD>':0, '<UNK>':1}
        for text in texts:
            for w in text.split():
                if w not in vocab:
                    vocab[w] = len(vocab)
                    if max_vocab_size and len(vocab) >= max_vocab_size:
                        return vocab
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


def fetch_rss_articles(feed_urls: Iterable[str], limit_per_feed: int = 25) -> List[dict]:
    articles = []
    seen = set()

    for url in feed_urls:
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                xml_data = response.read()
        except (urllib.error.URLError, ET.ParseError):
            continue

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            continue

        channel = root.find('channel')
        if channel is None:
            continue

        count = 0
        for item in channel.findall('item'):
            title = (item.findtext('title') or '').strip()
            description = (item.findtext('description') or '').strip()
            if not title or not description:
                continue

            key = (title.lower(), description.lower())
            if key in seen:
                continue
            seen.add(key)

            articles.append({
                'title': title,
                'text': description,
                'label': 0,
            })
            count += 1
            if count >= limit_per_feed:
                break

    return articles


def write_rows_csv(path: str, rows: List[dict]) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'text', 'label'])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'title': row.get('title', ''),
                'text': row.get('text', ''),
                'label': row.get('label', 0),
            })


def read_rows_csv(path: str) -> List[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
