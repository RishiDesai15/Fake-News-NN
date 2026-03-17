import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, dropout, pad_idx):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.lstm_title = nn.LSTM(embedding_dim,
                                  hidden_dim,
                                  batch_first=True,
                                  bidirectional=True)
        self.lstm_text  = nn.LSTM(embedding_dim,
                                  hidden_dim,
                                  batch_first=True,
                                  bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(4 * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, title, text):
        t_emb   = self.embedding(title)
        txt_emb = self.embedding(text)
        _, (h_t, _)   = self.lstm_title(t_emb)
        _, (h_txt, _) = self.lstm_text(txt_emb)
        h_t   = torch.cat((h_t[-2],   h_t[-1]),   dim=1)
        h_txt = torch.cat((h_txt[-2], h_txt[-1]), dim=1)
        h = torch.cat((h_t, h_txt), dim=1)
        return self.fc(h)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 num_filters, kernel_size, dropout, pad_idx):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(embedding_dim,
                               num_filters,
                               kernel_size)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(num_filters,
                               num_filters // 2,
                               kernel_size)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, text):
        x = self.embedding(text)            
        x = x.permute(0, 2, 1)              
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.global_pool(x).squeeze(2)
        return self.fc(x)
