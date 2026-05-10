import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Multi30kDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        try:
            from datasets import load_dataset
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"], check=True)
            from datasets import load_dataset
        raw = load_dataset("bentrevett/multi30k")
        split_map = {'train': 'train', 'val': 'validation', 'test': 'test'}
        self.data = raw[split_map[split]]

        self.de_nlp = spacy.load("de_core_news_sm")
        self.en_nlp = spacy.load("en_core_web_sm")

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_itos  = None
        self.tgt_itos  = None
        self.processed = None

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def build_vocab(self, de_sentences=None, en_sentences=None, min_freq=1):
        if de_sentences is None:
            de_sentences = [self.tokenize_de(item["de"]) for item in self.data]
        if en_sentences is None:
            en_sentences = [self.tokenize_en(item["en"]) for item in self.data]

        def make_vocab(tokenized):
            counter = Counter()
            for tokens in tokenized:
                counter.update(tokens)
            vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX,
                     SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX}
            for word, freq in counter.items():
                if freq >= min_freq and word not in vocab:
                    vocab[word] = len(vocab)
            itos = {idx: tok for tok, idx in vocab.items()}
            return vocab, itos

        self.src_vocab, self.src_itos = make_vocab(de_sentences)
        self.tgt_vocab, self.tgt_itos = make_vocab(en_sentences)

    def set_vocab(self, src_vocab, src_itos, tgt_vocab, tgt_itos):
        self.src_vocab = src_vocab
        self.src_itos  = src_itos
        self.tgt_vocab = tgt_vocab
        self.tgt_itos  = tgt_itos

    def process_data(self):
        assert self.src_vocab is not None, "Call build_vocab() or set_vocab() first"
        self.processed = []
        for item in self.data:
            de_tokens = self.tokenize_de(item["de"])
            en_tokens = self.tokenize_en(item["en"])
            src_ids = [SOS_IDX] + [self.src_vocab.get(t, UNK_IDX) for t in de_tokens] + [EOS_IDX]
            tgt_ids = [SOS_IDX] + [self.tgt_vocab.get(t, UNK_IDX) for t in en_tokens] + [EOS_IDX]
            self.processed.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(batch_size=64):
    from datasets import load_dataset
    print("Loading train split...")
    train_ds = Multi30kDataset(split='train')
    train_ds.build_vocab()
    train_ds.process_data()

    print("Loading val split...")
    val_ds = Multi30kDataset(split='val')
    val_ds.set_vocab(train_ds.src_vocab, train_ds.src_itos,
                     train_ds.tgt_vocab, train_ds.tgt_itos)
    val_ds.process_data()

    print("Loading test split...")
    test_ds = Multi30kDataset(split='test')
    test_ds.set_vocab(train_ds.src_vocab, train_ds.src_itos,
                      train_ds.tgt_vocab, train_ds.tgt_itos)
    test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, collate_fn=collate_fn)

    vocab_info = {
        "src_vocab": train_ds.src_vocab,
        "src_itos":  train_ds.src_itos,
        "tgt_vocab": train_ds.tgt_vocab,
        "tgt_itos":  train_ds.tgt_itos,
    }

    return train_loader, val_loader, test_loader, vocab_info


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab_info = get_dataloaders(batch_size=32)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")
    print(f"Src vocab size: {len(vocab_info['src_vocab'])}")
    print(f"Tgt vocab size: {len(vocab_info['tgt_vocab'])}")

    src, tgt = next(iter(train_loader))
    print(f"Sample src shape: {src.shape}")
    print(f"Sample tgt shape: {tgt.shape}")