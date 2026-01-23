import json
from torch.utils.data import DataLoader, Dataset, dataloader
from functools import partial
import torch
from collections import defaultdict

def get_dataloader(train_path, entity2emb, relation2emb, entity2id, relation2id,  query_len, batch_size, max_seq_len, max_seq_len_src, emb_dim, mode):
    dataset = KGDataset(train_path, entity2emb, relation2emb, entity2id, relation2id,  query_len, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle=(mode == 'train'),
        collate_fn=partial(KGDataset.collate_pad,
                           cutoff=max_seq_len,
                           cutoff_src=max_seq_len_src,
                           padding_token=0,
                           emb_dim=emb_dim
        ))

    while True:
        for batch in dataloader:
            yield batch

class KGDataset(Dataset):
    def __init__(self, train_path, entity2emb, relation2emb, entity2id, relation2id,  query_len, mode="train"):
        self.train_path = train_path
        self.entity2emb = entity2emb
        self.relation2emb = relation2emb
        with open(entity2id, 'r', encoding='utf-8') as f:
            self.entity_size = sum(1 for line in f if line.strip())
        with open(relation2id, 'r', encoding='utf-8') as f:
            self.relation_size = sum(1 for line in f if line.strip())
        self.mode = mode
        self.query_len = query_len
        self.read_data()

    def read_data(self):
        data = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip().split('\t'))
        with open(self.entity2emb, 'r', encoding='utf-8') as f:
            embs = json.load(f)
            self.entity2target = {i: emb for i, emb in enumerate(embs)}
        with open(self.relation2emb, 'r', encoding='utf-8') as f:
            embs = json.load(f)
            self.relation2target = {i: emb for i, emb in enumerate(embs)}
        Data = []
        self.target = []
        for p in data:
            self.target.append(self.entity2target[int(p[2])])
            p[0] = ('entity', int(p[0]))
            p[1] = ('relation', int(p[1]))
            p[2] = ('entity', int(p[2]))
            p[3] = ('ts', int(p[3]))
            Data.append(p)

        if self.query_len > 1:
            groups = defaultdict(list)
            self.data = []
            for idx, parts in enumerate(Data):
                if len(parts) < 4:
                    continue
                try:
                    head = parts[0][1]
                    time = parts[-1][1]
                    groups[head].append((idx, parts, time))
                except Exception:
                    continue

            for head, records in groups.items():
                records.sort(key=lambda x: x[2])
                for ridx, (orig_idx, parts, time) in enumerate(records):
                    history = records[max(0, ridx - self.query_len):ridx]
                    temp_data = []
                    temp_data.extend([parts[0], parts[1], parts[-1]])
                    for _, h_parts, _ in reversed(history):
                        temp_data.extend(h_parts)
                    self.data.append(temp_data)
        else:
            self.data = [row[:2] + row[3:] for row in Data]

    def __len__(self):
        return len(self.data)

    def get_data(self, a):
        data = []
        for name, id in a:
            if name == 'entity':
                data.append(self.entity2target[id])
            elif name == 'relation':
                data.append(self.relation2target[id])
        return data

    def __getitem__(self, idx):
        out_dict = { "encoder_input_ids": self.get_data(self.data[idx]), "decoder_input_ids": self.target[idx], }
        return out_dict

    @staticmethod
    def collate_pad(batch, cutoff: int, cutoff_src: int, padding_token: int, emb_dim: int):
        num_elems = len(batch)

        # -------------------------
        # encoder: padding
        # -------------------------
        encoder_list = []
        encoder_mask_list = []

        # -------------------------
        # decoder: padding 不做
        # -------------------------
        decoder_list = []
        decoder_mask_list = []

        for i in range(num_elems):
            # Get source tokens (encoder)
            toks_src = batch[i]["encoder_input_ids"]  # Shape (src_len, emb_dim)
            src_len = len(toks_src)

            # If src_len < cutoff_src, pad; else, truncate to cutoff_src
            if src_len < cutoff_src:
                # Pad with zeros to match cutoff_src length
                padded_src = torch.nn.functional.pad(torch.FloatTensor(toks_src), (0, 0, 0, cutoff_src - src_len),
                                                     value=0.0)
            else:
                # Truncate to cutoff_src
                padded_src = torch.FloatTensor(toks_src[:cutoff_src])

            encoder_list.append(padded_src)
            encoder_mask_list.append(torch.ones(cutoff_src).long())  # Mask for encoder (all ones)

            # Get target tokens (decoder)
            toks_tgt = batch[i]["decoder_input_ids"]
            l = len(toks_tgt)
            decoder_list.append(torch.nn.functional.pad(
                torch.FloatTensor(toks_tgt), (0, emb_dim - l), value=0.0
            ))
            decoder_mask_list.append(torch.ones(cutoff).long())  # Mask for decoder (all ones)

        # Stack all encoder sequences
        encoder_input_ids = torch.stack(encoder_list, dim=0)  # Shape (batch_size, cutoff_src, emb_dim)
        encoder_attention_mask = torch.stack(encoder_mask_list, dim=0)  # Shape (batch_size, cutoff_src)

        # Stack all decoder sequences
        decoder_input_ids = torch.stack(decoder_list, dim=0)  # Shape (batch_size, cutoff, emb_dim)
        decoder_input_ids = decoder_input_ids.unsqueeze(1)
        decoder_input_ids = decoder_input_ids.repeat(1, cutoff, 1)
        decoder_attention_mask = torch.stack(decoder_mask_list, dim=0)  # Shape (batch_size, cutoff)

        return {
            "input_ids": encoder_input_ids,  # (batch_size, cutoff_src, emb_dim)
            "attention_mask": encoder_attention_mask,  # (batch_size, cutoff_src)
            "decoder_input_ids": decoder_input_ids,  # (batch_size, cutoff, emb_dim)
            "decoder_attention_mask": decoder_attention_mask,  # (batch_size, cutoff)
        }, None