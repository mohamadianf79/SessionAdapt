"""Neural session baselines with the SAME interface/protocol as every other model.

  kind="gru"     : GRU4Rec-style  (embedding -> GRU -> linear over the catalogue), full-softmax cross-entropy.
  kind="sasrec"  : SASRec-style   (causal self-attention, tied item embeddings), full-softmax cross-entropy.

HONEST NOTES for the paper:
  * These are compact re-implementations trained with full cross-entropy (no negative sampling), i.e. the
    'CE loss' variants known to work well. They are NOT the official GRU4Rec code (BPR-max / top-k gain).
    State this. If a reviewer insists, run the official GRU4Rec_PyTorch_Official on the same split files.
  * A BOS token is prepended to every session, so the model is trained to predict the FIRST item too (n=0).
  * Right padding + causal mask => the output at position len-1 never sees padding.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Recommender
from ..eval.protocol import ranks_of_targets


class _GRUNet(nn.Module):
    def __init__(self, n_items, emb, hidden, dropout):
        super().__init__()
        self.n_items = n_items
        self.emb = nn.Embedding(n_items + 2, emb, padding_idx=n_items)   # n = PAD, n+1 = BOS
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_items)

    def hidden(self, x):                       # (B,L) -> (B,L,H)
        h, _ = self.gru(self.drop(self.emb(x)))
        return h

    def head(self, h):                         # (N,H) -> (N,n_items)
        return self.out(self.drop(h))


class _SASNet(nn.Module):
    def __init__(self, n_items, d, layers, heads, dropout, max_len):
        super().__init__()
        self.n_items, self.d = n_items, d
        self.emb = nn.Embedding(n_items + 2, d, padding_idx=n_items)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def hidden(self, x):
        B, L = x.shape
        h = self.emb(x) * (self.d ** 0.5) + self.pos(torch.arange(L, device=x.device))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        return self.norm(self.enc(self.drop(h), mask=causal))

    def head(self, h):
        return h @ self.emb.weight[: self.n_items].T


class NeuralRec(Recommender):
    def __init__(self, kind, hp, seed=0, max_epochs=30, patience=3, batch_size=256, max_len=50,
                 es_k=20, device=None, verbose=True):
        assert kind in ("gru", "sasrec")
        self.kind, self.hp, self.seed = kind, hp, seed
        self.max_epochs, self.patience, self.batch_size, self.max_len = max_epochs, patience, batch_size, max_len
        self.es_k, self.verbose = es_k, verbose
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = f"{kind}({hp})"

    def _build(self):
        n, hp = self.n_items, self.hp
        if self.kind == "gru":
            return _GRUNet(n, hp["emb"], hp["hidden"], hp["dropout"])
        return _SASNet(n, hp["d"], hp["layers"], hp["heads"], hp["dropout"], self.max_len)

    def _train_tensors(self, batch):
        seqs = []
        for s in batch:
            q = np.concatenate([[self.bos], s])[-(self.max_len + 1):]
            seqs.append(q)
        L = max(len(q) for q in seqs) - 1
        x = np.full((len(seqs), L), self.pad, dtype=np.int64)
        y = np.full((len(seqs), L), -100, dtype=np.int64)
        for i, q in enumerate(seqs):
            m = len(q) - 1
            x[i, :m] = q[:-1]; y[i, :m] = q[1:]
        return torch.from_numpy(x).to(self.dev), torch.from_numpy(y).to(self.dev)

    def fit(self, train_items, n_items, train_ts=None, val_inst=None):
        self.n_items, self.pad, self.bos = n_items, n_items, n_items + 1
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = self._build().to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"])
        seqs = [np.asarray(s) for s in train_items if len(s) >= 1]
        best_mrr, best_state, bad, ep = -1.0, None, 0, 0
        for ep in range(self.max_epochs):
            self.net.train()
            perm = rng.permutation(len(seqs))
            tot, nb = 0.0, 0
            for a in range(0, len(perm), self.batch_size):
                x, y = self._train_tensors([seqs[i] for i in perm[a:a + self.batch_size]])
                h = self.net.hidden(x)
                mask = y != -100
                loss = F.cross_entropy(self.net.head(h[mask]), y[mask])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += loss.item(); nb += 1
            if val_inst is None:
                continue
            r = ranks_of_targets(self, val_inst)
            mrr = float(np.where(r <= self.es_k, 1.0 / r, 0.0).mean())
            if self.verbose:
                print(f"  [{self.kind}] epoch {ep+1} loss={tot/nb:.4f} val MRR@{self.es_k}={mrr:.4f}", flush=True)
            if mrr > best_mrr:
                best_mrr, best_state, bad = mrr, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val, self.epochs_run = best_mrr, ep + 1
        return self

    @torch.no_grad()
    def score(self, prefixes, ts=None):
        self.net.eval()
        seqs = [np.concatenate([[self.bos], np.asarray(p, dtype=np.int64)])[-self.max_len:] for p in prefixes]
        L = max(len(q) for q in seqs)
        x = np.full((len(seqs), L), self.pad, dtype=np.int64)
        lens = np.array([len(q) for q in seqs])
        for i, q in enumerate(seqs):
            x[i, :len(q)] = q
        h = self.net.hidden(torch.from_numpy(x).to(self.dev))
        last = h[torch.arange(len(seqs), device=self.dev), torch.from_numpy(lens - 1).to(self.dev)]
        return self.net.head(last).float().cpu().numpy().astype(np.float64)
