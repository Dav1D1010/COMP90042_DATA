"""
Variant v9_wide: Tests the depth-vs-width hypothesis.
8L/512d, ~50M params, 500M pretraining tokens (10:1 ratio).
Same layers as v7 but wider hidden dimension — tests whether a
wider but shallower model captures more nuanced claim-evidence patterns.
Purpose: Compare depth (v8: 10L/384d) vs width (v9: 8L/512d).
Hypothesis: Wider model may better capture complex climate terminology
  at the cost of fewer layers for hierarchical reasoning.
Time on T4: ~5h pretraining + 0.5h fine-tuning = ~5.5h total
"""

import sys, os, time, math, json
from pathlib import Path
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_model import *

# ═══ V9: 8 layers, 512d — wider, testing depth-vs-width ═══
CONFIG = ModelConfig(d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=1024)
NUM_TOKENS = 500_000_000   # 10:1 ratio — slightly below optimal but testing width
BATCH_SIZE = 14             # Reduced for wider model (512d needs more VRAM)
LR = 3e-4
LABEL = "v9_wide"

# Tokenizer
class Tokenizer:
    PAD, UNK, BOS, EOS, SEP = 0, 1, 2, 3, 4
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor(model_file=model_path)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Climate change IPCC evidence claim supports refutes disputed global warming. " * 2000)
            spm.SentencePieceTrainer.train(input=f.name, model_prefix='/tmp/sp_v9', vocab_size=8000,
                model_type='bpe', pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=['<sep>'])
            self.sp = spm.SentencePieceProcessor(model_file='/tmp/sp_v9.model')
            os.unlink(f.name)
        self.vocab_size = self.sp.vocab_size()
        self.pad_token_id = self.PAD; self.bos_token_id = self.BOS; self.eos_token_id = self.EOS
    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos: ids = [self.BOS] + ids
        if add_eos: ids = ids + [self.EOS]
        return ids

tokenizer = Tokenizer()

# MLM Collator
class MLMCollator:
    def __init__(self, pad_id, mask_id, bos_id=2, eos_id=3, max_len=1024):
        self.pad_id, self.mask_id = pad_id, mask_id
        self.bos_id, self.eos_id = bos_id, eos_id
        self.max_len = max_len
    def __call__(self, batch):
        M = min(max(len(s) for s in batch), self.max_len)
        B = len(batch)
        input_ids = torch.full((B, M), self.pad_id, dtype=torch.long)
        labels = torch.full((B, M), -100, dtype=torch.long)
        mask = torch.zeros((B, M), dtype=torch.long)
        special = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        for i, seq in enumerate(batch):
            n = min(len(seq), M); t = torch.tensor(seq[:n])
            eligible = torch.tensor([x not in special for x in t.tolist()], dtype=torch.bool)
            if eligible.sum() == 0: input_ids[i,:n]=t; mask[i,:n]=1; continue
            n_mask = max(1, int(eligible.sum().item() * 0.15))
            idx = eligible.nonzero(as_tuple=False).squeeze(-1)
            mi = idx[torch.randperm(len(idx))[:n_mask]]
            mt = t.clone(); labels[i, mi] = t[mi]
            rng = torch.rand(n_mask)
            mt[mi[rng<0.8]] = self.mask_id
            rep = (rng>=0.8)&(rng<0.9)
            if rep.any(): mt[mi[rep]] = torch.randint(5, min(t.max()+1,8000), (rep.sum().item(),))
            input_ids[i,:n]=mt; mask[i,:n]=1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{LABEL}] Device: {device}")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", streaming=True, split="train")

class StreamDS(torch.utils.data.IterableDataset):
    def __init__(self, ds, tok, max_len=1024):
        self.ds, self.tok, self.max_len = ds, tok, max_len
    def __iter__(self):
        for ex in self.ds:
            yield self.tok.encode(ex["text"], add_bos=True, add_eos=True)[:self.max_len]
    def __len__(self): return 10_000_000

loader = DataLoader(StreamDS(ds, tokenizer, CONFIG.max_seq_len),
                    batch_size=BATCH_SIZE, collate_fn=MLMCollator(0, CONFIG.mask_token_id, 2, 3, CONFIG.max_seq_len))

model = ClimatronForPretraining(CONFIG).to(device)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda', enabled=True)

tps = BATCH_SIZE * CONFIG.max_seq_len
total_steps = max(1, NUM_TOKENS // tps)
warmup = max(1, int(total_steps * 0.05))
stable_end = int(total_steps * 0.90)
print(f"[{LABEL}] Model: {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"[{LABEL}] Token:Param ratio: {NUM_TOKENS/n_params:.1f}:1")
print(f"[{LABEL}] Pretraining {NUM_TOKENS:,} tokens, batch={BATCH_SIZE} (reduced for wider model)")

t0 = time.time(); step = 0; tokens = 0; running_loss = 0
for batch in loader:
    model.train()
    ids, lbls = batch["input_ids"].to(device), batch["labels"].to(device)
    with torch.amp.autocast('cuda', enabled=True):
        logits, _ = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lbls.view(-1), ignore_index=-100)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad()
    if step < warmup: lr = LR * step / warmup
    elif step < stable_end: lr = LR
    else: lr = LR * max(0, 1 - (step-stable_end)/(total_steps-stable_end))
    for pg in opt.param_groups: pg['lr'] = lr
    running_loss += loss.item(); tokens += tps; step += 1
    if step % 100 == 0:
        elapsed = time.time() - t0; speed = tokens/elapsed if elapsed>0 else 0
        print(f"  Step {step:>6d} | {100*tokens/NUM_TOKENS:5.1f}% | loss={running_loss/100:.4f} | {speed:,.0f} tok/s")
        running_loss = 0
    if tokens >= NUM_TOKENS: break

torch.save(model.state_dict(), f"{LABEL}_pretrain.pt")
elapsed = time.time() - t0
print(f"[{LABEL}] Pretraining done: {tokens:,} tokens in {timedelta(seconds=int(elapsed))} ({tokens/elapsed:,.0f} tok/s)")

# Fine-tuning
class_counts = torch.tensor([519., 199., 386., 124.])
loss_fn = StableImbalancedLoss(class_counts, ldam_margin=0.3, cb_beta=0.999, label_smoothing=0.1)

classifier = ClimatronForClassification(CONFIG, model).to(device)
classifier = apply_lora(classifier, r=8, alpha=16, dropout=0.1)
lora_params = [p for p in classifier.parameters() if p.requires_grad]

data_dir = Path(__file__).parent.parent.parent / "data"
try:
    with open(data_dir/"train-claims.json") as f: train_data = json.load(f)
    with open(data_dir/"dev-claims.json") as f: dev_data = json.load(f)
    with open(data_dir/"evidence.json") as f: evidence = json.load(f)
except: train_data, dev_data, evidence = {}, {}, {}

class ClfCollator:
    def __init__(self, tok, ev, max_len=512, max_ev=6):
        self.tok, self.ev, self.max_len, self.max_ev = tok, ev, max_len, max_ev
        self.lm = {"SUPPORTS":0,"REFUTES":1,"NOT_ENOUGH_INFO":2,"DISPUTED":3}
    def __call__(self, batch):
        ids_list, lbls = [], []
        for item in batch:
            ev_texts = [self.ev.get(eid,"") for eid in item.get("evidences",[])[:self.max_ev]]
            fmt = "<bos>" + "<sep>".join([item["claim_text"]]+ev_texts) + "<eos>"
            t = self.tok.encode(fmt); t = t[:self.max_len-1]+[self.tok.eos_token_id]
            ids_list.append(t); lbls.append(self.lm.get(item.get("claim_label","NOT_ENOUGH_INFO"),2))
        B,M = len(batch), min(max(len(s) for s in ids_list), self.max_len)
        ids = torch.full((B,M), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B,M), dtype=torch.long)
        for i,s in enumerate(ids_list): n=min(len(s),M); ids[i,:n]=torch.tensor(s[:n]); mask[i,:n]=1
        return {"input_ids":ids,"attention_mask":mask,"labels":torch.tensor(lbls)}

class ClfDS(torch.utils.data.Dataset):
    def __init__(self, claims): self.items = list(claims.items())
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        cid,d=self.items[i]
        return {"claim_id":cid,"claim_text":d["claim_text"],"claim_label":d.get("claim_label"),"evidences":d.get("evidences",[])}

c_train = DataLoader(ClfDS(train_data), batch_size=4, shuffle=True, collate_fn=ClfCollator(tokenizer, evidence))
c_dev = DataLoader(ClfDS(dev_data), batch_size=4, shuffle=False, collate_fn=ClfCollator(tokenizer, evidence))

opt_ft = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.1)
scaler_ft = torch.amp.GradScaler('cuda', enabled=True)
best_acc = 0; best_state = None

for epoch in range(5):
    classifier.train(); total_loss = 0
    for batch in c_train:
        ids=batch["input_ids"].to(device); am=batch["attention_mask"].to(device); targets=batch["labels"].to(device)
        with torch.amp.autocast('cuda', enabled=True):
            logits=classifier(ids,am); loss=loss_fn(logits, targets)
        scaler_ft.scale(loss).backward()
        scaler_ft.unscale_(opt_ft); torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler_ft.step(opt_ft); scaler_ft.update(); opt_ft.zero_grad()
        total_loss += loss.item()
    classifier.eval(); correct=total=0
    with torch.no_grad():
        for batch in c_dev:
            ids=batch["input_ids"].to(device); am=batch["attention_mask"].to(device); targets=batch["labels"].to(device)
            preds=classifier(ids,am).argmax(-1); correct+=(preds==targets).sum().item(); total+=targets.size(0)
    acc=correct/total if total>0 else 0
    print(f"[{LABEL}] Epoch {epoch+1}: loss={total_loss/len(c_train):.4f} val_acc={acc:.4f}")
    if acc>best_acc: best_acc=acc; best_state={k:v.cpu().clone() for k,v in classifier.state_dict().items()}

if best_state: classifier.load_state_dict(best_state)
torch.save(classifier.state_dict(), f"{LABEL}_classifier.pt")
print(f"[{LABEL}] Done. Best val_acc={best_acc:.4f}")
