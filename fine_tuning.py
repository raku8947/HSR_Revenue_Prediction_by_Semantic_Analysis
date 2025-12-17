"""
功能概述：
对BERT模型进行无监督领域适应训练，将其转化为适用于 B 站评论语境的模型。

使用方法：运行：
python fine_tuning.py ./data/all_comments_merged.csv

输入：./data/all_comments_merged.csv (preprocess得到的评论文本)
输出：output/ (保存各个阶段训练好的模型权重)

脚本执行流程：

1. 【词表调整】
   - 扫描评论数据，统计高频 B 站表情包（如 [doge]、[笑哭]）。
   - 将这些表情作为新 token 加入 tokenizer，并初始化权重，避免被切分成无关字符。

2. 【MLM 训练 (Masked Language Modeling)】
   - 在评论数据上继续预训练 BERT，让模型适应 B 站的口语和游戏术语。
   - 包含每轮 Epoch 结束后的简单的 Loss/PPL 评估。

3. 【TSDAE 训练 (无监督句向量)】
   - Phase-1：冻结 BERT Encoder，只训练 Decoder 的 Cross-Attention，做初步对齐。
   - Phase-2：解冻所有参数，进行全量训练。
   - 目的：将 BERT 从字词模型转化为句向量模型。

4. 【SimCSE 训练 (对比学习)】
   - 从 TSDAE 的模型开始，分两路进行训练对比：
     (A) Baseline：普通的随机 Batch 采样。
     (B) Precomputed (自定义 Batch)：
         使用了脚本中定义的 `SimCSEBatchBuilderCarryK` 类。
         这是一个为了解决评论区“复读机”和“同质化”问题写的硬逻辑。
         它在构造 Batch 时会强制过滤掉：
         - 同一个用户的评论
         - 同一个视频下同一个 Root 的回复
         - Token 重叠率过高的句子（位图近似检测）
         确保负样本（Batch 内的其他句子）尽量不相似。

5. 【测试指标】
   - 训练过程中会计算各类任务指标指标写入 CSV，用于监控模型是否坍塌。
"""


import logging
import os
import random
import sys
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Iterable, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets import Dataset

# Sentence-Transformers
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers import InputExample, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import SentenceEvaluator

# Transformers (仅用于 MLM)
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AddedToken,
    TrainerCallback,
)


# ---------------------------
# 0) Logging & RNG
# ---------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------
# 1) 统一配置
# ---------------------------
@dataclass
class Config:
    """训练与数据处理用超参数集合。"""

    encoder_name: str = "google-bert/bert-base-chinese"
    max_seq_len: int = 96

    # TSDAE
    tsdae_batch_size: int = 64
    pre_tsdae_epochs: int = 2
    tsdae_epochs: int = 2
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.2
    eval_steps: int = 500
    save_steps: int = 1000000
    log_steps: int = 200

    # TSDAE 噪声
    noise_del_ratio: float = 0.6

    # MLM
    mlm_epochs: int = 5
    mlm_batch_size: int = 128
    mlm_prob: float = 0.15

    # MNR / SimCSE
    mnr_epochs: int = 1
    mnr_batch_size: int = 128
    mnr_lr: float = 5e-5

    # 表情处理
    emoji_freq_threshold: int = 100
    emoji_unk_token: str = "<EMOJI>"


CFG = Config()
RUN_TAG = f"tsdae-bert-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join("output", RUN_TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info("输出目录: %s", OUTPUT_DIR)
METRICS_CSV = os.path.join(OUTPUT_DIR, "test_metrics.csv")


# ---------------------------
# 2) 读入参数与数据
# ---------------------------
if len(sys.argv) < 2:
    print(f"用法: python {sys.argv[0]} path/to/comments.csv")
    sys.exit(1)

filepath = sys.argv[1]
assert os.path.exists(filepath), f"输入文件不存在: {filepath}"

df: pd.DataFrame = pd.read_csv(filepath)
required = {'上级评论ID', '评论ID', '用户ID', '评论内容', 'character', 'video_type'}
miss = required - set(df.columns)
assert not miss, f"DataFrame 缺少必要列: {miss}"
logging.info("读入数据: %s | 行数=%d", os.path.basename(filepath), len(df))


# ---------------------------
# 3) 文本清洗 + 表情扩表
# ---------------------------
import re
from collections import Counter


def prepare_text_and_vocab(
    df: pd.DataFrame,
    col: str,
    emoji_freq_threshold: int,
    emoji_unk_token: str,
) -> List[str]:
    """
    将评论文本中的 [emoji] 清洗为：
      - 高频 [xxx] 保留为独立 token；
      - 低频 [xxx] 统一替换为 <EMOJI>。
    返回高频 emoji 列表，用于后续扩表。
    """
    pat = re.compile(r"\[.*?\]")

    all_emojis = []
    for s in df[col].astype(str):
        all_emojis.extend(pat.findall(s))

    ctr = Counter(all_emojis)
    hot = sorted([e for e, c in ctr.items() if c >= emoji_freq_threshold])
    hot_set = set(hot)

    if ctr:
        df[col] = df[col].astype(str).apply(
            lambda s: pat.sub(
                lambda m: m.group(0) if m.group(0) in hot_set else emoji_unk_token,
                s,
            )
        )

    logging.info(
        "表情词表 | 阈值=%d | 高频=%d | 样例=%s",
        emoji_freq_threshold,
        len(hot),
        hot[:10] if hot else [],
    )
    return hot


def compute_init_vectors_for_tokens(
    base_tokenizer,
    base_model,
    tokens: List[str],
) -> Dict[str, torch.Tensor]:
    """
    为新增 token 生成语义初始化：
      - 去掉外层方括号；
      - 将内部文本分词后的 subword embedding 做均值；
      - 若全部为 UNK，则退化为 embedding 矩阵均值。
    """
    emb_mat = base_model.get_input_embeddings().weight.detach()
    emb_mean = emb_mat.mean(dim=0)
    unk_id = getattr(base_tokenizer, "unk_token_id", None)

    def inner_text(tok: str) -> str:
        return tok[1:-1] if tok.startswith("[") and tok.endswith("]") and len(tok) >= 3 else tok

    vecs = {}
    for tok in tokens:
        inner = inner_text(tok)
        sub_ids = base_tokenizer.encode(inner, add_special_tokens=False)
        valid = [
            i for i in sub_ids
            if (i is not None)
            and (unk_id is None or i != unk_id)
            and (0 <= i < emb_mat.size(0))
        ]
        if valid:
            idx = torch.tensor(valid, dtype=torch.long, device=emb_mat.device)
            vecs[tok] = emb_mat.index_select(0, idx).mean(dim=0)
        else:
            vecs[tok] = emb_mean
    vecs[CFG.emoji_unk_token] = emb_mean
    return vecs


@torch.no_grad()
def apply_init_vectors_to_model(
    hf_model,
    hf_tokenizer,
    init_dict: Dict[str, torch.Tensor],
):
    """将 init_dict 中的向量拷入对应 token embedding，并重新 tie_weights。"""
    emb = hf_model.get_input_embeddings()
    if emb is None or not init_dict:
        return
    for tok, vec in init_dict.items():
        tid = hf_tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0:
            emb.weight[tid].copy_(vec.to(emb.weight.device).to(emb.weight.dtype))
    if hasattr(hf_model, "tie_weights"):
        try:
            hf_model.tie_weights()
        except Exception:
            pass


def _as_added_tokens(tokens: List[str]) -> List[AddedToken]:
    return [AddedToken(t, lstrip=False, rstrip=False, single_word=False) for t in tokens]


def add_tokens_and_resize(
    hf_model,
    hf_tokenizer,
    tokens_to_add: List[str],
    init_vectors: Dict[str, torch.Tensor],
    ensure_token: Optional[str] = None,
) -> int:
    """
    向 tokenizer 扩表并 resize embedding；随后应用语义初始化。
    若 ensure_token 不在词表，自动补入。
    """
    add_list = list(tokens_to_add) if tokens_to_add else []
    if ensure_token and ensure_token not in hf_tokenizer.get_vocab():
        add_list.append(ensure_token)
    if not add_list:
        return 0

    num_added = hf_tokenizer.add_tokens(
        _as_added_tokens(add_list),
        special_tokens=False,
    )
    if num_added > 0:
        hf_model.resize_token_embeddings(len(hf_tokenizer))
        apply_init_vectors_to_model(hf_model, hf_tokenizer, init_vectors)
    return num_added


def build_st_encoder_with_tokens(
    backbone_name: str,
    max_seq_len: int,
    tokens_to_add: List[str],
    init_vectors: Dict[str, torch.Tensor],
    ensure_token: Optional[str],
) -> SentenceTransformer:
    """
    构建 SentenceTransformer 编码器：Transformer + CLS Pooling，
    并在构建时完成 tokenizer/embedding 扩表与初始化。
    """
    word = models.Transformer(backbone_name, max_seq_length=max_seq_len)
    add_tokens_and_resize(
        word.auto_model,
        word.tokenizer,
        tokens_to_add,
        init_vectors,
        ensure_token=ensure_token,
    )
    pool = models.Pooling(word.get_word_embedding_dimension(), "cls")
    st = SentenceTransformer(modules=[word, pool])
    st.max_seq_length = max_seq_len
    return st


def build_mlm_with_tokens(
    backbone_name: str,
    tokens_to_add: List[str],
    init_vectors: Dict[str, torch.Tensor],
    ensure_token: Optional[str],
):
    """
    构建 MLM 模型与 tokenizer，并完成扩表与初始化。
    """
    tok = AutoTokenizer.from_pretrained(backbone_name)
    mlm = AutoModelForMaskedLM.from_pretrained(backbone_name)
    add_tokens_and_resize(
        mlm,
        tok,
        tokens_to_add=tokens_to_add,
        init_vectors=init_vectors,
        ensure_token=ensure_token,
    )
    return tok, mlm


# === 执行清洗与扩表 ===
hot_emojis = prepare_text_and_vocab(
    df,
    col='评论内容',
    emoji_freq_threshold=CFG.emoji_freq_threshold,
    emoji_unk_token=CFG.emoji_unk_token,
)
_base_tok_for_init = AutoTokenizer.from_pretrained(CFG.encoder_name)
_base_mlm_for_init = AutoModelForMaskedLM.from_pretrained(CFG.encoder_name)
init_vectors = compute_init_vectors_for_tokens(
    _base_tok_for_init,
    _base_mlm_for_init,
    hot_emojis,
)

# ---------------------------
# 4) 构建 ST 编码器 + TSDAE 数据
# ---------------------------
logging.info("加载 Encoder: %s", CFG.encoder_name)
model = build_st_encoder_with_tokens(
    CFG.encoder_name,
    CFG.max_seq_len,
    hot_emojis,
    init_vectors,
    ensure_token=CFG.emoji_unk_token,
)
enc_tok = model[0].tokenizer  # SentenceTransformer 的 Transformer 子模块
emoji_base_st_dir = os.path.join(OUTPUT_DIR, "emoji_base_st")
model.save(emoji_base_st_dir)
logging.info("已保存 emoji 扩表后的 ST 基座模型到: %s", emoji_base_st_dir)


def _collect_special_ids(tok) -> set:
    """收集 tokenizer 的特殊 token id（用于噪声时剔除）。"""
    return {
        i for i in [
            getattr(tok, "cls_token_id", None),
            getattr(tok, "sep_token_id", None),
            getattr(tok, "bos_token_id", None),
            getattr(tok, "eos_token_id", None),
            getattr(tok, "pad_token_id", None),
            getattr(tok, "unk_token_id", None),
        ] if i is not None
    }


SPECIAL_IDS = _collect_special_ids(enc_tok)


def make_noisy(
    texts: List[str],
    ratio: float,
    seed: Optional[int] = None,
) -> List[str]:
    """
    简单删除噪声：从非特殊 token 中按比例删除，空则至少保留一个。
    用于 TSDAE 输入扰动与 quick tests 多视图生成。
    """
    rnd = random.Random(seed) if seed is not None else random
    noisy = []
    for text in texts:
        ids = enc_tok.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=CFG.max_seq_len,
        )
        core = [i for i in ids if i not in SPECIAL_IDS]
        if not core:
            noisy.append(text)
            continue

        n = len(core)
        k = max(1, math.ceil((1.0 - ratio) * n))
        if k >= n:
            kept = core
        else:
            pos = sorted(rnd.sample(range(n), k))
            kept = [core[i] for i in pos]

        noisy.append(enc_tok.decode(kept, skip_special_tokens=True))
    return noisy


def build_tsdae_dataset(
    train_texts: List[str],
    eval_texts: List[str],
    cfg: Config,
    precomputed_train_lens: List[int],
    seed: int = SEED,
):
    """
    TSDAE 数据构造（重构版）：
      - train/eval 切分在函数外完成，避免 train_texts 隐含包含 eval；
      - 仅对 train 做“按长度排序 + block shuffle”以减少 padding；
      - 对 train/eval 都用 set_transform 注入噪声字段。
    """
    assert len(train_texts) == len(precomputed_train_lens), "train_texts 与预计算长度不一致"

    # 仅重排训练集
    lens = precomputed_train_lens
    idx = sorted(range(len(train_texts)), key=lambda i: lens[i])
    blocks = [idx[i:i + 512] for i in range(0, len(idx), 512)]
    random.Random(seed).shuffle(blocks)
    reidx = [j for b in blocks for j in b]
    train_sorted = [train_texts[i] for i in reidx]

    train_ds = Dataset.from_dict({"text": train_sorted})
    eval_ds = Dataset.from_dict({"text": list(eval_texts)})

    def _inject_noise(batch):
        return {
            "noisy": make_noisy(batch["text"], cfg.noise_del_ratio),
            "text": batch["text"],
        }

    train_ds.set_transform(_inject_noise, columns=["text"], output_all_columns=True)
    eval_ds.set_transform(_inject_noise, columns=["text"], output_all_columns=True)

    logging.info("TSDAE Train 样本检查 | noisy: %s", train_ds[0]["noisy"][:120].replace("\n", " "))
    logging.info("TSDAE Train 样本检查 | text : %s", train_ds[0]["text"][:120].replace("\n", " "))
    logging.info("TSDAE Eval  样本检查 | noisy: %s", eval_ds[0]["noisy"][:120].replace("\n", " "))
    logging.info("TSDAE Eval  样本检查 | text : %s", eval_ds[0]["text"][:120].replace("\n", " "))

    return train_ds, eval_ds


# 不直接删 df 行，只用 mask 构造训练文本列表
def process_tokens(text: str):
    # 获取 IDs，不包含 [CLS], [SEP]
    ids = enc_tok.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=CFG.max_seq_len,
    )
    return ids, len(ids)

# 使用 zip 将结果拆分到两列
# 注意：这比 apply 两次要快，且只分词一次
logging.info("开始预处理分词...")
df['input_ids'], df['__tok_len__'] = zip(*df['评论内容'].astype(str).apply(process_tokens))
valid_mask = df['__tok_len__'] >= 4
valid_idx = df.index[valid_mask].to_numpy()
assert valid_idx.size > 0, "过滤后没有有效句子（token数>=4）"

TSDAE_EVAL_MAX = 5000
TSDAE_EVAL_RATIO = 0.02
TSDAE_EVAL_MIN = 100

n_valid = valid_idx.size
eval_size = np.clip(int(TSDAE_EVAL_RATIO * n_valid), TSDAE_EVAL_MIN, TSDAE_EVAL_MAX)

rng = np.random.default_rng(SEED)
eval_idx = rng.choice(valid_idx, size=eval_size, replace=False)
eval_idx_set = set(map(int, eval_idx.tolist()))
train_idx = np.array([i for i in valid_idx if int(i) not in eval_idx_set], dtype=valid_idx.dtype)

# 训练/测试文本与长度
train_texts = df.loc[train_idx, '评论内容'].astype(str).tolist()
eval_texts = df.loc[eval_idx, '评论内容'].astype(str).tolist()
train_lens = df.loc[train_idx, '__tok_len__'].astype(int).tolist()

# TSDAE dataset 构造（不再内部切分）
train_dataset, eval_dataset = build_tsdae_dataset(
    train_texts=train_texts,
    eval_texts=eval_texts,
    cfg=CFG,
    precomputed_train_lens=train_lens,
    seed=SEED,
)

logging.info(
    "TSDAE 数据集规模 | Valid=%d | Train=%d | Eval=%d",
    n_valid,
    len(train_dataset),
    len(eval_dataset),
)
df.to_pickle("comments_preprocessed.pkl")
logging.info("已保存预处理后的评论数据到: %s", "comments_preprocessed.pkl")

# ---------------------------
# 5) Quick Tests 统一入口
# ---------------------------
def encode_norm(
    m: SentenceTransformer,
    texts,
    batch_size: int = 64,
):
    """编码文本并做 L2 归一化，返回 (N, D) Tensor。"""
    embs = m.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    return F.normalize(embs, p=2, dim=1)


def quick_eval_embeddings(
    m: SentenceTransformer,
    eval_texts,
    max_n: int = 1000,
):
    """
    评估句向量一致性：
      - alignment: clean vs noisy 的对角余弦平均；
      - R@1: 以 clean 为库，noisy 为 query 的自检召回。
    """
    N = min(max_n, len(eval_texts))
    base = eval_texts[:N]
    noisy = make_noisy(base, CFG.noise_del_ratio, seed=SEED + 1)
    emb_clean = encode_norm(m, base)
    emb_noisy = encode_norm(m, noisy)
    sims = emb_noisy @ emb_clean.T
    align = float(torch.diag(sims).mean())
    r1 = float((torch.argmax(sims, dim=1) == torch.arange(N, device=sims.device)).float().mean())
    return align, r1


class TokenizedSentencesDataset:
    """对原始句子即时 tokenization，用于 MLM 训练/评测。"""

    def __init__(
        self,
        sentences,
        tokenizer,
        max_length,
        cache_tokenization: bool = False,
    ):
        self.tokenizer = tokenizer
        self.raw_sentences = list(sentences)
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self._cache = [None] * len(self.raw_sentences) if cache_tokenization else None

    def __getitem__(self, item):
        if self._cache is not None:
            if self._cache[item] is None:
                self._cache[item] = self.tokenizer(
                    self.raw_sentences[item],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_attention_mask=True,
                )
            return self._cache[item]
        return self.tokenizer(
            self.raw_sentences[item],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )

    def __len__(self):
        return len(self.raw_sentences)


def quick_eval_mlm(
    mlm_model,
    mlm_tokenizer,
    eval_texts,
    max_batches: int = 50,
    batch_size: int = 64,
    max_length: int = CFG.max_seq_len,
    mlm_prob: float = CFG.mlm_prob,
):
    """在小批量上估计 MLM loss/ppl，用于阶段性回归检测。"""
    device = next(mlm_model.parameters()).device

    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    ds = TokenizedSentencesDataset(eval_texts, mlm_tokenizer, max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=mlm_tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    mlm_model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = mlm_model(**batch)
            total_loss += float(out.loss.item())
            steps += 1

    if steps == 0:
        return float("nan"), float("nan")
    avg_loss = total_loss / steps
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return avg_loss, ppl


def quick_eval_simcse_nce(
    st_model: SentenceTransformer,
    eval_texts,
    max_n: int = 2048,
    batch_size: int = 128,
    use_dropout: bool = True,
):
    """
    使用 MultipleNegativesRankingLoss 构建 NCE loss，
    反映“同句为正，对其他 batch 内句子为负”的训练难度。
    """
    N = min(max_n, len(eval_texts))
    if N < batch_size:
        return float("nan")
    device = getattr(
        st_model,
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    st_model.to(device)
    samples = [InputExample(texts=[s, s]) for s in eval_texts[:N]]
    dl = DataLoader(
        samples,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        collate_fn=st_model.smart_batching_collate,
    )
    nce_loss = losses.MultipleNegativesRankingLoss(st_model)
    prev_mode = st_model.training
    st_model.train(use_dropout)
    total, steps = 0.0, 0
    with torch.no_grad():
        for features, labels in dl:
            features = [{
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in f.items()
            } for f in features]
            loss_val = nce_loss(features, labels)
            total += float(loss_val.item())
            steps += 1
    st_model.train(prev_mode)
    return (total / steps) if steps > 0 else float("nan")


@torch.no_grad()
def eval_alignment_uniformity(
    m: SentenceTransformer,
    eval_texts,
    max_n: int = 1000,
    t: float = 2.0,
    alpha: float = 2.0,
):
    """
    计算 alignment / uniformity：
      - align_l2: L2^alpha 正对距离；
      - align_cos: 正对 cosine；
      - uniformity: log E[exp(-t ||zi - zj||^2)]。
    """
    N = min(max_n, len(eval_texts))
    if N < 2:
        return {"align_l2": float("nan"), "align_cos": float("nan"), "uniformity": float("nan")}
    base = eval_texts[:N]
    noisy = make_noisy(base, CFG.noise_del_ratio, seed=SEED + 1)
    emb_clean = encode_norm(m, base)
    emb_noisy = encode_norm(m, noisy)
    cos_diag = torch.sum(emb_clean * emb_noisy, dim=1)
    sqd = (2 - 2 * cos_diag).clamp_min(0)
    align_l2 = float((sqd ** (alpha / 2)).mean().item())
    align_cos = float(cos_diag.mean().item())

    Z = torch.cat([emb_clean, emb_noisy], dim=0)
    cos_mat = cos_sim(Z, Z)
    sq = (2 - 2 * cos_mat).clamp_min(0)
    M = sq.size(0)
    uniformity = float(torch.log(torch.exp(-t * sq[~torch.eye(
        M, dtype=torch.bool, device=sq.device
    )]).mean()).item())
    return {"align_l2": align_l2, "align_cos": align_cos, "uniformity": uniformity}


@torch.no_grad()
def eval_anisotropy(
    m: SentenceTransformer,
    eval_texts,
    max_n: int = 5000,
    topk_for_pca: tuple = (1, 3, 10),
):
    """
    利用平均余弦与 PCA 方差占比诊断 embedding 空间的各向异性/坍缩倾向。
    """
    N = min(max_n, len(eval_texts))
    if N < 3:
        return {"mean_cos_offdiag": float("nan")}
    base = eval_texts[:N]
    emb = encode_norm(m, base).detach().cpu()
    cos_mat = cos_sim(emb, emb)
    mean_cos_offdiag = float(cos_mat[~torch.eye(N, dtype=torch.bool)].mean().item())
    X = emb.numpy()
    pca = PCA(n_components=min(X.shape[0], X.shape[1])).fit(X)
    var_ratio = pca.explained_variance_ratio_
    out = {"mean_cos_offdiag": mean_cos_offdiag}
    for K in topk_for_pca:
        out[f"pca_var_top{K}"] = float(np.sum(var_ratio[:min(K, var_ratio.shape[0])]))
    return out


def _get_mlm_base_module(mlm):
    """兼容常见 HF 模型命名，获取 MLM 的 encoder/base 模块。"""
    if hasattr(mlm, "bert"):
        return mlm.bert
    if hasattr(mlm, "roberta"):
        return mlm.roberta
    if hasattr(mlm, "deberta"):
        return mlm.deberta
    if hasattr(mlm, "electra"):
        return mlm.electra
    if hasattr(mlm, "base_model"):
        return mlm.base_model
    return mlm


def sync_mlm_base_from_st_inplace(
    mlm_model,
    st_word_module,
):
    """
    将 SentenceTransformer 的 Transformer 子模块权重覆盖到 MLM base，
    并重新 tie_weights()，用于评估当前 encoder 对应的 MLM ppl。
    """
    base_sd = st_word_module.auto_model.state_dict()
    mlm_base = _get_mlm_base_module(mlm_model)
    mlm_base.load_state_dict(base_sd, strict=False)
    if hasattr(mlm_model, "tie_weights"):
        mlm_model.tie_weights()


def sync_st_from_mlm_base_inplace(
    st_model: SentenceTransformer,
    mlm_model,
):
    """
    将 MLM base 的权重注入当前 SentenceTransformer 的 Transformer 子模块。
    （不读磁盘，直接内存同步。）
    """
    st_word_module = st_model[0]
    mlm_base = _get_mlm_base_module(mlm_model)
    st_word_module.auto_model.load_state_dict(mlm_base.state_dict(), strict=False)


def _record_test_metrics(
    tag: str,
    metrics: Dict,
):
    """将阶段性指标写入METRICS_CSV，并打印关键信息。"""
    rec = {"tag": tag, "timestamp": datetime.now().isoformat()}
    rec.update(metrics)

    try:
        csv_path = Path(METRICS_CSV)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_row = pd.DataFrame([rec])
        if not csv_path.exists():
            # 首次写入：带表头
            df_row.to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            # 追加：不写表头
            df_row.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")

        logging.info("已写入测试记录到: %s", csv_path)
    except Exception as e:
        logging.warning("写入测试记录 CSV 失败：%s", e)

    logging.info(
        "[Test/%s] align=%.4f | R@1=%.4f | NCE=%.4f | "
        "align_l2=%.4f | align_cos=%.4f | uniformity=%.4f | "
        "aniso=%.4f | PCA(1/3/10)=%.3f/%.3f/%.3f | "
        "MLM loss=%.4f | ppl=%.2f",
        tag,
        rec.get("emb_align", float("nan")),
        rec.get("emb_r1", float("nan")),
        rec.get("simcse_nce", float("nan")),
        rec.get("align_l2", float("nan")),
        rec.get("align_cos", float("nan")),
        rec.get("uniformity", float("nan")),
        rec.get("aniso_mean_cos", float("nan")),
        rec.get("pca_top1", float("nan")),
        rec.get("pca_top3", float("nan")),
        rec.get("pca_top10", float("nan")),
        rec.get("mlm_loss", float("nan")),
        rec.get("mlm_ppl", float("nan")),
    )


def run_stage_tests(
    tag,
    st_model: SentenceTransformer,
    eval_texts,
    mlm_model=None,
    mlm_tokenizer=None,
    st_word_module=None,
):
    """
    统一的阶段性测试入口：
      - Embedding 对齐性 / uniformity / 各向异性；
      - SimCSE NCE loss；
      - 若提供 MLM 模型，则同步权重后评估 MLM loss/ppl。
    """
    logging.info("========== Quick Tests (%s) ==========", tag)

    mlm_loss = mlm_ppl = float("nan")
    if (mlm_model is not None) and (st_word_module is not None):
        logging.info("将st模型权重刷入mlm模型")
        sync_mlm_base_from_st_inplace(mlm_model, st_word_module)
        mlm_loss, mlm_ppl = quick_eval_mlm(
            mlm_model,
            st_word_module.tokenizer,
            eval_texts[:5000],
            max_batches=50,
            batch_size=min(CFG.mlm_batch_size, 64),
            max_length=CFG.max_seq_len,
            mlm_prob=CFG.mlm_prob,
        )
    elif (mlm_model is not None) and (mlm_tokenizer is not None):
        logging.info("将mlm模型权重刷入st模型")
        sync_st_from_mlm_base_inplace(st_model, mlm_model)
        mlm_loss, mlm_ppl = quick_eval_mlm(
            mlm_model,
            mlm_tokenizer,
            eval_texts[:5000],
            max_batches=50,
            batch_size=min(CFG.mlm_batch_size, 64),
            max_length=CFG.max_seq_len,
            mlm_prob=CFG.mlm_prob,
        )

    align, r1 = quick_eval_embeddings(st_model, eval_texts, max_n=1000)
    nce = quick_eval_simcse_nce(
        st_model,
        eval_texts,
        max_n=2048,
        batch_size=min(128, CFG.mnr_batch_size),
    )
    uni = eval_alignment_uniformity(
        st_model,
        eval_texts,
        max_n=1000,
        t=2.0,
        alpha=2.0,
    )
    aniso = eval_anisotropy(
        st_model,
        eval_texts,
        max_n=2000,
        topk_for_pca=(1, 3, 10),
    )

    metrics = {
        "emb_align": align,
        "emb_r1": r1,
        "simcse_nce": nce,
        "align_l2": uni.get("align_l2", float("nan")),
        "align_cos": uni.get("align_cos", float("nan")),
        "uniformity": uni.get("uniformity", float("nan")),
        "aniso_mean_cos": aniso.get("mean_cos_offdiag", float("nan")),
        "pca_top1": aniso.get("pca_var_top1", float("nan")),
        "pca_top3": aniso.get("pca_var_top3", float("nan")),
        "pca_top10": aniso.get("pca_var_top10", float("nan")),
        "mlm_loss": mlm_loss,
        "mlm_ppl": mlm_ppl,
    }
    _record_test_metrics(tag, metrics)

    return metrics


# ---------------------------
# 6) 回调与 evaluator（按 epoch 注入测试）
# ---------------------------
class MlmPerEpochCallback(TrainerCallback):
    """
    MLM 阶段：每个 epoch 结束时，
      1) 用 MLM 权重刷新 SentenceTransformer encoder；
      2) 对当前 encoder + MLM 执行一次 quick tests。
    """

    def __init__(self, st_model, mlm_tokenizer, mlm_model, eval_texts, tag_prefix="After-MLM-ep"):
        self.st_model = st_model
        self.mlm_tokenizer = mlm_tokenizer
        self.mlm_model = mlm_model
        self.eval_texts = eval_texts
        self.tag_prefix = tag_prefix

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return
        ep = int(state.epoch)
        tag = f"{self.tag_prefix}{ep}"
        run_stage_tests(
            tag=tag,
            st_model=self.st_model,
            eval_texts=self.eval_texts,
            mlm_model=self.mlm_model,            
            mlm_tokenizer=self.mlm_tokenizer,
        )


class TsdaePhase2Callback(TrainerCallback):
    """
    TSDAE Phase-2：每个 epoch 结束时，对当前 encoder + MLM 执行 quick tests。
    """

    def __init__(self, st_model, mlm_model, eval_texts, tag_prefix="After-TSDAE-Phase2-ep"):
        self.st_model = st_model
        self.mlm_model = mlm_model
        self.eval_texts = eval_texts
        self.tag_prefix = tag_prefix

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return
        ep = int(state.epoch)
        tag = f"{self.tag_prefix}{ep}"
        run_stage_tests(
            tag=tag,
            st_model=self.st_model,
            eval_texts=self.eval_texts,
            mlm_model=self.mlm_model,
            st_word_module=self.st_model[0],
        )


class PerEpochSimCseEvaluator(SentenceEvaluator):
    """
    用于 MNR Baseline / Precomputed：
      - 每轮调用视为“一个 epoch 结束后的评测”，epoch 从 0 开始；
      - 内部调用 run_stage_tests，并将 SimCSE NCE 作为评测 score（取负）。
    """

    def __init__(self, eval_texts, mlm_model, st_word_module, tag_prefix: str):
        self.eval_texts = eval_texts
        self.mlm_model = mlm_model
        self.st_word_module = st_word_module
        self.tag_prefix = tag_prefix

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str,
        epoch: int,
        steps: int,
    ) -> float:
        tag = f"{self.tag_prefix}{epoch}"
        metrics = run_stage_tests(
            tag=tag,
            st_model=model,
            eval_texts=self.eval_texts,
            mlm_model=self.mlm_model,
            st_word_module=self.st_word_module,
        )
        nce = metrics.get("simcse_nce", float("nan"))
        return -nce if not math.isnan(nce) else 0.0


# ---------------------------
# 7) MLM 训练封装（单次多 epoch + callback）
# ---------------------------
def train_mlm(
    mlm_tokenizer,
    mlm_model,
    st_model: SentenceTransformer,
    train_texts,
    eval_texts,
    cfg: Config,
    output_dir: str,
):
    """
    使用 HF Trainer 进行 MLM 训练。
    单次多 epoch 训练，通过 callback 在每个 epoch 结束时做 quick tests。
    """
    os.makedirs(output_dir, exist_ok=True)

    train_ds = TokenizedSentencesDataset(
        train_texts,
        mlm_tokenizer,
        cfg.max_seq_len,
    )
    dev_ds = TokenizedSentencesDataset(
        list(eval_texts),
        mlm_tokenizer,
        cfg.max_seq_len,
        cache_tokenization=True,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=mlm_tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_prob,
    )
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.mlm_epochs,
        per_device_train_batch_size=cfg.mlm_batch_size,
        per_device_eval_batch_size=cfg.mlm_batch_size,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=1,
        logging_steps=cfg.log_steps,
        prediction_loss_only=True,
        fp16=False,
        bf16=True,
        group_by_length=True,
        run_name="MLM",
    )
    trainer = Trainer(
        model=mlm_model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )
    trainer.add_callback(
        MlmPerEpochCallback(
            st_model=st_model,
            mlm_tokenizer=mlm_tokenizer,
            mlm_model=mlm_model,
            eval_texts=eval_texts,
        )
    )

    mlm_tokenizer.save_pretrained(output_dir)
    logging.info("开始 MLM 训练 ...")
    trainer.train()
    mlm_model.save_pretrained(output_dir)
    logging.info("MLM 模型已保存到: %s", output_dir)


# ---------------------------
# 8) TSDAE 两阶段封装（Phase-1: 冻结；Phase-2: callback 每 epoch 测试）
# ---------------------------
class DenoisingAutoEncoderLossWithMask(DenoisingAutoEncoderLoss):
    """
    TSDAE 重构损失：
      - encoder 输出句向量；
      - decoder 做自回归重建（teacher forcing），使用显式 mask。
    """
    def retokenize(
        self,
        sentence_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        input_ids = sentence_features["input_ids"]
        device = input_ids.device
        sentences_decoded = self.tokenizer_encoder.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return self.tokenizer_decoder(
            sentences_decoded,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=None,
        ).to(device)

    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        source_features, target_features = tuple(sentence_features)
        if self.need_retokenization:
            target_features = self.retokenize(target_features)

        reps = self.encoder(source_features)["sentence_embedding"]
        target_length = target_features["input_ids"].shape[1]
        decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        label_ids = target_features["input_ids"][:, 1:]

        decoder_attention_mask = target_features.get(
            "attention_mask",
            torch.ones_like(target_features["input_ids"]),
        )[:, : target_length - 1]
        encoder_attention_mask = source_features.get(
            "attention_mask",
            torch.ones(reps.size(0), 1, device=reps.device, dtype=decoder_attention_mask.dtype),
        )[:, :1]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            encoder_hidden_states=reps[:, None],
            encoder_attention_mask=encoder_attention_mask,
            labels=None,
            return_dict=None,
            use_cache=False,
        )
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_decoder.pad_token_id
        )
        return ce_loss_fct(
            lm_logits.view(-1, lm_logits.shape[-1]),
            label_ids.reshape(-1),
        )


def freeze_all_but_cross_attn(st_model, dae_loss):
    """Phase-1：冻结 encoder + decoder 其余参数，仅训练 decoder 中包含 'cross' 的子层。"""
    for p in st_model.parameters():
        p.requires_grad = False
    for p in dae_loss.parameters():
        p.requires_grad = False
    for name, p in dae_loss.decoder.named_parameters():
        if "cross" in name.lower():
            p.requires_grad = True
    logging.info("TSDAE Phase-1 冻结策略：仅训练 decoder cross-attention")


def run_tsdae_two_phase(
    st_model: SentenceTransformer,
    train_ds,
    eval_ds,
    cfg: Config,
    eval_texts,
    mlm_model=None,
):
    """
    运行 TSDAE 两阶段训练：
      - Phase-1（pre_tsdae_epochs）：冻结 encoder，只训练 decoder cross-attention；
           训练结束后测试一次；
      - Phase-2（tsdae_epochs）：全量解冻，单次多 epoch 训练，每个 epoch 结束时测试。
    """
    logging.info("创建tie-decoder用于 TSDAE")
    dae_loss = DenoisingAutoEncoderLossWithMask(
        st_model,
        tie_encoder_decoder=True,
    )
    # Phase-1：冻结 encoder，仅训练 cross-attention
    args_phase1 = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.pre_tsdae_epochs,
        per_device_train_batch_size=cfg.tsdae_batch_size,
        per_device_eval_batch_size=cfg.tsdae_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        logging_steps=cfg.log_steps,
        run_name="tsdae-crossattn",
    )

    freeze_all_but_cross_attn(st_model, dae_loss)
    logging.info("TSDAE Phase-1 开始训练 ...")
    SentenceTransformerTrainer(
        model=st_model,
        args=args_phase1,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=dae_loss,
    ).train()

    # Phase-1 结束后测试一次
    if mlm_model is not None:
        run_stage_tests(
            "After-TSDAE-Phase1",
            st_model,
            eval_texts,
            mlm_model=mlm_model,
            st_word_module=st_model[0],
        )

    # Phase-2：全量解冻，单次多 epoch 训练 + epoch-end callback 测试
    for p in st_model.parameters():
        p.requires_grad = True
    for p in dae_loss.parameters():
        p.requires_grad = True

    args_phase2 = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.tsdae_epochs,
        per_device_train_batch_size=cfg.tsdae_batch_size,
        per_device_eval_batch_size=cfg.tsdae_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        logging_steps=cfg.log_steps,
        run_name="tsdae",
    )

    trainer2 = SentenceTransformerTrainer(
        model=st_model,
        args=args_phase2,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=dae_loss,
    )

    if mlm_model is not None:
        trainer2.add_callback(
            TsdaePhase2Callback(
                st_model=st_model,
                mlm_model=mlm_model,
                eval_texts=eval_texts,
            )
        )

    logging.info("TSDAE Phase-2 开始训练，总 epoch=%d ...", cfg.tsdae_epochs)
    trainer2.train()


# ---------------------------
# 9) 两路 MNR（SimCSE）封装
# ---------------------------
def train_mnr_baseline(
    st_model: SentenceTransformer,
    dedup_texts: List[str],
    cfg: Config,
    eval_texts,
    mlm_model=None,
):
    """
    MNR Baseline：随机 batch 的 SimCSE（同句为正对）。
    单次多 epoch 训练，通过 evaluator 在每个 epoch 结束时调用 quick tests。
    """

    mnr_loss = losses.MultipleNegativesRankingLoss(st_model)
    samples = [InputExample(texts=[s, s]) for s in dedup_texts]
    loader = DataLoader(
        samples,
        shuffle=True,
        batch_size=cfg.mnr_batch_size,
        drop_last=True,
        collate_fn=st_model.smart_batching_collate,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.mnr_epochs
    warmup_steps = math.ceil(total_steps * cfg.warmup_ratio)

    evaluator = PerEpochSimCseEvaluator(
        eval_texts=eval_texts,
        mlm_model=mlm_model,
        st_word_module=st_model[0],
        tag_prefix="After-MNR-Baseline-ep",
    )

    logging.info(
        "MNR Baseline 训练 | epochs=%d | steps/epoch=%d | total_steps=%d | warmup=%d",
        cfg.mnr_epochs,
        steps_per_epoch,
        total_steps,
        warmup_steps,
    )

    st_model.fit(
        train_objectives=[(loader, mnr_loss)],
        epochs=cfg.mnr_epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        # evaluation_steps=steps_per_epoch,
        optimizer_params={"lr": cfg.mnr_lr},
        show_progress_bar=True,
        use_amp=False,
    )


def compute_root_ids_vectorized(
    comment_id: np.ndarray,
    parent_id: np.ndarray,
) -> np.ndarray:
    """
    计算每条评论的 root 评论 ID（用于约束“同视频不同 root”）。
    """
    N = comment_id.shape[0]
    id2idx = {int(cid): i for i, cid in enumerate(comment_id)}
    parent_index = np.full(N, -1, dtype=np.int64)
    non_root = parent_id != 0
    if non_root.any():
        parent_index[non_root] = np.array(
            [id2idx.get(int(pid), -1) for pid in parent_id[non_root]],
            dtype=np.int64,
        )
    root_index = np.where(parent_index == -1, np.arange(N, dtype=np.int64), parent_index)
    changed, it, max_iter = True, 0, 32
    while changed and it < max_iter:
        it += 1
        nxt = root_index.copy()
        valid = (root_index != -1) & (parent_index[root_index] != -1)
        nxt[valid] = parent_index[root_index[valid]]
        changed = np.any(nxt != root_index)
        root_index = nxt
    return comment_id[root_index]


class SimCSEBatchBuilderCarryK:
    """
    约束友好的 SimCSE batch 构造器（用于 precomputed 分支）：
      - 每批同视频样本数 ≤ K，且 root 不重复；
      - 不同用户；
      - token 集 overlap ≤ τ（1024-bit 位图近似 + 精确复核）。
    通过长度分桶 + carry 机制提升满批率。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        MAX_SEQ_LEN: int = 96,
        batch_size: int = 128,
        epoch_seed: Optional[int] = None,
        length_bucket_width: int = 8,
        hard_min_len: int = 8,
        prefer_min_len_ratio: float = 0.6,
        overlap_ratio_thresh: float = 0.3,
        max_per_video_per_batch: int = 2,
        stop_topk: int = 5,
        candidate_chunk_mul: int = 16,
        bitset_bits: int = 1024,
        suspicious_margin: float = 0.95,
        rounds: int = 3,
    ):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.MAX, self.B, self.seed = int(MAX_SEQ_LEN), int(batch_size), epoch_seed
        self.bw, self.hard_min = int(length_bucket_width), int(hard_min_len)
        self.pref_min = int(math.floor(prefer_min_len_ratio * MAX_SEQ_LEN))
        self.tau, self.margin = float(overlap_ratio_thresh), float(suspicious_margin)
        self.Kvideo = int(max_per_video_per_batch)
        self.stop_topk, self.chunk_mul = int(stop_topk), int(candidate_chunk_mul)
        self.M_bits, self.rounds = int(bitset_bits), int(rounds)

        # 构造后填充
        self.keep_idx = None
        self.video_id = None
        self.user_id = None
        self.root_id = None
        self.tok_lens = None
        self.len_bucket = None
        self.flat_ids = None
        self.offsets = None
        self.uniq_token_lens = None
        self.bitset_words = None

    def build_epoch_batches(self) -> List[np.ndarray]:
        """生成一个 epoch 的 batch 列表（仅 batch 顺序随机、成员固定）。"""
        self._prepare_if_needed()
        rng = np.random.default_rng(self.seed)
        used = np.zeros(self.keep_idx.size, dtype=bool)
        all_batches: List[np.ndarray] = []

        W = self.bitset_words.shape[1]
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

        for _ in range(self.rounds):
            rem_mask = ~used
            if not rem_mask.any():
                break
            uniq_videos_in_scope = np.unique(self.video_id[rem_mask]).size
            if self.B > self.Kvideo * uniq_videos_in_scope:
                break

            buckets = {}
            for b in np.unique(self.len_bucket[rem_mask]):
                idx = np.where((self.len_bucket == b) & rem_mask)[0]
                rng.shuffle(idx)
                buckets[int(b)] = idx
            if not buckets:
                break
            bucket_order = sorted(buckets.keys(), reverse=True)
            ptr = {b: 0 for b in bucket_order}

            carry_rel: List[int] = []
            carry_vid_count = {}
            carry_vid_roots = {}
            carry_users = set()

            def _reset():
                carry_rel.clear()
                carry_vid_count.clear()
                carry_vid_roots.clear()
                carry_users.clear()

            grab = self.chunk_mul * self.B
            while True:
                pool_rel = self._grab_pool(buckets, bucket_order, ptr, used, grab)
                if pool_rel.size == 0 and len(carry_rel) == 0:
                    break
                if pool_rel.size == 0 and len(carry_rel) > 0:
                    break
                pool_rel = pool_rel[~used[pool_rel]]
                if pool_rel.size == 0:
                    continue

                C = max(self.B + self.B // 2, self.B + 32)
                if pool_rel.size < C:
                    cand_mat = pool_rel[: int(pool_rel.size)].reshape(1, int(pool_rel.size))
                else:
                    Kblocks = pool_rel.size // C
                    cand_mat = pool_rel[: Kblocks * C].reshape(Kblocks, C)

                for bi in range(cand_mat.shape[0]):
                    block_rel = cand_mat[bi]
                    order = self._rank_block(block_rel, W, lut)
                    pos = 0
                    while pos < order.size:
                        idx_rel = block_rel[order[pos]]
                        pos += 1
                        if used[idx_rel]:
                            continue
                        u = int(self.user_id[idx_rel])
                        if u in carry_users:
                            continue
                        v = int(self.video_id[idx_rel])
                        r = int(self.root_id[idx_rel])
                        if self.Kvideo > 0 and carry_vid_count.get(v, 0) >= self.Kvideo:
                            continue
                        if r in carry_vid_roots.get(v, set()):
                            continue
                        if not self._ok_with_carry(idx_rel, carry_rel, W, lut):
                            continue

                        carry_rel.append(idx_rel)
                        carry_users.add(u)
                        carry_vid_count[v] = carry_vid_count.get(v, 0) + 1
                        carry_vid_roots.setdefault(v, set()).add(r)

                        if len(carry_rel) >= self.B:
                            take_rel = np.array(carry_rel[:self.B], dtype=np.int64)
                            used[take_rel] = True
                            all_batches.append(self.keep_idx[take_rel])
                            _reset()
                            pos = 0
            if not all_batches:
                break

        logging.info("[CarryK] 生成批次数: %d", len(all_batches))
        return all_batches

    def _prepare_if_needed(self):
        """一次性预处理：编码 video/user/root、token 频次、停用集合与位图等。"""
        if self.keep_idx is not None:
            return
        df = self.df
        video_keys = (df["character"].astype(str) + "||" + df["video_type"].astype(str)).astype("category")
        video_id = video_keys.cat.codes.to_numpy(np.int64)
        user_id = df["用户ID"].astype(str).astype("category").cat.codes.to_numpy(np.int64)
        cid = df["评论ID"].to_numpy(np.int64)
        pid = df["上级评论ID"].to_numpy(np.int64)
        root_cid = compute_root_ids_vectorized(cid, pid)
        root_id = pd.Series(root_cid).astype("category").cat.codes.to_numpy(np.int64)

        # texts = df["评论内容"].astype(str).tolist()
        # all_ids: List[np.ndarray] = []
        # for i in range(0, len(texts), 8192):
        #     enc = self.tok(
        #         texts[i:i + 8192],
        #         add_special_tokens=False,
        #         truncation=True,
        #         max_length=self.MAX,
        #         return_attention_mask=False,
        #     )
        #     all_ids.extend([np.asarray(x, dtype=np.int32) for x in enc["input_ids"]])
        # tok_lens = np.array([len(x) for x in all_ids], dtype=np.int32)

        all_ids = [
            np.asarray(x[:self.MAX], dtype=np.int32)
            for x in df["input_ids"].tolist()
        ]
        tok_lens = np.array(df['__tok_len__'])

        keep = (tok_lens >= self.hard_min) & (tok_lens <= self.MAX)
        keep_idx = np.nonzero(keep)[0].astype(np.int64)

        self.keep_idx = keep_idx
        self.video_id = video_id[keep_idx]
        self.user_id = user_id[keep_idx]
        self.root_id = root_id[keep_idx]
        self.tok_lens = tok_lens[keep_idx]
        kept_ids = [all_ids[i] for i in keep_idx]

        # token 频次统计，用于构造停用集合
        vocab_size = len(self.tok)
        freq = np.zeros(vocab_size, dtype=np.int64)
        for ids in kept_ids:
            if ids.size:
                freq += np.bincount(ids, minlength=vocab_size)

        special = {
            x for x in [
                getattr(self.tok, "cls_token_id", None),
                getattr(self.tok, "sep_token_id", None),
                getattr(self.tok, "bos_token_id", None),
                getattr(self.tok, "eos_token_id", None),
                getattr(self.tok, "pad_token_id", None),
                getattr(self.tok, "unk_token_id", None),
            ] if x is not None and x >= 0
        }
        freq2 = freq.copy()
        for sid in special:
            if 0 <= sid < vocab_size:
                freq2[sid] = -1

        k = min(self.stop_topk, vocab_size)
        approx_top = np.argpartition(
            -freq2,
            kth= k - 1
        )[:k]
        stop_ids = np.array([i for i in approx_top if freq2[i] > 0], dtype=np.int32)
        # 日志：打印停用 token 的具体字符串 + 频次（按频次降序，显示前 30 个）
        stop_ids_list = stop_ids.tolist()
        stop_tokens = self.tok.convert_ids_to_tokens(stop_ids_list)
        pairs = sorted(
            [(tid, tok, int(freq[tid])) for tid, tok in zip(stop_ids_list, stop_tokens)],
            key=lambda x: -x[2],
        )
        top_k = 30
        preview = " | ".join(
            f"{tok}({cnt})" for _, tok, cnt in pairs[:top_k]
        )
        logging.info(
            "[Prep] 停用 token 共 %d 个；Top-%d：%s",
            len(pairs),
            min(top_k, len(pairs)),
            preview,
        )

        # 去停用 + 唯一 token 集（用于 overlap 约束）
        offsets = np.zeros(self.keep_idx.size + 1, dtype=np.int64)
        flat_list = []
        uniq_lens = np.empty(self.keep_idx.size, dtype=np.int32)
        for i, arr in enumerate(kept_ids):
            if arr.size == 0:
                uniq = np.empty(0, dtype=np.int32)
            else:
                mask = ~np.isin(arr, stop_ids)
                filtered = arr[mask]
                uniq = np.unique(filtered) if filtered.size else np.empty(0, dtype=np.int32)
            flat_list.append(uniq)
            offsets[i + 1] = offsets[i] + uniq.size
            uniq_lens[i] = uniq.size

        flat_ids = np.empty(int(offsets[-1]), dtype=np.int32)
        p = 0
        for uniq in flat_list:
            n = uniq.size
            if n:
                flat_ids[p:p + n] = uniq
            p += n

        self.flat_ids = flat_ids
        self.offsets = offsets
        self.uniq_token_lens = uniq_lens

        # 位图近似（1024-bit）用于快速 overlap 检测
        W = self.M_bits // 64
        bitset = np.zeros((self.keep_idx.size, W), dtype=np.uint64)
        mask = self.M_bits - 1
        A, B = np.uint64(11400714819323198485), np.uint64(14029467366897019727)
        for i in range(self.keep_idx.size):
            s = flat_ids[offsets[i]:offsets[i + 1]]
            if s.size == 0:
                continue
            h = (A * s.astype(np.uint64) + B) & np.uint64(mask)
            w = (h >> 6).astype(np.int64)
            b = (h & 63).astype(np.uint64)
            if w.size:
                order = np.argsort(w, kind='mergesort')
                w_sorted, b_sorted = w[order], b[order]
                j = 0
                while j < w_sorted.size:
                    jj = j + 1
                    acc = np.uint64(0)
                    ww = w_sorted[j]
                    while jj < w_sorted.size and w_sorted[jj] == ww:
                        acc |= (np.uint64(1) << b_sorted[jj])
                        jj += 1
                    acc |= (np.uint64(1) << b_sorted[j])
                    bitset[i, ww] |= acc
                    j = jj
        self.bitset_words = bitset
        self.len_bucket = (self.tok_lens // self.bw).astype(np.int32)

    def _grab_pool(self, buckets, bucket_order, ptr, used, grab) -> np.ndarray:
        """先从长句桶再到短句桶抓取候选，尽量充满候选池。"""
        pool, need = [], grab
        for b in bucket_order:
            if need <= 0:
                break
            if b * self.bw < self.pref_min:
                continue
            src = buckets[b]
            p = ptr[b]
            if p >= src.size:
                continue
            take = min(src.size - p, need)
            cand = src[p:p + take]
            cand = cand[~used[cand]]
            if cand.size:
                pool.append(cand)
                need -= cand.size
            ptr[b] = p + take
        if need > 0:
            for b in bucket_order[::-1]:
                if need <= 0:
                    break
                if b * self.bw >= self.pref_min:
                    continue
                src = buckets[b]
                p = ptr[b]
                if p >= src.size:
                    continue
                take = min(src.size - p, need)
                cand = src[p:p + take]
                cand = cand[~used[cand]]
                if cand.size:
                    pool.append(cand)
                    need -= cand.size
                ptr[b] = p + take
        if not pool:
            return np.array([], dtype=np.int64)
        return np.concatenate(pool, axis=0)

    def _rank_block(self, block_rel: np.ndarray, W: int, lut: np.ndarray) -> np.ndarray:
        """
        对一个候选块内的样本按“冲突度 + 长度”排序：
          - 冲突度：近似 overlap / 同用户 / 同视频同 root；
          - 长度：偏向更长的句子。
        """
        k = block_rel.size
        vids = self.video_id[block_rel]
        roots = self.root_id[block_rel]
        users = self.user_id[block_rel]
        lens = self.tok_lens[block_rel]

        A = self.bitset_words[block_rel]
        inter = np.zeros((k, k), dtype=np.uint16)
        for w in range(W):
            X = A[:, w][:, None] & A[:, w][None, :]
            Xu8 = X.view(np.uint8).reshape(k, k, 8)
            inter += lut[Xu8].astype(np.uint16).sum(axis=2)

        denom = np.minimum(
            self.uniq_token_lens[block_rel][:, None],
            self.uniq_token_lens[block_rel][None, :],
        ).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_approx = inter.astype(np.float32) / np.maximum(denom, 1.0)

        thresh = self.tau * self.margin
        sus = (ratio_approx > thresh)
        np.fill_diagonal(sus, False)
        same_user = (users[:, None] == users[None, :])
        np.fill_diagonal(same_user, False)
        same_vid_root = (vids[:, None] == vids[None, :]) & (roots[:, None] == roots[None, :])
        np.fill_diagonal(same_vid_root, False)

        deg = (sus | same_user | same_vid_root).sum(axis=1)
        return np.lexsort((-lens, deg))

    def _ok_with_carry(self, idx_rel: int, carry_rel: List[int], W: int, lut: np.ndarray) -> bool:
        """
        检查候选样本与当前 carry 中所有样本的 overlap 是否满足阈值。
        先用位图近似筛选，再用精确交集确认。
        """
        if not carry_rel:
            return True
        row = self.bitset_words[idx_rel][None, :]
        C = self.bitset_words[np.array(carry_rel, dtype=np.int64)]
        inter_words = (C & row)
        Xu8 = inter_words.view(np.uint8).reshape(inter_words.shape[0], W, 8)
        inter_cnt = lut[Xu8].astype(np.uint16).sum(axis=2).sum(axis=1).astype(np.int32)
        denom = np.minimum(
            self.uniq_token_lens[idx_rel],
            self.uniq_token_lens[np.array(carry_rel, dtype=np.int64)],
        )
        ratio_approx = inter_cnt.astype(np.float32) / np.maximum(denom.astype(np.float32), 1.0)

        sus_mask = (ratio_approx > (self.tau * self.margin))
        if not sus_mask.any():
            return True

        flat, off = self.flat_ids, self.offsets

        def exact_ok(j_rel: int) -> bool:
            a = flat[off[idx_rel]:off[idx_rel + 1]]
            b = flat[off[j_rel]:off[j_rel + 1]]
            if a.size == 0 or b.size == 0:
                return True
            p = q = inter = 0
            while p < a.size and q < b.size:
                if a[p] == b[q]:
                    inter += 1
                    p += 1
                    q += 1
                elif a[p] < b[q]:
                    p += 1
                else:
                    q += 1
            d = float(min(a.size, b.size))
            return (inter / d) <= self.tau if d > 0 else True

        for j in np.array(carry_rel, dtype=np.int64)[sus_mask]:
            if not exact_ok(int(j)):
                return False
        return True


def train_mnr_precomputed(
    st_model: SentenceTransformer,
    df: pd.DataFrame,
    enc_tok,
    cfg: Config,
    eval_texts,
    mlm_model=None,
    k_per_video: int = 3,
):
    """
    MNR Precomputed：使用 SimCSEBatchBuilderCarryK 预先构造满足约束的 batch，
    再按标准 DataLoader 训练。单次多 epoch 训练，通过 evaluator 在每个 epoch 结束时测试。
    """
    mnr_loss = losses.MultipleNegativesRankingLoss(st_model)

    builder = SimCSEBatchBuilderCarryK(
        df=df,
        tokenizer=enc_tok,
        MAX_SEQ_LEN=cfg.max_seq_len,
        batch_size=cfg.mnr_batch_size,
        epoch_seed=SEED,
        length_bucket_width=8,
        hard_min_len=4,
        prefer_min_len_ratio=0.6,
        overlap_ratio_thresh=0.25,
        max_per_video_per_batch=k_per_video,
        stop_topk=2,
        candidate_chunk_mul=16,
        bitset_bits=1024,
        suspicious_margin=0.7,
        rounds=3,
    )
    batches = builder.build_epoch_batches()
    if not batches:
        logging.info("Precomputed-MNR 无满批，跳过训练。")
        return

    order_idx = np.concatenate(batches, axis=0)

    class _Pairs(torch.utils.data.Dataset):
        """按给定行号序列构造 SimCSE 同句样本。"""

        def __init__(self, _df, order, col='评论内容'):
            self.df = _df
            self.order = np.asarray(order, dtype=np.int64)
            self.col = col
            self.texts = _df[col].tolist()

        def __len__(self):
            return self.order.shape[0]

        def __getitem__(self, i):
            s = str(self.texts[int(self.order[i])])
            return InputExample(texts=[s, s])

    ds_flat = _Pairs(df, order_idx, '评论内容')
    loader = DataLoader(
        ds_flat,
        batch_size=cfg.mnr_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=st_model.smart_batching_collate,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.mnr_epochs
    warmup_steps = math.ceil(total_steps * cfg.mnr_warmup_ratio) if hasattr(cfg, "mnr_warmup_ratio") else math.ceil(
        total_steps * cfg.warmup_ratio
    )

    evaluator = PerEpochSimCseEvaluator(
        eval_texts=eval_texts,
        mlm_model=mlm_model,
        st_word_module=st_model[0],
        tag_prefix="After-MNR-Precomputed-ep",
    )

    logging.info(
        "Precomputed-MNR 训练 | epochs=%d | steps/epoch=%d | total_steps=%d | warmup=%d",
        cfg.mnr_epochs,
        steps_per_epoch,
        total_steps,
        warmup_steps,
    )

    st_model.fit(
        train_objectives=[(loader, mnr_loss)],
        epochs=cfg.mnr_epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        # evaluation_steps=steps_per_epoch,
        optimizer_params={"lr": cfg.mnr_lr},
        show_progress_bar=True,
        use_amp=False,
    )


# ---------------------------
# 10) 主流程：MLM -> TSDAE -> 两路 SimCSE
# ---------------------------
# (A) 初始基线测试：Before-MLM（ST 初始 + MLM 初始）
mlm_tokenizer, mlm_model = build_mlm_with_tokens(
    CFG.encoder_name,
    hot_emojis,
    init_vectors,
    ensure_token=CFG.emoji_unk_token,
)
run_stage_tests(
    "Before-MLM",
    model,
    eval_texts,
    mlm_model=mlm_model,
    mlm_tokenizer=mlm_tokenizer,
)

# (B) MLM 训练 + 每 epoch 测试 + After-MLM 汇总测试
mlm_output_dir = os.path.join(OUTPUT_DIR, "mlm")
train_mlm(
    mlm_tokenizer,
    mlm_model,
    model,
    train_texts,
    eval_texts,
    CFG,
    mlm_output_dir,
)

sync_st_from_mlm_base_inplace(model, mlm_model)
run_stage_tests(
    "After-MLM",
    model,
    eval_texts,
    mlm_model=mlm_model,
    st_word_module=model[0],
)
word = models.Transformer(mlm_output_dir, max_seq_length=CFG.max_seq_len)
pool = models.Pooling(word.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[word, pool])
model.max_seq_length = CFG.max_seq_len
# (C) TSDAE 两阶段：Phase-1 结束 & Phase-2 每个 epoch 后测试
run_tsdae_two_phase(
    model,
    train_dataset,
    eval_dataset,
    CFG,
    eval_texts,
    mlm_model=mlm_model,
)

# TSDAE 完成后保存一个“对比起点”快照
tsdae_ckpt_dir = os.path.join(OUTPUT_DIR, "final_tsdae")
model.save(tsdae_ckpt_dir)
logging.info("已保存 TSDAE encoder 到: %s", tsdae_ckpt_dir)

# (D) 从 TSDAE 起点分叉两条 SimCSE 分支
st_mnr_baseline = SentenceTransformer(tsdae_ckpt_dir)
st_mnr_precomp = SentenceTransformer(tsdae_ckpt_dir)

# (E) MNR Baseline 分支：单次多 epoch + 每个 epoch 测试
dedup_texts = pd.Series(train_texts).drop_duplicates().tolist()
train_mnr_baseline(
    st_mnr_baseline,
    dedup_texts,
    CFG,
    eval_texts,
    mlm_model=mlm_model,
)

# (F) MNR Precomputed 分支：单次多 epoch + 每个 epoch 测试
train_mnr_precomputed(
    st_mnr_precomp,
    df,
    enc_tok,
    CFG,
    eval_texts,
    mlm_model=mlm_model,
    k_per_video=3,
)

# (G) 保存最终三个 encoder 版本
final_baseline_dir = os.path.join(OUTPUT_DIR, "final_mnr_baseline")
final_precomp_dir = os.path.join(OUTPUT_DIR, "final_mnr_precomputed")

st_mnr_baseline.save(final_baseline_dir)
st_mnr_precomp.save(final_precomp_dir)

logging.info("已保存 MNR Baseline encoder 到: %s", final_baseline_dir)
logging.info("已保存 MNR Precomputed encoder 到: %s", final_precomp_dir)

# (H) 输出所有测试结果为单一 CSV
logging.info("已保存测试指标 CSV: %s", METRICS_CSV)
