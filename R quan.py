import numpy as np
import pandas as pd
import sys

cancerType = "LUAD"

# -------- 读取数据 --------
snp = pd.read_table(
    filepath_or_buffer=f'C:/Users/11402/Desktop/NIGCNDriver-master/{cancerType}/Cancer_List/mut9.txt',
    header=0, index_col=0, sep='\t'
)
final_result = pd.read_table(
    filepath_or_buffer=f'C:/Users/11402/Desktop/NIGCNDriver-master/{cancerType}/Cancer_List/result/one graphsage/final_result_with_attention.txt',
    header=0, index_col=0, sep='\t'
)

# 基因/样本索引
genes = final_result.index.to_numpy()
patients = final_result.columns.to_numpy()

# 转 NumPy，加速后续操作
scores = final_result.to_numpy()  # shape: (G, P)
mutarr = snp.reindex(index=final_result.index, columns=final_result.columns).to_numpy()  # 对齐

G, P = scores.shape

# -------- 构造 patient_genes（每个病人取其突变基因中模型得分 Top 50%）--------
col_lists = []  # 每个元素是一个病人的"基因名列表"
max_len = 0

for j in range(P):
    mut_idx = np.flatnonzero(mutarr[:, j])
    if mut_idx.size == 0:
        col_lists.append([])
        continue

    vals = scores[mut_idx, j]
    k = max(1, mut_idx.size // 2)
    topk_local = np.argpartition(-vals, k - 1)[:k]
    topk_idx = mut_idx[topk_local]
    order = np.argsort(-scores[topk_idx, j])
    topk_idx = topk_idx[order]
    gene_names = genes[topk_idx].tolist()
    col_lists.append(gene_names)
    if len(gene_names) > max_len:
        max_len = len(gene_names)

filled = np.empty((max_len, P), dtype=object)
filled[:] = np.nan
for j, names in enumerate(col_lists):
    if names:
        filled[:len(names), j] = names

patient_genes = pd.DataFrame(filled, columns=patients)

# -------- 候选集合提取 --------
vals = patient_genes.to_numpy().ravel()
mask = pd.notna(vals)
candidate_names = pd.unique(vals[mask]).tolist()
print(f"候选基因数量: {len(candidate_names)}")

# -------- 加权Condorcet评分 --------
def calculate_weighted_condorcet(patient_genes, candidate_names):
    C = len(candidate_names)
    name_to_id = {name: i for i, name in enumerate(candidate_names)}
    weighted_counts = np.zeros((C, C), dtype=np.float64)
    rank_lists = []
    for j in range(patient_genes.shape[1]):
        col = patient_genes.iloc[:, j].dropna().to_numpy()
        if col.size == 0:
            rank_lists.append([])
            continue
        ids = [name_to_id[x] for x in col if x in name_to_id]
        seen = set(); dedup = []
        for t in ids:
            if t not in seen:
                seen.add(t); dedup.append(t)
        rank_lists.append(dedup)

    all_ids = np.arange(C)
    for ids in rank_lists:
        if not ids:
            continue
        n = len(ids)
        weights = np.exp(-0.3* np.arange(n))
        for i, gene1 in enumerate(ids):
            if i < n - 1:
                losers = ids[i + 1:]
                weighted_counts[gene1, losers] += weights[i]
            unv = np.setdiff1d(all_ids, ids, assume_unique=True)
            if unv.size > 0:
                weighted_counts[gene1, unv] += weights[i]

    vote_win = weighted_counts.sum(axis=1)
    vote_loss = weighted_counts.sum(axis=0)
    den = vote_win + vote_loss
    den = np.where(den == 0, 1, den)
    weighted_scores = vote_win / den
    return weighted_scores, weighted_counts

weighted_scores, weighted_matrix = calculate_weighted_condorcet(patient_genes, candidate_names)

# -------- 结果排序和保存 --------
gene_result = pd.DataFrame(
    {
        'weighted_score': weighted_scores,
        'win_count': [weighted_matrix[i].sum() for i in range(len(candidate_names))],
        'loss_count': [weighted_matrix[:, i].sum() for i in range(len(candidate_names))],
        'total_comparisons': [weighted_matrix[i].sum() + weighted_matrix[:, i].sum() for i in range(len(candidate_names))]
    },
    index=candidate_names
).sort_values('weighted_score', ascending=False)

# 获取排名前500的基因
top_500_genes = gene_result.head(500)

# 保存前500基因的排名结果
top_500_genes.to_csv(
    f'C:/Users/11402/Desktop/NIGCNDriver-master/{cancerType}/Cancer_List/result/one graphsage/无权.txt',
    sep='\t'
)

print(f"\n前500个驱动基因已保存到文件!")
print(f"文件路径: C:/Users/11402/Desktop/NIGCNDriver-master/{cancerType}/Cancer_List/result/one graphsage/无权.txt")
