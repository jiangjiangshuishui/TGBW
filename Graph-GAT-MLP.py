# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ========= 运行/加速配置 =========
cancerType = "PRAD"
DATA_ROOT = r"C:/Users/11402/Desktop/NIGCNDriver-master"
OUT_DIR = f"{DATA_ROOT}/{cancerType}/Cancer_List/result/MULTINET邻居15加0.6"
KFOLDS = 5  # 折数（5 是经验好用值）
FAST_MODE = True  # True: 训练用全图结构，只在loss处mask（最快）；False: 严格结构留出
EPOCHS = 500
PATIENCE = 20
LR = 5e-4
EMBED_DIM = 96  # 稍降维，进一步提速（可改回128）
KERNEL_DIM = 48
BETA = 1500.0  # 正样本权重
DECODER = "cosine"  # "cosine"更快；可改为"corr"（带行中心化）或"dot"
ALPHA = 2.0  # 仅 corr 时生效

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
USE_AMP = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ========= 读数据 =========
exp = pd.read_table(f"{DATA_ROOT}/{cancerType}/Cancer_List/exp3.txt", header=0, index_col=0, sep="\t")
mut = pd.read_table(f"{DATA_ROOT}/{cancerType}/Cancer_List/mut3.txt", header=0, index_col=0, sep="\t")
adj_df = pd.read_table(f"{DATA_ROOT}/{cancerType}/Cancer_List/mut_driver3.txt", header=0, index_col=0, sep="\t")

G, N = adj_df.shape
print("Device:", device, "| G:", G, "N:", N)

Xg = torch.tensor(mut.values, dtype=torch.float32, device=device)  # [G,N]
Xs = torch.tensor(exp.values, dtype=torch.float32, device=device)  # [G,N]
Adj = torch.tensor(adj_df.values, dtype=torch.float32, device=device)  # [G,N]

# ========= 固定邻居采样 =========
def sample_neighbors(adj_matrix, num_neighbors):
    """
    固定邻居采样，每个节点只选择一定数量的邻居进行聚合
    :param adj_matrix: 邻接矩阵
    :param num_neighbors: 每个节点采样的邻居数量
    :return: 采样后的邻接矩阵
    """
    row, col = adj_matrix.nonzero(as_tuple=True)
    num_nodes = adj_matrix.shape[0]
    sampled_neighbors = torch.zeros_like(adj_matrix, dtype=torch.float32)

    for node in range(num_nodes):
        neighbors = col[row == node]
        if len(neighbors) > num_neighbors:
            sampled_neighbors_for_node = neighbors[:num_neighbors]
        else:
            sampled_neighbors_for_node = neighbors
        sampled_neighbors[node, sampled_neighbors_for_node] = 1.0

    return sampled_neighbors

# 固定邻居数目
NUM_NEIGHBORS = 15 # 每个节点固定选择20个邻居
sampled_adj = sample_neighbors(Adj, NUM_NEIGHBORS)

# ========= 预计算 归一化邻接（稀疏 CSR） =========
@torch.no_grad()
def build_sparse_aggr(adj_dense: torch.Tensor):
    # gene <- sample : A_gs = D_g^-1 * A
    d_g = adj_dense.sum(dim=1).clamp_min(1.0)  # [G]
    A_gs = adj_dense / d_g.unsqueeze(1)  # [G,N]
    # sample <- gene : A_sg = D_s^-1 * A^T
    d_s = adj_dense.sum(dim=0).clamp_min(1.0)  # [N]
    A_sg = adj_dense.t() / d_s.unsqueeze(1)  # [N,G]
    # 转 CSR 稀疏（需要torch>=1.12; 若不支持，可退回 dense 版）
    A_gs_sp = A_gs.to_sparse_csr()
    A_sg_sp = A_sg.to_sparse_csr()
    return A_gs_sp, A_sg_sp


A_gs_sp, A_sg_sp = build_sparse_aggr(sampled_adj if FAST_MODE else (Adj * 0.0))  # FAST_MODE=False 时会在每折重新构

# ========= GraphSAGE 层与模型继续保持不变 =========
class SAGEBipartiteLayer(nn.Module):
    def __init__(self, A_gs_sp, A_sg_sp, in_g_self, in_s_self, in_from_s_to_g, in_from_g_to_s, embed_dim, act=F.relu,
                 bias=False):
        super().__init__()
        self.A_gs_sp = A_gs_sp
        self.A_sg_sp = A_sg_sp
        self.act = act
        self.Wg_self = nn.Linear(in_g_self, embed_dim, bias=bias)
        self.Wg_neigh = nn.Linear(in_from_s_to_g, embed_dim, bias=bias)
        self.Ws_self = nn.Linear(in_s_self, embed_dim, bias=bias)
        self.Ws_neigh = nn.Linear(in_from_g_to_s, embed_dim, bias=bias)

    def forward(self, Xg_self: torch.Tensor, Xs_self: torch.Tensor,
                Xs_for_g: torch.Tensor, Xg_for_s: torch.Tensor):
        # 稀疏乘法
        neigh_g = torch.sparse.mm(self.A_gs_sp, Xs_for_g)  # [G, Fs]
        neigh_s = torch.sparse.mm(self.A_sg_sp, Xg_for_s)  # [N, Fg]
        out_g = self.Wg_self(Xg_self) + self.Wg_neigh(neigh_g)
        out_s = self.Ws_self(Xs_self) + self.Ws_neigh(neigh_s)
        return self.act(out_g), self.act(out_s)

# 其余部分的模型架构与之前相同


class Decoder(nn.Module):
    def __init__(self, embed_dim, kernel_dim, mode="cosine", alpha=2.0):
        super().__init__()
        self.mode  = mode
        self.lm_x  = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.lm_y  = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # 映射后再相似度，更稳定
        x = self.lm_x(x)  # [G,K]
        y = self.lm_y(y)  # [N,K]

        if self.mode == "dot":
            scores = x @ y.t()
            return torch.sigmoid(scores)

        if self.mode == "cosine":
            x_n = x / (x.norm(dim=1, keepdim=True) + 1e-6)
            y_n = y / (y.norm(dim=1, keepdim=True) + 1e-6)
            cos = x_n @ y_n.t()                         # [-1,1]
            return (cos + 1.0) * 0.5                    # -> [0,1]

        # "corr"（行中心化相关系数，已是优化版，无 GxG/NxN）
        x_c = x - x.mean(dim=1, keepdim=True)
        y_c = y - y.mean(dim=1, keepdim=True)
        lxy = x_c @ y_c.t()                             # [G,N]
        lxx = (x_c * x_c).sum(dim=1)                    # [G]
        lyy = (y_c * y_c).sum(dim=1)                    # [N]
        std = torch.sqrt(lxx).unsqueeze(1) * torch.sqrt(lyy).unsqueeze(0)
        corr = (lxy / (std + 1e-6)).clamp(-1.0, 1.0)
        # 缩放到[0,1]
        sig = torch.sigmoid(self.alpha * corr)
        max_v = torch.sigmoid(torch.tensor(self.alpha, dtype=sig.dtype, device=sig.device))
        min_v = torch.sigmoid(torch.tensor(-self.alpha, dtype=sig.dtype, device=sig.device))
        return (sig - min_v) / (max_v - min_v + 1e-12)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.W_Q = nn.Linear(input_dim, output_dim)  # Query weight
        self.W_K = nn.Linear(input_dim, output_dim)  # Key weight
        self.W_V = nn.Linear(input_dim, output_dim)  # Value weight

    def forward(self, X_g, X_s):
        # Step 1: 生成 Query, Key, Value
        Q = self.W_Q(X_g)  # [G, D] -> [G, K]
        K = self.W_K(X_s)  # [N, D] -> [N, K]
        V = self.W_V(X_s)  # [N, D] -> [N, K]

        # Step 2: 计算 Attention scores (Cosine Similarity 或 Dot Product)
        attention_scores = torch.matmul(Q, K.T)  # [G, N]
        attention_scores = attention_scores / (X_g.size(1) ** 0.5)  # Scaling

        # Step 3: Softmax 归一化，得到 Attention 权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [G, N]

        # Step 4: 使用 Attention 权重加权 Value (V)
        output = torch.matmul(attention_weights, V)  # [G, K]

        return output, attention_weights

class GraphSAGEBipartiteWithAttention(nn.Module):
    def __init__(self, adj, x_feature, y_feature, mask, A_gs_sp, A_sg_sp, embed_dim=EMBED_DIM, kernel_dim=KERNEL_DIM,
                 beta=BETA, decoder=DECODER):
        super().__init__()
        self.adj = adj
        self.x = x_feature  # [G,N]
        self.y = y_feature  # [N,G]
        self.mask = mask
        self.beta = beta

        # 使用注意力机制的层
        self.attention_layer = AttentionLayer(input_dim=embed_dim, output_dim=embed_dim)

        self.layer1 = SAGEBipartiteLayer(A_gs_sp, A_sg_sp,
                                         in_g_self=self.x.size(1), in_s_self=self.y.size(1),
                                         in_from_s_to_g=self.y.size(1), in_from_g_to_s=self.x.size(1),
                                         embed_dim=embed_dim, act=F.relu, bias=False)
        self.layer2 = SAGEBipartiteLayer(A_gs_sp, A_sg_sp,
                                         in_g_self=embed_dim, in_s_self=embed_dim,
                                         in_from_s_to_g=embed_dim, in_from_g_to_s=embed_dim,
                                         embed_dim=embed_dim, act=F.relu, bias=False)

        # === 方法一：SAGE 之后的投影 MLP（基因/样本各一条） ===
        self.proj_gene = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.proj_samp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 解码器
        self.decoder = Decoder(embed_dim, kernel_dim, mode=decoder, alpha=ALPHA)

    def loss_fun(self, pred: torch.Tensor):
        y_true = torch.masked_select(self.adj, self.mask)
        y_pred = torch.masked_select(pred, self.mask)
        pos_w = torch.full_like(y_true, self.beta)
        one_w = torch.ones_like(y_true)
        weight = torch.where(y_true.eq(1), pos_w, one_w)
        return nn.BCELoss(weight=weight, reduction="mean")(y_pred, y_true)

    def forward(self):
        # Step 1: 聚合第一层和第二层GraphSAGE
        g1, s1 = self.layer1(self.x, self.y, self.y, self.x)
        g2, s2 = self.layer2(g1, s1, s1, g1)

        # Step 2: 使用注意力机制计算加权表示
        g2_attention, _ = self.attention_layer(g2, s2)  # 基因特征的加权表示
        s2_attention, _ = self.attention_layer(s2, g2)  # 样本特征的加权表示

        # Step 3: 使用投影 MLP 对加权特征进行处理
        g2 = self.proj_gene(g2_attention)  # [G, d] -> [G, d]
        s2 = self.proj_samp(s2_attention)  # [N, d] -> [N, d]

        # Step 4: 解码并得到预测结果
        pred = self.decoder(g2, s2)  # [G, N] 预测的基因-样本关系（概率）
        return pred


# ========= K 折划分 =========
def kfold_indices(n, k, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


folds = kfold_indices(N, KFOLDS, seed=42)


# ========= 训练（AMP+早停） =========
def train_one_fold(model, epochs=EPOCHS, patience=PATIENCE):
    best_loss, bad, best_state = float("inf"), 0, None
    opt = optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            pred = model()
            loss = model.loss_fun(pred)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if ep % 20 == 0 or ep == epochs:
            print(f"[Train] epoch={ep:03d} loss={loss.item():.6f}")
        cur = float(loss.item())
        if cur + 1e-6 < best_loss:
            best_loss, bad = cur, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] epoch={ep}, best_loss={best_loss:.6f}")
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
        final_pred = model()
    return final_pred


# ========= 主流程：K 折（一次训练→一批列的预测）=========
os.makedirs(OUT_DIR, exist_ok=True)
all_pred = torch.zeros(G, N, dtype=torch.float32, device="cpu")  # 放CPU便于拼接/保存

for fold_id, test_cols in enumerate(folds, 1):
    test_cols = np.array(test_cols, dtype=np.int64)
    print(f"\n===== Fold {fold_id}/{KFOLDS}: test_samples={len(test_cols)} =====")
    # 创建训练掩码，排除测试列
    mask = torch.ones(G, N, dtype=torch.bool, device=device)
    mask[:, torch.from_numpy(test_cols).to(device)] = False

    adj_train = Adj  # 使用全图结构或进行图结构分割（视FAST_MODE而定）

    # 创建模型
    model = GraphSAGEBipartiteWithAttention(
        adj=adj_train, x_feature=Xg, y_feature=Xs.t(), mask=mask,
        A_gs_sp=A_gs_sp, A_sg_sp=A_sg_sp,
        embed_dim=EMBED_DIM, kernel_dim=KERNEL_DIM, beta=BETA, decoder=DECODER
    ).to(device)

    pred_full = train_one_fold(model)  # 计算每折的预测结果
    fold_scores = pred_full[:, torch.from_numpy(test_cols).to(device)].detach().cpu().numpy()  # 获取每折的测试结果
    all_pred[:, test_cols] = torch.from_numpy(fold_scores)

# 保存预测结果
final_df = pd.DataFrame(all_pred.numpy(), index=adj_df.index, columns=adj_df.columns)
final_df.to_csv(f"{OUT_DIR}/final_result_with_attention.txt", sep="\t")
print("\nDone! Saved to:", f"{OUT_DIR}/final_result_with_attention.txt")
