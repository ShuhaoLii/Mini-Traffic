import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
from torch_geometric.data import Data


class ContrastiveGAT (nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_clusters, patch_len):
        super (ContrastiveGAT, self).__init__ ()
        self.gat_conv = GATConv (out_channels, out_channels // num_heads, heads=num_heads, concat=True)
        self.kmeans = KMeans (n_clusters=num_clusters)
        self.projector = nn.Sequential (
            nn.Linear (in_channels * patch_len, out_channels),
            nn.ReLU (),
            nn.Linear (out_channels, out_channels)
        )

    def contrastive_loss(self, z_i, z_j, temperature=0.5):
        batch_size = z_i.size (0)
        z = torch.cat ((z_i, z_j), dim=0)
        sim_matrix = F.cosine_similarity (z.unsqueeze (1), z.unsqueeze (0), dim=2)
        sim_ij = torch.diag (sim_matrix, batch_size)
        sim_ji = torch.diag (sim_matrix, -batch_size)
        positives = torch.cat ((sim_ij, sim_ji), dim=0)

        nominator = torch.exp (positives / temperature)
        denominator = torch.sum (torch.exp (sim_matrix / temperature), dim=1) - torch.exp (
            torch.tensor ([1 / temperature]))
        loss = -torch.log (nominator / denominator).mean ()
        return loss

    def contrastive_learning(self, x):
        z_i = self.projector (x)
        z_j = self.projector (x)  # This should be another augmentation in practice
        loss = self.contrastive_loss (z_i, z_j)
        return loss, z_i

    def forward(self, x):
        batch_size, num_patch, nvars, patch_len = x.size ()

        # 将每个patch作为图的节点，初始化节点特征
        patches = x.view (batch_size * num_patch, -1)  # 展平patch

        # 对比学习找到正对
        loss, z_i = self.contrastive_learning (patches)

        # 无监督聚类
        clusters = self.kmeans.fit_predict (z_i.cpu ().detach ().numpy ())

        # 构造邻接矩阵
        edge_index = []
        for i in range (len (clusters)):
            for j in range (len (clusters)):
                if clusters[i] == clusters[j]:
                    edge_index.append ([i, j])
        edge_index = torch.tensor (edge_index, dtype=torch.long).t ().contiguous ()

        # 创建图数据
        data = Data (x=z_i, edge_index=edge_index)

        # 图注意力卷积操作
        x = self.gat_conv (data.x, data.edge_index)

        # 恢复原来的形状
        x = x.view (batch_size, num_patch, -1)  # 先展平至(batch_size, num_patch, out_channels)
        x = x.view (batch_size, num_patch, nvars, patch_len)

        return x, loss


# 参数示例
in_channels = 64  # 输入通道数
out_channels = 64  # 输出通道数
num_heads = 8  # 注意力头的数量
num_clusters = 10  # 聚类的类别数
patch_len = 32  # patch的长度

model = ContrastiveGAT (in_channels, out_channels, num_heads, num_clusters, patch_len)
input_tensor = torch.rand ((32, 16, 64, 32))  # 示例输入张量
output, loss = model (input_tensor)
print (output.shape)  # 输出形状
print (loss)  # 对比学习的损失
