
# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath




class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        
        #print(f"x shape :{x.shape}")
        
        b, c, n = x.shape
        
        #b, c, n, _ = x.shape
        x_j = x_j.squeeze(-1)
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, -1)
        
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        #print(f"edge_index:{edge_index.shape}")
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value
        


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        
        x=x.unsqueeze(-1)
        #print(x.shape)
        #print(x_j.shape)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)

class MultiHeadEdgeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, act='relu', norm=None, bias=True):
        super(MultiHeadEdgeConv2d, self).__init__()
        self.heads = nn.ModuleList([EdgeConv2d(in_channels, out_channels, act, norm, bias) for _ in range(num_heads)])

    def forward(self, x, edge_index, y=None):
        head_outputs = [head(x, edge_index, y) for head in self.heads]
        return torch.cat(head_outputs, dim=1)  # Concatenate outputs of all heads


class EdgeConv2d_global(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) with global feature fusion
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d_global, self).__init__()
        # 局部卷积网络
        self.nn = BasicConv([in_channels * 2 + in_channels, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # edge_index[1] 指代当前节点，edge_index[0] 指代邻居节点
        x_i = batched_index_select(x, edge_index[1])  # 当前节点的特征
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])  # 邻居节点特征
        else:
            x_j = batched_index_select(x, edge_index[0])
        
        # 计算相对特征差异
        edge_features = x_j - x_i  # 邻居特征与当前特征的差异

        # --- 全局特征融合部分 ---
        # 全局池化 (Global Pooling)，获取全局特征
        global_feature = torch.mean(x, dim=1, keepdim=True)  # 全局平均池化，也可以改为最大池化
        
        # 重复全局特征，使其维度与局部特征一致
        global_feature_repeated = global_feature.repeat(1, x_i.size(1), 1)  # 全局特征扩展到每个节点
        
        # 将局部特征和全局特征进行拼接
        combined_features = torch.cat([x_i, edge_features, global_feature_repeated], dim=1)
        
        # 卷积操作
        max_value, _ = torch.max(self.nn(combined_features), -1, keepdim=True)
        return max_value


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        #print(f"r:{r}")
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, N = x.shape  
        y = None
        if self.r > 1:
            y = F.avg_pool1d(x, self.r)  
            y = y.reshape(B, C, -1).contiguous()
            #print(f"y:{y.shape}")
        x = x.reshape(B, C, -1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, N).contiguous() 

    # def forward(self, x, relative_pos=None):
    #     B, C, H, W = x.shape
    #     y = None
    #     if self.r > 1:
    #         y = F.avg_pool2d(x, self.r, self.r)
    #         y = y.reshape(B, C, -1, 1).contiguous()
    #     x = x.reshape(B, C, -1, 1).contiguous()
    #     edge_index = self.dilated_knn_graph(x, y, relative_pos)
    #     x = super(DyGraphConv2d, self).forward(x, edge_index, y)
    #     return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='mr', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=500, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        dilation=1
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        
        
        self.relative_pos = None
        


  
    def forward(self, x):
        _tmp = x
        
        #print(f"torch_vertex {x.shape}")
        B, C, N = x.shape
        
        
        
        
        x = self.fc1(x)
        B, C, N = x.shape  
        
        relative_pos = None
        
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x