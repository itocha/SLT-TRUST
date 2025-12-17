from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.Tensor as Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import softmax, spmm
from torch_sparse import SparseTensor

from utils.mc_dropout import MC_Dropout
from utils.slt_modules import PartialFrozenConv2d, PartialFrozenLinear


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()

        self.args = args
        self.num_bayes_layers = args.num_bayes_layers

        conv1_out = int(20 * args.scaling_rate)
        conv2_out = int(20 * args.scaling_rate)
        fc1_out = int(100 * args.scaling_rate)
        fc2_out = int(10)

        in_channels = 3 if args.dataset == "cifar10" else 1

        if args.partial_frozen_slt:
            Conv2d = PartialFrozenConv2d
            Linear = PartialFrozenLinear

            if args.m_init_method == "layer_wise" and isinstance(
                args.pruning_rate, (list, tuple)
            ):
                conv1_pr = args.pruning_rate[0]
                conv2_pr = args.pruning_rate[1]
                fc1_pr = args.pruning_rate[2]
                fc2_pr = args.pruning_rate[3]
            elif hasattr(args.pruning_rate, "item"):  # It's a tensor
                conv1_pr = conv2_pr = fc1_pr = fc2_pr = (
                    args.pruning_rate.item()
                )
            else:
                conv1_pr = conv2_pr = fc1_pr = fc2_pr = args.pruning_rate

            conv1_args = {
                "sparsity": conv1_pr,
                "algo": args.ep_algo,
                "scale_method": "dynamic_scaled",
            }
            conv2_args = {
                "sparsity": conv2_pr,
                "algo": args.ep_algo,
                "scale_method": "dynamic_scaled",
            }
            fc1_args = {
                "sparsity": fc1_pr,
                "algo": args.ep_algo,
                "scale_method": "dynamic_scaled",
            }
            fc2_args = {
                "sparsity": fc2_pr,
                "algo": args.ep_algo,
                "scale_method": "dynamic_scaled",
            }

        else:
            Conv2d = nn.Conv2d
            Linear = nn.Linear
            conv1_args = conv2_args = fc1_args = fc2_args = {}

        self.conv1 = Conv2d(
            in_channels, conv1_out, kernel_size=5, padding="same", **conv1_args
        )
        self.conv2 = Conv2d(
            conv1_out, conv2_out, kernel_size=5, padding="same", **conv2_args
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(7)
        self.softmax = nn.Softmax(dim=1)

        self.fc1_input_size = int(conv2_out * 2 * 2)

        self.fc1 = Linear(self.fc1_input_size, fc1_out, **fc1_args)
        self.fc2 = Linear(fc1_out, fc2_out, **fc2_args)

        self.dropout_layers = nn.ModuleList(
            [
                MC_Dropout(p=args.dropout_rate)
                for _ in range(args.num_bayes_layers)
            ]
        )

    def forward(self, x):

        if self.args.dataset == "mnist":
            x = F.pad(x, (2, 2, 2, 2))

        x = self.conv1(x)
        x = self.relu1(x)

        if self.num_bayes_layers >= 3:
            x = self.dropout_layers[2](x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        if self.num_bayes_layers >= 2:
            x = self.dropout_layers[1](x)

        x = self.maxpool2(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)

        if self.num_bayes_layers >= 1:
            x = self.dropout_layers[0](x)

        x = self.fc2(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, args=None, conv_args=None):
        super(BasicBlock, self).__init__()
        if args and args.partial_frozen_slt:
            Conv2d = PartialFrozenConv2d
        else:
            Conv2d = nn.Conv2d

        self.conv1 = Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            **conv_args,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, **conv_args
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    **conv_args,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.args = args

    def forward(self, x, threshold=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut[0](x) if len(self.shortcut) > 0 else x
        if len(self.shortcut) > 0:
            shortcut = self.shortcut[1](shortcut)
        out += shortcut
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ResNet18, self).__init__()
        self.args = args
        base = int(64 * args.scaling_rate)
        self.in_planes = base

        # Choose Conv2d and Linear implementation based on args
        if args.partial_frozen_slt:
            Conv2d = PartialFrozenConv2d
            Linear = PartialFrozenLinear
            # Handle both tensor and list cases for pruning_rate
            if hasattr(args.pruning_rate, "item"):  # It's a tensor
                pruning_rate_val = args.pruning_rate.item()
            else:
                pruning_rate_val = (
                    args.pruning_rate[0]
                    if isinstance(args.pruning_rate, (list, tuple))
                    else args.pruning_rate
                )
            conv_args = {
                "sparsity": pruning_rate_val,
                "algo": "local_ep",
                "scale_method": "dynamic_scaled",
            }
            linear_args = {
                "sparsity": pruning_rate_val,
                "algo": "local_ep",
                "scale_method": "dynamic_scaled",
            }
        else:
            Conv2d = nn.Conv2d
            Linear = nn.Linear
            conv_args = {}
            linear_args = {}

        # Initial convolution
        self.conv1 = Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, **conv_args
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Create layers with proper filter scaling
        self.layer1 = self._make_layer(base, 2, stride=1, conv_args=conv_args)
        self.layer2 = self._make_layer(
            base * 2, 2, stride=2, conv_args=conv_args
        )
        self.layer3 = self._make_layer(
            base * 4, 2, stride=2, conv_args=conv_args
        )
        self.layer4 = self._make_layer(
            base * 8, 2, stride=2, conv_args=conv_args
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_input_size = base * 8
        self.fc = Linear(self.fc_input_size, num_classes, **linear_args)

        # Initialize dropout layers for Bayesian inference
        self.dropout_layers = nn.ModuleList(
            [
                MC_Dropout(p=args.dropout_rate)
                for _ in range(args.num_bayes_layers)
            ]
        )
        self.num_bayes_layers = args.num_bayes_layers

    def _make_layer(self, planes, num_blocks, stride, conv_args):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                BasicBlock(
                    self.in_planes, planes, stride, self.args, conv_args
                )
            )
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, threshold=None):

        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Layer 1
        out = self.layer1[0](out)
        if self.num_bayes_layers >= 8:
            out = self.dropout_layers[7](out)

        out = self.layer1[1](out)
        if self.num_bayes_layers >= 7:
            out = self.dropout_layers[6](out)

        # Layer 2
        out = self.layer2[0](out)
        if self.num_bayes_layers >= 6:
            out = self.dropout_layers[5](out)

        out = self.layer2[1](out)
        if self.num_bayes_layers >= 5:
            out = self.dropout_layers[4](out)

        # Layer 3
        out = self.layer3[0](out)
        if self.num_bayes_layers >= 4:
            out = self.dropout_layers[3](out)

        out = self.layer3[1](out)
        if self.num_bayes_layers >= 3:
            out = self.dropout_layers[2](out)

        # Layer 4
        out = self.layer4[0](out)
        if self.num_bayes_layers >= 2:
            out = self.dropout_layers[1](out)

        out = self.layer4[1](out)
        if self.num_bayes_layers >= 1:
            out = self.dropout_layers[0](out)

        # Final layers
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args

        # Get input features and number of classes from dataset
        if args.dataset == "cora":
            self.num_features = 1433
            self.num_classes = 7
        else:
            raise ValueError(f"Unsupported dataset for GCN: {args.dataset}")

        scaled_hidden_channels = int(args.hidden_channels * args.scaling_rate)

        # GCN layers
        self.convs = nn.ModuleList()
        if args.partial_frozen_slt:
            self.convs.append(
                GCNConv(
                    self.num_features,
                    scaled_hidden_channels,
                    args=args,
                    layer_index=0,
                )
            )

            for i in range(1, args.num_layers - 1):
                self.convs.append(
                    GCNConv(
                        scaled_hidden_channels,
                        scaled_hidden_channels,
                        args=args,
                        layer_index=i,
                    )
                )

            self.convs.append(
                GCNConv(
                    scaled_hidden_channels,
                    self.num_classes,
                    args=args,
                    layer_index=args.num_layers - 1,
                )
            )
        else:
            self.convs.append(
                GCNConv(self.num_features, scaled_hidden_channels, args=args)
            )

            for _ in range(args.num_layers - 2):
                self.convs.append(
                    GCNConv(
                        scaled_hidden_channels,
                        scaled_hidden_channels,
                        args=args,
                    )
                )

            self.convs.append(
                GCNConv(scaled_hidden_channels, self.num_classes, args=args)
            )

        # Initialize dropout layers for Bayesian inference
        self.num_bayes_layers = args.num_bayes_layers
        self.dropout_layers = nn.ModuleList(
            [
                MC_Dropout(p=args.dropout_rate)
                for _ in range(self.num_bayes_layers)
            ]
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            dropout_index = self.num_bayes_layers - (
                self.args.num_layers - i - 1
            )
            if dropout_index >= 0 and dropout_index < self.num_bayes_layers:
                x = self.dropout_layers[dropout_index](x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            in case :obj:`normalize` is set to :obj:`True`, and not added
            otherwise. (default: :obj:`None`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        args=None,
        layer_index=0,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support "
                f"adding self-loops to the graph when no "
                f"on-the-fly normalization is applied"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.args = args
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.layer_index = layer_index

        if args.m_init_method == "layer_wise":
            # Check if pruning_rate is a tensor, if so convert to list first
            if hasattr(args.pruning_rate, "item"):  # It's a tensor
                self.sparsity = args.pruning_rate.item()
            elif isinstance(args.pruning_rate, (list, tuple)):
                self.sparsity = args.pruning_rate[self.layer_index]
            else:
                self.sparsity = args.pruning_rate
        else:
            # Check if pruning_rate is a list, if so use first element, otherwise use as is
            if isinstance(args.pruning_rate, list):
                self.sparsity = args.pruning_rate[0]
            elif hasattr(args.pruning_rate, "item"):  # It's a tensor
                self.sparsity = args.pruning_rate.item()
            else:
                self.sparsity = args.pruning_rate

        if args.partial_frozen_slt:
            self.lin = PartialFrozenLinear(
                in_channels,
                out_channels,
                sparsity=self.sparsity,
                algo=args.ep_algo,
                scale_method="dynamic_scaled",
            )
            self.bias = None
        else:
            self.lin = Linear(in_channels, out_channels)
            if bias:
                self.bias = Parameter(torch.empty(out_channels))
                zeros(self.bias)
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None
    ) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' received a tuple "
                f"of node features as input while this layer "
                f"does not support bipartite message passing. "
                f"Please try other layers such as 'SAGEConv' or "
                f"'GraphConv' instead"
            )

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.reshape(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


# ---- GATConv ----
class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        args=None,
        layer_index: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        kwargs.setdefault("node_dim", 0)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.attn_dropout_p = dropout
        self.args = args
        self.layer_index = layer_index

        if (
            args is not None
            and getattr(args, "m_init_method", None) == "layer_wise"
        ):
            self.sparsity = args.pruning_rate[self.layer_index]
        else:
            pr = getattr(args, "pruning_rate", 0.0)
            self.sparsity = pr[0] if isinstance(pr, list) else pr

        out_features_all = heads * out_channels
        if args is not None and getattr(args, "partial_frozen_slt", False):
            self.lin = PartialFrozenLinear(
                in_channels,
                out_features_all,
                sparsity=self.sparsity,
                algo=args.ep_algo,
                scale_method="dynamic_scaled",
            )
        else:
            self.lin = Linear(in_channels, out_features_all)

        self.att_l = Parameter(torch.empty(1, heads, out_channels))
        self.att_r = Parameter(torch.empty(1, heads, out_channels))

        if bias:
            if concat:
                self.bias = Parameter(torch.empty(out_features_all))
            else:
                self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        if self.bias is not None:
            zeros(self.bias)

    def _edge_index_from_sparse(self, adj: SparseTensor) -> Tensor:
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0).long()

    def forward(
        self, x: Tensor, edge_index: Adj, *, drop_index: int = -1
    ) -> Tensor:
        if isinstance(edge_index, SparseTensor):
            edge_index = self._edge_index_from_sparse(edge_index)

        # Set drop_index for PartialFrozenLinear if applicable
        if hasattr(self.lin, "set_drop_index"):
            self.lin.set_drop_index(drop_index)

        x = self.lin(x)
        N = x.size(0)
        x = x.reshape(N, self.heads, self.out_channels)

        alpha_l = (x * self.att_l).sum(dim=-1)
        alpha_r = (x * self.att_r).sum(dim=-1)

        out = self.propagate(
            edge_index, x=x, alpha=(alpha_l, alpha_r), size=None
        )

        if self.concat:
            out = out.reshape(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(
        self,
        x_j: Tensor,
        alpha_i: Tensor,
        alpha_j: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.attn_dropout_p, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.dataset == "cora":
            self.num_features = 1433
            self.num_classes = 7
        elif args.dataset == "citeseer":
            self.num_features = 3703
            self.num_classes = 6
        elif args.dataset == "pubmed":
            self.num_features = 500
            self.num_classes = 3
        elif args.dataset == "ogbn-arxiv":
            self.num_features = 128
            self.num_classes = 40
        elif args.dataset == "ogbg-molhiv":
            self.num_features = getattr(
                args, "emb_dim", 100
            )
            self.num_classes = 2
        elif args.dataset == "ogbg-molbace":
            self.num_features = getattr(
                args, "emb_dim", 100
            )
            self.num_classes = 2
        else:
            raise ValueError(f"Unsupported dataset for GAT: {args.dataset}")

        scaled_hidden_channels = int(args.hidden_channels * args.scaling_rate)

        # Graph classification specific layers
        self.is_graph_classification = args.dataset in [
            "ogbg-molhiv",
            "ogbg-molbace",
        ]

        heads = getattr(args, "gat_heads", 8)
        out_heads = getattr(args, "gat_out_heads", 1)
        concat = getattr(args, "gat_concat", True)
        negative_slope = getattr(args, "gat_negative_slope", 0.2)
        attn_dropout = getattr(args, "attn_dropout", 0.5)

        def per_head_out(total_hidden: int, heads: int, concat: bool) -> int:
            if concat:
                if total_hidden % heads != 0:
                    raise ValueError(
                        f"hidden_channels({total_hidden}) must be divisible by heads({heads}) when concat=True"
                    )
                return total_hidden // heads
            else:
                return total_hidden

        hidden_per_head = per_head_out(scaled_hidden_channels, heads, concat)

        self.convs = nn.ModuleList()
        if args.partial_frozen_slt:
            self.convs.append(
                GATConv(
                    self.num_features,
                    hidden_per_head,
                    heads=heads,
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=attn_dropout,
                    args=args,
                    layer_index=0,
                )
            )
            for i in range(1, args.num_layers - 1):
                in_dim = scaled_hidden_channels if concat else hidden_per_head
                self.convs.append(
                    GATConv(
                        in_dim,
                        hidden_per_head,
                        heads=heads,
                        concat=concat,
                        negative_slope=negative_slope,
                        dropout=attn_dropout,
                        args=args,
                        layer_index=i,
                    )
                )
            # 出力層は通常 concat=False, heads=out_heads
            if self.is_graph_classification:
                self.convs.append(
                    GATConv(
                        scaled_hidden_channels if concat else hidden_per_head,
                        hidden_per_head,
                        heads=out_heads,
                        concat=False,
                        negative_slope=negative_slope,
                        dropout=attn_dropout,
                        args=args,
                        layer_index=args.num_layers - 1,
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        scaled_hidden_channels if concat else hidden_per_head,
                        self.num_classes,
                        heads=out_heads,
                        concat=False,
                        negative_slope=negative_slope,
                        dropout=attn_dropout,
                        args=args,
                        layer_index=args.num_layers - 1,
                    )
                )
        else:
            self.convs.append(
                GATConv(
                    self.num_features,
                    hidden_per_head,
                    heads=heads,
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=attn_dropout,
                    args=args,
                )
            )
            for _ in range(args.num_layers - 2):
                in_dim = scaled_hidden_channels if concat else hidden_per_head
                self.convs.append(
                    GATConv(
                        in_dim,
                        hidden_per_head,
                        heads=heads,
                        concat=concat,
                        negative_slope=negative_slope,
                        dropout=attn_dropout,
                        args=args,
                    )
                )
            if self.is_graph_classification:
                self.convs.append(
                    GATConv(
                        scaled_hidden_channels if concat else hidden_per_head,
                        hidden_per_head,
                        heads=out_heads,
                        concat=False,
                        negative_slope=negative_slope,
                        dropout=attn_dropout,
                        args=args,
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        scaled_hidden_channels if concat else hidden_per_head,
                        self.num_classes,
                        heads=out_heads,
                        concat=False,
                        negative_slope=negative_slope,
                        dropout=attn_dropout,
                        args=args,
                    )
                )

        self.num_bayes_layers = args.num_bayes_layers
        self.dropout_layers = nn.ModuleList(
            [
                MC_Dropout(p=args.dropout_rate)
                for _ in range(self.num_bayes_layers)
            ]
        )

        # Graph classification specific layers
        if self.is_graph_classification:
            # Global pooling and final classification layer
            final_hidden_dim = (
                hidden_per_head if not concat else scaled_hidden_channels
            )
            self.classifier = nn.Linear(final_hidden_dim, self.num_classes)

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            drop_index = self.num_bayes_layers - (self.args.num_layers - i - 1)
            x = conv(x, edge_index, drop_index=drop_index)
            x = F.elu(x)
            if drop_index >= 0 and drop_index < self.num_bayes_layers:
                x = self.dropout_layers[drop_index](x)

        if self.is_graph_classification:
            x = self.convs[-1](x, edge_index)
            x = F.elu(x)

            if batch is not None:
                from torch_geometric.nn import global_mean_pool

                x = global_mean_pool(x, batch)
            else:
                x = x.mean(dim=0, keepdim=True)

            x = self.classifier(x)
            return F.log_softmax(x, dim=1)
        else:
            x = self.convs[-1](x, edge_index)
            return F.log_softmax(x, dim=1)
