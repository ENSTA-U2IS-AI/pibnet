from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import dgl  # noqa: F401 for docs
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union, Dict
from warnings import warn

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn, aggregate_and_concat
from physicsnemo.models.layers import get_activation
from physicsnemo.models.module import Module
from physicsnemo.utils.profiling import profile

from physicsnemo.models.meshgraphnet.meshgraphnet import MetaData
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP, MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum


class PIBNet(Module):
    """
    PIBNet network architecture

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : Dict[str, int]
        Dictionary with the number of edge features for each edge type
    output_dim : int
        Number of outputs
    num_levels : int
        Number of levels
    first_top_processor_size : int
        Number of message passing blocks in the first top processor block
    last_top_processor_size : int
        Number of message passing blocks in the last top processor block
    bottom_processor_size : int
        Number of message passing blocks in the bottom processor block
    hidden_dim_scaling: int
        Scaling factor of the hidden dimension
    distant_edge_sample_period: int
        Number of message passing apply to each distant edge sample
    mlp_activation_fn : Union[str, List[str]],
        Activation function to use, by default 'relu'
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : Union[int, None], optional
        Number of MLP layers for the node feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no node encoder
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : Union[int, None], optional
        Number of MLP layers for the edge feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no edge encoder
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : Union[int, None], optional
        Number of MLP layers for the node feature decoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no decoder
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_concat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0 (checkpointing disabled)
    checkpoint_offloading: bool, optional
        Whether to offload the checkpointing to the CPU, by default False
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: Dict[str, int],
        output_dim: int,
        num_levels: int,
        first_top_processor_size: int,
        last_top_processor_size: int,
        bottom_processor_size: List[int],
        hidden_dim_scaling: int,
        distant_edge_sample_period: int,
        mlp_activation_fn: Union[str, List[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        norm_type="LayerNorm",
    ):
        super().__init__(meta=MetaData())

        activation_fn = get_activation(mlp_activation_fn)

        if norm_type not in ["LayerNorm", "TELayerNorm"]:
            raise ValueError("Norm type should be either 'LayerNorm' or 'TELayerNorm'")

        if not torch.cuda.is_available() and norm_type == "TELayerNorm":
            warn("TELayerNorm is not supported on CPU. Switching to LayerNorm.")
            norm_type = "LayerNorm"

        self.etypes = input_dim_edges.keys()

        is_distant_edges_encoder_initialized = False

        self.edge_encoders = nn.ModuleDict()
        for etype, edge_input_dim in input_dim_edges.items():
            init_edge_encoder = True
            if edge_input_dim > 0:
                if etype == 'neighbors':
                    output_dim_processor = hidden_dim_processor

                elif 'distant' in etype:
                    if not is_distant_edges_encoder_initialized:
                        is_distant_edges_encoder_initialized = True
                        output_dim_processor = int(hidden_dim_processor * hidden_dim_scaling ** (num_levels - 1))
                        etype = 'distant'
                    else:
                        init_edge_encoder = False
                else:
                    for i in range(num_levels):
                        if etype in [f"down_{i}_{i+1}", f"up_{i+1}_{i}"]:
                            output_dim_processor = int(hidden_dim_processor * hidden_dim_scaling ** (i+1))
                
                # print(etype, output_dim_processor)
                if init_edge_encoder:
                    self.edge_encoders[etype] = MeshGraphMLP(
                        edge_input_dim,
                        output_dim=output_dim_processor,
                        hidden_dim=hidden_dim_edge_encoder,
                        hidden_layers=num_layers_edge_encoder,
                        activation_fn=activation_fn,
                        norm_type=norm_type,
                        recompute_activation=recompute_activation,
                    )
                
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )
        
        self.long_range_etypes = []
        for etype in set(input_dim_edges.keys()):
            etype = etype.split('_')[0]
            if etype in ['distant']:
                self.long_range_etypes.append(etype)
        self.long_range_etypes = list(set(self.long_range_etypes))

        self.processor = MultiLayerMeshGraphNetProcessor(
            num_levels=num_levels,
            long_range_etypes=self.long_range_etypes,
            first_top_processor_size=first_top_processor_size,
            last_top_processor_size=last_top_processor_size,
            bottom_processor_size=bottom_processor_size,
            distant_edge_sample_period=distant_edge_sample_period,
            input_dim=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
            scaling=hidden_dim_scaling
        )

        self.num_levels = num_levels
        self.hidden_dim = hidden_dim_processor

    @profile
    def forward(
        self,
        node_features: Tensor,
        edge_features: Dict[str, Tensor],
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
        **kwargs,
    ) -> Tensor:
        
        multilevel_node_features = {'0': self.node_encoder(node_features)}

        for etype in self.etypes:
            # print(etype, edge_features[etype].shape, graph.num_edges(etype=etype))
            if etype.split('_')[0] in self.long_range_etypes:
                edge_encoder = self.edge_encoders[etype.split('_')[0]]
            else:
                edge_encoder = self.edge_encoders[etype]
            edge_features[etype] = edge_encoder(edge_features[etype])
            
        x = self.processor(multilevel_node_features, edge_features, graph)
        x = x['0']
        x = self.node_decoder(x)
        return x

class MultiLayerMeshGraphNetProcessor(nn.Module):
    """MeshGraphNet processor block"""

    def __init__(
        self,
        num_levels: int,
        long_range_etypes: List[str],
        first_top_processor_size: int,
        bottom_processor_size: int,
        last_top_processor_size: int,
        distant_edge_sample_period: int,
        input_dim: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.ReLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        scaling: float = 1,
    ):
        super().__init__()
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments
        self.checkpoint_offloading = (
            checkpoint_offloading if (num_processor_checkpoint_segments > 0) else False
        )

        edge_block_invars = dict(
            hidden_layers=num_layers_edge,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=False,
        )
        node_block_invars = dict(
            hidden_layers=num_layers_node,
            activation_fn=activation_fn,
            norm_type=norm_type,
            aggregation=aggregation,
            recompute_activation=False,
        )

        in_layers = []

        # Top layers
        for _ in range(first_top_processor_size):
            in_layers.append(EdgeBlock(
                etype="neighbors",
                level=0,
                input_dim_nodes=input_dim,
                input_dim_edges=input_dim,
                output_dim=input_dim,
                hidden_dim=input_dim,
                **edge_block_invars
            ))
            in_layers.append(NodeBlock(
                etype="neighbors",
                level_in=0,
                level_out=0,
                input_dim_nodes=input_dim,
                input_dim_edges=input_dim,
                output_dim=input_dim,
                hidden_dim=input_dim,
                **node_block_invars
            ))

        # Downward layers
        for level in range(num_levels-1):
            in_layers.append(NodeDimensionExpander(
                level=level,
                input_dim=int(input_dim * scaling ** level),
                output_dim=int(input_dim * scaling ** (level+1)),
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=False
            ))

            in_layers.append(EdgeBlock(
                etype=f"down_{level}_{level+1}",
                level=level,
                input_dim_nodes=int(input_dim * scaling ** level),
                input_dim_edges=int(input_dim * scaling ** (level+1)),
                output_dim=int(input_dim * scaling ** (level+1)),
                hidden_dim=int(input_dim * scaling ** level),
                **edge_block_invars
            ))
            in_layers.append(NodeBlock(
                etype=f"down_{level}_{level+1}",
                level_in=level,
                level_out=level+1,
                input_dim_nodes=int(input_dim * scaling ** level),
                input_dim_edges=int(input_dim * scaling ** (level+1)),
                output_dim=int(input_dim * scaling ** (level+1)),
                hidden_dim=int(input_dim * scaling ** (level+1)),
                **node_block_invars
            ))

        bottom_layers = []
        # Bottom layers
        for _ in range(bottom_processor_size):
            for etype in long_range_etypes:
                bottom_layers.append(EdgeBlock(
                    etype=etype,
                    level=num_levels-1,
                    input_dim_nodes=int(input_dim * scaling ** (num_levels - 1)),
                    input_dim_edges=int(input_dim * scaling ** (num_levels - 1)),
                    output_dim=int(input_dim * scaling ** (num_levels - 1)),
                    hidden_dim=int(input_dim * scaling ** (num_levels - 1)),
                    **edge_block_invars
                ))
            bottom_layers.append(NodeBlock(
                etype=long_range_etypes if len(long_range_etypes) > 1 else long_range_etypes[0],
                level_in=num_levels-1,
                level_out=num_levels-1,
                input_dim_nodes=int(input_dim * scaling ** (num_levels - 1)),
                input_dim_edges=int(input_dim * scaling ** (num_levels - 1)),
                output_dim=int(input_dim * scaling ** (num_levels - 1)),
                hidden_dim=int(input_dim * scaling ** (num_levels - 1)),
                **node_block_invars
            ))

        out_layers = []
        # Upward layers
        for level in range(num_levels-1, 0, -1):

            out_layers.append(AggNodeBlock(level))

            out_layers.append(EdgeBlock(
                etype=f"up_{level}_{level-1}",
                level=level,
                input_dim_nodes=int(input_dim * scaling ** level),
                input_dim_edges=int(input_dim * scaling ** level),
                output_dim=int(input_dim * scaling ** level),
                hidden_dim=int(input_dim * scaling ** level),
                **edge_block_invars
            ))
            out_layers.append(NodeBlock(
                etype=f"up_{level}_{level-1}",
                level_in=level,
                level_out=level-1,
                input_dim_nodes=int(input_dim * scaling ** level),
                input_dim_edges=int(input_dim * scaling ** level),
                output_dim=int(input_dim * scaling ** (level-1)),
                hidden_dim=int(input_dim * scaling ** (level-1)),
                **node_block_invars
            ))

        # Top layers
        for _ in range(last_top_processor_size):
            out_layers.append(EdgeBlock(
                etype="neighbors",
                level=0,
                input_dim_nodes=input_dim,
                input_dim_edges=input_dim,
                output_dim=input_dim,
                hidden_dim=input_dim,
                **edge_block_invars
            ))
            out_layers.append(NodeBlock(
                etype="neighbors",
                level_in=0,
                level_out=0,
                input_dim_nodes=input_dim,
                input_dim_edges=input_dim,
                output_dim=input_dim,
                hidden_dim=input_dim,
                **node_block_invars
            ))


        self.in_layers = nn.ModuleList(in_layers)
        self.bottom_layers = nn.ModuleList(bottom_layers)
        self.out_layers = nn.ModuleList(out_layers)

        self.distant_edge_sample_period = distant_edge_sample_period
        self.bottom_processor_size = bottom_processor_size
        

    @profile
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        
        for module in self.in_layers:
            edge_features, node_features = module(
                edge_features, node_features, graph
            )
        
        index = 0
        for module in self.bottom_layers:
            edge_features, node_features = module(
                edge_features, node_features, graph, edge_index=index // self.distant_edge_sample_period
            )
            if isinstance(module, NodeBlock):
                index += 1
        
        for module in self.out_layers:
            edge_features, node_features = module(
                edge_features, node_features, graph
            )

        return node_features


class EdgeBlock(nn.Module):

    def __init__(
        self,
        etype: str = None,
        level: int = None,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.etype = etype
        self.level = level

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat

        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    @profile
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
        edge_index: int = None,
    ) -> Tensor:
        
        if edge_index is not None:
            etype = f'{self.etype}_{edge_index}'
        else:
            etype = self.etype

        # print('edge_block', etype, efeat[etype].shape, self.level, nfeat[str(self.level)].shape)
        sub_graph = dgl.edge_type_subgraph(graph, [etype])
        efeat_new = self.edge_mlp(efeat[etype], nfeat[str(self.level)], sub_graph)
        efeat[etype] = efeat_new + efeat[etype]
        return efeat, nfeat


class NodeBlock(nn.Module):

    def __init__(
        self,
        etype: str,
        level_in: int = None,
        level_out: int = None,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        aggregation: str = "sum",
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.etype = etype
        self.level_in = level_in
        self.level_out = level_out
        self.aggregation = aggregation

        self.node_mlp = MeshGraphMLP(
        # self.node_mlp = MeshGraphSwiGLUMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
        edge_index: int = None,
    ) -> Tuple[Tensor, Tensor]:
        
        if edge_index is not None:
            if isinstance(self.etype, str):
                etype = f'{self.etype}_{edge_index}'
            elif isinstance(self.etype, list):
                etype = [f'{etype}_{edge_index}' for etype in self.etype]

            if len(etype) == 1:
                etype = etype[0]

        else:
            etype = self.etype
            
        if isinstance(etype, str):
            sub_graph = dgl.edge_type_subgraph(graph, [etype])
            # update edge features
            # print('node_block', etype, efeat[etype].shape, sub_graph.num_edges(etype=etype), graph.num_edges(etype=etype), self.level_in, nfeat[str(self.level_in)].shape, graph.ndata[f"lvl_{self.level_in}"].sum(), self.level_out)
            cat_feat = aggregate_and_concat(efeat[etype], nfeat[str(self.level_in)], sub_graph, self.aggregation)
        
        elif isinstance(etype, list):
            sub_graph = dgl.edge_type_subgraph(graph, etype)
            # update edge features
            # print('node_block', etype, sub_graph.num_edges(etype=etype), graph.num_edges(etype=etype), nfeat[str(self.level_in)].shape, graph.ndata[f"lvl_{self.level_in}"].sum())
            cat_feat = aggregate_and_concat({etype: efeat[etype] for etype in etype}, nfeat[str(self.level_in)], sub_graph, self.aggregation)
        
        # update node features + residual connection
        nfeat[str(self.level_out)] = self.node_mlp(cat_feat) + nfeat[str(self.level_out)]

        return efeat, nfeat

class NodeDimensionExpander(nn.Module):

    def __init__(
        self,
        level: int = None,
        input_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 0,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.level = level

        if input_dim == output_dim:
            self.node_mlp = nn.Identity()

        else:
            self.node_mlp = MeshGraphMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
            )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
                
        proj_nfeat = self.node_mlp(nfeat[str(self.level)])
        nfeat[str(self.level+1)] = proj_nfeat
        nfeat[f"{self.level}_proj"] = proj_nfeat.clone()

        return efeat, nfeat

class AggNodeBlock(nn.Module):

    def __init__(
        self,
        level: int = None,
    ):
        super().__init__()

        self.level = level

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
    
        mask = ~torch.logical_and(graph.ndata[f"lvl_{self.level}"] == 0, graph.ndata[f"lvl_{self.level-1}"] == 1).unsqueeze(-1)
        # print('aggblock', mask.shape)
        nfeat[str(self.level)] = nfeat[str(self.level)] * mask + nfeat[f"{self.level-1}_proj"] * ~mask

        return efeat, nfeat
