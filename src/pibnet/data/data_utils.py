import dgl
import torch
import numpy as np
from typing import Tuple, List, Union


def get_edge_dir_and_dist(
    edges_src: torch.tensor, 
    edges_dst: torch.tensor, 
    positions: torch.tensor
):
    """
    Gets the direction and the length of an edge

    Parameters
    ----------
    edges_src: torch.tensor
        Source node indexes of the edges 
    edges_dst: torch.tensor
        Destination node indexes of the edges
    positions: torch.tensor
        Node positions 
    """
    rel_pos = positions[edges_dst] - positions[edges_src]
    dist = torch.linalg.norm(rel_pos, axis=-1, keepdim=True)
    rel_dir = rel_pos / dist
    return rel_dir, dist 

def add_new_edges(
    g: dgl.DGLGraph, 
    new_src: torch.tensor, 
    new_dst: torch.tensor, 
    etype: str, 
    sample: dict, 
    cfg
) -> dgl.DGLGraph:
    """
    Adds new edges to a graph

    Parameters
    ----------
    g: dgl.DGLGraph
        Graph to add edges to
    new_src: torch.tensor
        Source node indexes of the new edges 
    new_dst: torch.tensor
        Destination node indexes of the new edges
    etype: str
        Type of the new edges
    sample: dict
        Parameters of the dataset sample
    cfg: config
    """
    positions = g.ndata['position']

    rel_dir, dist = get_edge_dir_and_dist(new_src, new_dst, positions)
    data_new_edges = {
        'rel_dir': rel_dir,
        'dist': dist,
    }
    if cfg.data.equation == 'helmholtz':
        data_new_edges['wavelength'] = sample['source']['wavelength'] * torch.ones((len(new_src), 1))
        
        data_new_edges['phase'] = torch.cat([
            torch.sin(2 * torch.pi / sample['source']['wavelength'] * dist), 
            torch.cos(2 * torch.pi / sample['source']['wavelength'] * dist)
        ], axis=-1)
    elif cfg.data.equation == 'laplace':
        data_new_edges['amplitudes'] = torch.tensor(sample['source']['amplitudes']).view(1, -1) * torch.ones((len(new_src), 1))
         

    g.add_edges(
        new_src, 
        new_dst, 
        data=data_new_edges, 
        etype=etype
    )
    return dgl.to_simple(g, copy_ndata=True, copy_edata=True)


def get_edges_to_add(
    g: dgl.DGLGraph, 
    new_edges_ratio: float, 
    candidate_edge_ratio: int = 1, 
    node_mask: torch.tensor = None
) -> Tuple[torch.tensor]:
    """
    Return the node indexes of new edge sources and destinations

    Parameters
    ----------
    g: dgl.DGLGraph
        Graph to add edges to
    new_edges_ratio: float
        Ratio of new edges to add relative to the total number of nodes
    candidate_edge_ratio: float
        Number of edge proposal for each new edge to add, by default 1
    node_mask: torch.tensor
        Mask to select nodes that must taken into account, by default None
    """
    
    orig_nodes = g.nodes()[node_mask] if node_mask is not None else g.nodes()
    num_nodes = len(orig_nodes)
    new_nodes = torch.arange(num_nodes)
    
    positions = g.ndata['position'][node_mask] if node_mask is not None else g.ndata['position']

    assert new_edges_ratio <= 1, "the ratio of new edge to add per available node must be <= 1"

    num_candidate_edges = new_edges_ratio * num_nodes
    if num_candidate_edges < 1:
        new_src = torch.randperm(num_nodes, dtype=torch.int32)[:int(num_nodes * num_candidate_edges)]
    else:
        num_candidate_edges = int(np.round(num_candidate_edges))
        new_src = new_nodes.unsqueeze(-1).repeat(1, num_candidate_edges).reshape(-1)

    if candidate_edge_ratio > 1:
        new_dst = torch.stack([
            torch.randint_like(new_src, low=0, high=num_nodes, dtype=torch.int32) for i in range(candidate_edge_ratio)
        ], dim=-1)
        new_rel_pos = positions[new_dst] - positions[new_src].reshape(-1, 1, 3)
        new_dist = torch.linalg.norm(new_rel_pos, axis=-1)
        edges_to_keep = torch.min(new_dist, dim=-1)[1]
        new_dst = new_dst[torch.arange(new_dst.size(0), device=new_dst.device), edges_to_keep]
    else:
        new_dst = torch.randint_like(new_src, low=0, high=num_nodes, dtype=torch.int32)

    # remove self loops
    self_loops = new_src == new_dst
    new_src, new_dst = new_src[~self_loops], new_dst[~self_loops]

    # Add reverse edges
    new_src, new_dst = torch.cat([new_src, new_dst]), torch.cat([new_dst, new_src])
    return orig_nodes[new_src], orig_nodes[new_dst]