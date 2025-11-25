import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import dgl
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import json

from data_gen_utils.utils import timeit
from data_gen_utils.binary_tree import TreeOpTransform


def to_multi_level_binary_tree(index, cfg):
    """
    Create a multi-level representation of the graph at a given index.

    Parameters
    ----------
        index: int
            graph index in the dataset
        cfg: dataset config
    """
    # Load graph
    raw_graph = dgl.load_graphs(os.path.join(cfg.data_path, f"graph/{str(index).zfill(6)}.bin"))[0][0]

    # Create new graph data with neighborhood edges
    graph_data = {
        ("obs", "neighbors", "obs"): raw_graph.edges(),
    }
    # Add empty downsampling and upsampling edges 
    for lvl in range(1, cfg.num_levels):
        graph_data[("obs", f"down_{lvl-1}_{lvl}", "obs")] = ([], [])
        graph_data[("obs", f"up_{lvl}_{lvl-1}", "obs")] = ([], [])
        
    # Create a dgl heterograph
    graph = dgl.heterograph(graph_data, idtype=torch.int32)

    positions = raw_graph.ndata['position']
    groups = raw_graph.ndata['group']

    tree_op = TreeOpTransform(2)

    lvls = {
        0: torch.ones_like(groups),
    }
    for lvl in range(1, cfg.num_levels):
        lvls[lvl] =  torch.zeros_like(groups)
    
    group_offset = 0
    for group in range(len(torch.unique(groups))):
        tree_edge_indices = tree_op(positions[groups == group])

        for lvl in range(1, cfg.num_levels):
            nodes_prev_lvl = (tree_edge_indices[lvl][1] + group_offset).to(torch.int32)
            nodes_curr_lvl = (tree_edge_indices[lvl][0] + group_offset).to(torch.int32)

            lvls[lvl][torch.unique(nodes_curr_lvl).tolist()] = 1

            graph.add_edges(nodes_prev_lvl, nodes_curr_lvl, etype=("obs", f"down_{lvl-1}_{lvl}", "obs"))
            graph.add_edges(nodes_curr_lvl, nodes_prev_lvl, etype=("obs", f"up_{lvl}_{lvl-1}", "obs"))

        group_offset += torch.sum(groups == group)

    for key, value in raw_graph.ndata.items():
        graph.ndata[key] = value
    for lvl in lvls:
        graph.ndata[f"lvl_{lvl}"] = lvls[lvl]

    # print({etype: graph.num_edges(etype=etype) for etype in graph.etypes})

    dgl.save_graphs(os.path.join(cfg.data_path, 'graph_binary_tree', f"{str(index).zfill(6)}.bin"), graph)
    
    
@timeit
@hydra.main(version_base=None, config_path="../../configs/data_generation", config_name="data_gen")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dir_name = cfg.save_name
    data_path = os.path.join(cfg.save_path, dir_name)
    os.makedirs(os.path.join(data_path, 'graph_binary_tree'), exist_ok=True)

    with open_dict(cfg):
        cfg['dir_name'] = dir_name
        cfg['data_path'] = data_path

    with open(os.path.join(data_path, 'samples.json')) as f:
        samples = json.load(f)

    # num_workers = max(1, cpu_count() - 4)
    # worker_func = partial(to_multi_level_binary_tree, cfg=cfg)
    # with Pool(processes=num_workers) as pool:
    #     pool.starmap(worker_func, zip(range(len(samples)), samples))

    for index, sample in enumerate(samples):
        to_multi_level_binary_tree(index, sample, cfg)

if __name__ == "__main__":
    main()