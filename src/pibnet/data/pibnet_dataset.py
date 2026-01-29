import torch
from torch.utils.data import Dataset
import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import json
import roma
from copy import deepcopy
from time import time

from data.data_utils import *

def random_rotmat():
     """
     Returns a random rotation matrix whose rotation angles are multiples of pi/2
     """
     return roma.euler_to_rotmat('xyz', [
          np.random.choice([-np.pi/2., 0., np.pi/2, np.pi]), 
          np.random.choice([-np.pi/2., 0., np.pi/2, np.pi]), 
          np.random.choice([-np.pi/2., 0., np.pi/2, np.pi])
     ]).to(torch.float32)

class BaseDGLDataset(Dataset):
     """
     Dataset base class

     Parameters
     ----------
     data_root: str
          Path of the dataset directory
     cfg: config
     """
     def __init__(self, data_root, cfg):
          self.cfg = cfg
          self.data_root = data_root

          with open(os.path.join(data_root, 'samples.json')) as f:
               self.samples = json.load(f)

class PIBNetDataset(BaseDGLDataset):
     """
     Dataset class for training

     Parameters
     ----------
     data_root: str
          Path of the dataset directory
     cfg: training config
     """
     def __init__(self, data_root, cfg):
          super().__init__(data_root, cfg)
          
          self.graphs = []
          for i, sample in enumerate(self.samples):

               raw_graph = dgl.load_graphs(os.path.join(data_root, f"graph_binary_tree/{str(i).zfill(6)}.bin"))[0][0]

               if len(raw_graph.etypes) == 1:
                    # raw_graph is a homogeneous graph
                    graph_data = {("obs", 'neighbors', "obs"): raw_graph.edges()}
               else:
                    # raw_graph is a heterogeneous graph
                    graph_data = {("obs", etype, "obs"): raw_graph.edges(etype=etype) for etype in list(set(cfg.data.etypes) & set(raw_graph.etypes))}
               
               for etype in cfg.data.etypes:
                    if ("obs", etype, "obs") not in graph_data:
                         graph_data[("obs", etype, "obs")] = ([], [])

               graph = dgl.heterograph(graph_data, idtype=torch.int32)

               num_nodes = raw_graph.num_nodes()

               positions = raw_graph.ndata['position']
               group = raw_graph.ndata['group']
               env_dim = positions.shape[-1]

               labels = raw_graph.ndata['labels']

               if cfg.data.equation == 'laplace':
                    if labels.dim() == 1:
                         labels = labels.reshape(-1, 1)
                    elif labels.dim() == 2:
                         labels = labels[..., :1]
                    else:
                         raise ValueError

               graph.ndata['position'] = positions
               graph.ndata['labels'] = labels
               graph.ndata['group'] = group

               if cfg.data.source.type == 'monopole':
                    source_rel_pos = positions - torch.tensor(sample['source']['position']).view(1, -1)[:, :env_dim]
                    source_dist = torch.linalg.norm(source_rel_pos, axis=-1, keepdim=True)
                    source_dir = source_rel_pos / source_dist
                    
                    graph.ndata['source_dir'] = source_dir
                    graph.ndata['source_dist'] = source_dist

                    graph.ndata['labels_normalized'] = labels * source_dist

               elif cfg.data.source.type == 'plane':
                    source_dir = torch.tensor(sample['source']['direction']).view(1, -1)[:, :env_dim]
                    graph.ndata['source_dir'] = source_dir * torch.ones((num_nodes, 1))

                    center = positions.mean(dim=0, keepdim=True)
                    center_rel_pos = positions - center
                    center_dist = torch.linalg.norm(center_rel_pos, axis=-1, keepdim=True)
                    center_dir = center_rel_pos / center_dist
                    
                    graph.ndata['center_dist'] = center_dist
                    graph.ndata['center_dir'] = center_dir

               else:
                    raise NotImplementedError

               if cfg.data.equation == 'helmholtz':
                    wavelength = sample['source']['wavelength']
                    graph.ndata['wavelength'] = wavelength * torch.ones((num_nodes, 1))
               
                    if cfg.data.source.type == 'monopole':
                         graph.ndata['source_phase'] = torch.cat([
                              torch.sin(2 * torch.pi / wavelength * source_dist), 
                              torch.cos(2 * torch.pi / wavelength * source_dist)
                         ], axis=-1)

                    elif cfg.data.source.type == 'plane':
                         graph.ndata['source_phase'] = torch.cat([
                              torch.sin(2 * torch.pi / wavelength * (source_dir * positions).sum(axis=-1, keepdim=True)), 
                              torch.cos(2 * torch.pi / wavelength * (source_dir * positions).sum(axis=-1, keepdim=True))
                         ], axis=-1)

               elif cfg.data.equation == 'laplace':
                    direction = torch.tensor(sample['source']['direction']).view(1, -1)[:, :env_dim]
                    graph.ndata['direction'] = direction * torch.ones((num_nodes, 1))
                    amplitudes = torch.tensor(sample['source']['amplitudes']).view(1, -1)
                    graph.ndata['amplitudes'] = amplitudes * torch.ones((num_nodes, 1))

                    graph.ndata['boundary_cond'] = torch.cat([
                              sample['source']['amplitudes'][1] / source_dist, 
                              2 * sample['source']['amplitudes'][2] * (direction * source_dir).sum(axis=-1, keepdim=True)
                         ], axis=-1)

               else:
                    raise NotImplementedError(f"equation must be one of 'laplace' or 'helmholtz', not {cfg.data.equation}.")

               for level in range(cfg.data.num_levels):
                    if f"lvl_{level}" in raw_graph.ndata:
                         graph.ndata[f"lvl_{level}"] = raw_graph.ndata[f"lvl_{level}"]

               for etype in graph.etypes:

                    if graph.num_edges(etype=etype) > 0:

                         num_edges = graph.num_edges(etype=etype)
                         edges_src, edges_dst = graph.edges(etype=etype)
                         rel_dir, dist = get_edge_dir_and_dist(edges_src, edges_dst, positions)
                         graph.edges[etype].data['rel_dir'] = rel_dir
                         graph.edges[etype].data['dist'] = dist

                         if cfg.data.equation == 'helmholtz':
                              graph.edges[etype].data['wavelength'] = wavelength * torch.ones((num_edges, 1))
                              graph.edges[etype].data['phase'] = torch.cat([
                                   torch.sin(2 * torch.pi / wavelength * graph.edges[etype].data['dist']), 
                                   torch.cos(2 * torch.pi / wavelength * graph.edges[etype].data['dist'])
                              ], axis=-1)

                         elif cfg.data.equation == 'laplace':
                              graph.edges[etype].data['amplitudes'] = amplitudes * torch.ones((num_edges, 1))

                              graph.ndata['boundary_cond'] = torch.cat([
                                   sample['source']['amplitudes'][0] * torch.ones((num_nodes, 1)),
                                   sample['source']['amplitudes'][1] / source_dist, 
                                   2 * sample['source']['amplitudes'][2] * (direction * source_dir).sum(axis=-1, keepdim=True)
                              ], axis=-1)

               self.graphs.append(graph)

     def __len__(self):
          return len(self.graphs)

     def __getitem__(self, index):

          graph = deepcopy(self.graphs[index])
          sample = self.samples[index]

          # Apply data augmentation
          if self.cfg.apply_data_augmentation:
               rand_roma = random_rotmat().T
               graph.ndata['position'] = torch.matmul(graph.ndata['position'], rand_roma)
               graph.ndata['source_dir'] = torch.matmul(graph.ndata['source_dir'], rand_roma)
               for etype in graph.etypes:
                    if graph.num_edges(etype=etype) > 0:
                         graph.edges[etype].data['rel_dir'] = torch.matmul(graph.edges[etype].data['rel_dir'], rand_roma)
               
               if self.cfg.data.source.type == 'plane':
                    graph.ndata['center_dir'] = torch.matmul(graph.ndata['center_dir'], rand_roma)

          # Add distant edges 
          for etype in self.cfg.data.new_edges_per_node_ratio:
               if self.cfg.data.new_edges_per_node_ratio[etype] > 0.:
                    if self.cfg.data.num_levels > 1:
                         node_mask = graph.ndata[f"lvl_{self.cfg.data.num_levels-1}"] == 1
                    else:
                         node_mask = None
                    new_src, new_dst = get_edges_to_add(graph, self.cfg.data.new_edges_per_node_ratio[etype], self.cfg.data.candidate_edge_ratio, node_mask=node_mask)

                    graph = add_new_edges(graph, new_src, new_dst, etype=etype, sample=sample, cfg=self.cfg)
               
          return graph


class PIBNetDatasetTest(PIBNetDataset):
     """
     Dataset class for testing

     Parameters
     ----------
     data_root: str
          Path of the dataset directory
     cfg: testing config
     """
     def __init__(self, data_root, cfg):
          super().__init__(data_root, cfg)

     def __len__(self):
          return len(self.graphs)

     def __getitem__(self, index):

          graph = deepcopy(self.graphs[index])
          sample = self.samples[index]

          # Add distant edges 
          for etype in self.cfg.data.new_edges_per_node_ratio:
               if self.cfg.data.new_edges_per_node_ratio[etype] > 0.:
                    if self.cfg.data.num_levels > 1:
                         node_mask = graph.ndata[f"lvl_{self.cfg.data.num_levels-1}"] == 1
                    else:
                         node_mask = None
                    new_src, new_dst = get_edges_to_add(graph, self.cfg.data.new_edges_per_node_ratio[etype], self.cfg.data.candidate_edge_ratio, node_mask=node_mask)

                    graph = add_new_edges(graph, new_src, new_dst, etype=etype, sample=sample, cfg=self.cfg)
                    
          return graph