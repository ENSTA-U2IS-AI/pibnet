import os
os.environ["DGLBACKEND"] = "pytorch"
from typing import Any, List, Dict, Union
import numpy as np
import meshio
import torch
import dgl


def merge_meshes(*meshes: List[meshio.Mesh]) -> meshio.Mesh:
    """
    Merge a list of meshio mesh.

    Parameters
    ----------
        mesh: List[meshio.Mesh]
            The list of meshio meshes to merge
    """
    points_list = []
    points_groups = []
    edges_list = []
    edges_groups = []
    triangles_list = []
    triangles_groups = []
    total_num_points = 0

    for mesh_indice, mesh in enumerate(*meshes):
        assert (mesh.point_data['group'] == 0).all(), 'cannot merge meshes that have already been merged'
        points = mesh.points
        points_list.append(points)
        num_points = points.shape[0]
        points_groups.append(np.ones(num_points) * mesh_indice)

        edges = mesh.cells_dict['line'] + total_num_points
        edges_list.append(edges)
        num_edges = edges.shape[0]
        edges_groups.append(np.ones(num_edges) * mesh_indice)

        if 'triangle' in mesh.cells_dict.keys():
            triangles = mesh.cells_dict['triangle'] + total_num_points
            triangles_list.append(triangles)
            num_triangles = triangles.shape[0]
            triangles_groups.append(np.ones(num_triangles) * mesh_indice)

        total_num_points += num_points

    points = np.vstack(points_list)
    points_groups = np.hstack(points_groups)

    edges = np.vstack(edges_list)
    edges_groups = np.hstack(edges_groups)
    cells={"line": edges}
    cell_data = {
        "group": [edges_groups],
    }

    if triangles_list != []:
        triangles = np.vstack(triangles_list)
        triangles_groups = np.hstack(triangles_groups)
        cells['triangle'] = triangles
        cell_data['group'].append(triangles_groups)

    # Create a MeshIO mesh
    return meshio.Mesh(
        points=points, 
        cells=cells,
        point_data={"group": points_groups},
        cell_data=cell_data
    )


def mesh_to_graph(mesh: meshio.Mesh, gt: Union[np.ndarray, None]) -> dgl.DGLGraph:
    """
    Convert a meshio mesh into a DGL graph object and include the ground truth solution.

    Parameters
    ----------
        mesh: meshio.Mesh
            The mesh to convert
        gt: Union[np.ndarray, None]
            The ground truth solution on the boundaries
    """
    edges = torch.from_numpy(mesh.cells_dict['line']).to(torch.int32)
    graph = dgl.graph((edges[:, 0], edges[:, 1]))

    graph.ndata['position'] = torch.from_numpy(mesh.points).to(torch.float32)
    if gt is not None:
        graph.ndata['labels'] = torch.from_numpy(gt).to(torch.float32)
    for attr, value in mesh.point_data.items():
        graph.ndata[attr] = torch.from_numpy(value).to(torch.float32)
    
    for attr, value in mesh.cell_data.items():
        # print(attr, value[0].shape)
        graph.edata[attr] = torch.from_numpy(value[0]).to(torch.float32)

    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
    graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True)

    return graph
        