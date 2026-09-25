"""核对一个查询点的 MOL2 与 Rosetta params 是否保留相同的分子连接图.

本脚本只用于输入门禁。Rosetta 会重新命名重复原子名并加入虚拟环原子, 因此比较
元素与键级标记的图同构, 不比较原子名称或虚拟原子.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import networkx as nx


def read_mol2_graph(path: Path) -> nx.Graph:
    """从 MOL2 读取带元素与键级的化学图.

    - path: 一个候选配体的 MOL2 文件.
    - return: 数字原子编号为节点, 元素为节点属性; ar、am 与 Rosetta 的芳香键级 4 一致.
    """
    graph = nx.Graph()
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.removeprefix("@<TRIPOS>")
            continue
        fields = line.split()
        if section == "ATOM" and len(fields) >= 6:
            graph.add_node(int(fields[0]), element=fields[5].split(".")[0].upper(), name=fields[1])
        elif section == "BOND" and len(fields) >= 4:
            order = "4" if fields[3] in {"ar", "am"} else fields[3]
            graph.add_edge(int(fields[1]), int(fields[2]), order=order)
    return graph


def read_params_graph(path: Path) -> nx.Graph:
    """从 Rosetta params 读取原始配体原子与键, 排除新增的虚拟环原子.

    - path: mol2genparams.py 生成的一个候选配体 params 文件.
    - return: Rosetta 原子名为节点, 元素与键级属性与 MOL2 图一致.
    """
    graph = nx.Graph()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "ATOM" and fields[2] != "VIRT":
            name = fields[1]
            element = name[:2].upper() if name[:2].upper() in {"CL", "BR"} else name[0].upper()
            graph.add_node(name, element=element)
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "BOND_TYPE" and fields[1] in graph and fields[2] in graph:
            graph.add_edge(fields[1], fields[2], order=fields[3])
    return graph


def main() -> None:
    """逐候选核对 MOL2 与 params 的元素、连接、键级和组分.

    - ligand_dir: 一个查询点专属的 ligands 目录, 每个 Lxx.mol2 对应一个 Lxx.params.
    - return: 全部图同构时打印原子、键、组分数量; 首个不一致处终止门禁.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ligand_dir", type=Path)
    args = parser.parse_args()

    for mol2_path in sorted(args.ligand_dir.glob("*.mol2")):
        mol2_graph = read_mol2_graph(mol2_path)
        params_graph = read_params_graph(mol2_path.with_suffix(".params"))
        same_graph = nx.is_isomorphic(
            mol2_graph,
            params_graph,
            node_match=lambda first, second: first["element"] == second["element"],
            edge_match=lambda first, second: first["order"] == second["order"],
        )
        if not same_graph:
            same_connections = nx.is_isomorphic(
                mol2_graph,
                params_graph,
                node_match=lambda first, second: first["element"] == second["element"],
            )
            mol2_orders = Counter(data["order"] for _, _, data in mol2_graph.edges(data=True))
            params_orders = Counter(data["order"] for _, _, data in params_graph.edges(data=True))
            mapping = next(
                nx.isomorphism.GraphMatcher(
                    mol2_graph,
                    params_graph,
                    node_match=lambda first, second: first["element"] == second["element"],
                ).isomorphisms_iter()
            )
            changed_bonds = [
                (
                    mol2_graph.nodes[first]["name"],
                    mol2_graph.nodes[second]["name"],
                    data["order"],
                    mapping[first],
                    mapping[second],
                    params_graph.edges[mapping[first], mapping[second]]["order"],
                )
                for first, second, data in mol2_graph.edges(data=True)
                if data["order"] != params_graph.edges[mapping[first], mapping[second]]["order"]
            ]
            raise ValueError(
                f"{mol2_path.stem}: 元素与连接同构={same_connections}, "
                f"MOL2 原子/键={mol2_graph.number_of_nodes()}/{mol2_graph.number_of_edges()}, "
                f"params 原子/键={params_graph.number_of_nodes()}/{params_graph.number_of_edges()}, "
                f"MOL2 键级={dict(mol2_orders)}, params 键级={dict(params_orders)}, "
                f"第一个元素/连接同构下不同的键={changed_bonds[:10]}"
            )
        print(
            f"{mol2_path.stem}: {mol2_graph.number_of_nodes()} atoms, "
            f"{mol2_graph.number_of_edges()} bonds, "
            f"{nx.number_connected_components(mol2_graph)} components"
        )


if __name__ == "__main__":
    main()
