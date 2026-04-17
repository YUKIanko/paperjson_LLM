import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# --- 設定 ---
json_dir = "output2"
# 複数のフィールドを指定
target_fields = [
    "genes",
    "experimental_methods",
    "model_organisms",
    "software_tools",
    "reagents_chemicals"]

normalize_tags = True
max_title_len = 30

# --- 正規化関数 ---
def normalize_tag(tag):
    tag = tag.lower()
    tag = re.sub(r"[^a-z0-9]", "", tag)
    return tag

# --- 準備 ---
G = nx.Graph()
paper_tags = defaultdict(list)  # 論文 → タグリスト
tag_to_papers = defaultdict(set)

# --- ノードと論文↔タグのエッジ構築 ---
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_title = data.get("title", filename.replace(".json", ""))
        short_title = full_title[:max_title_len] + "..." if len(full_title) > max_title_len else full_title

        G.add_node(short_title, type="paper", full_title=full_title)
        # 複数のフィールドをループ処理
        for field in target_fields:
            if field == "genes":
                raw_tags = data.get("genes", [])
            else:
                raw_tags = data.get("proper_nouns", {}).get(field, [])
            #正規化
            if normalize_tags:
                raw_tags = [normalize_tag(t) for t in raw_tags]
            #タグごとにノードを追加
            for tag in raw_tags:
                tag_node = f"{field.upper()}::{tag}"
                G.add_node(tag_node, type=field)
                G.add_edge(short_title, tag_node, edge_type="paper-tag")
                tag_to_papers[tag].add(short_title)
                paper_tags[short_title].append(tag)

#エッジの重みを格納する辞書
edge_weights = defaultdict(int)
for tag, papers_set in tag_to_papers.items():
    papers_list = list(papers_set)
    for i in range(len(papers_list)):
        for j in range(i + 1, len(papers_list)):
            p1, p2 = sorted((papers_list[i], papers_list[j]))
            edge_weights[p1, p2] += 1

#色変え
for (p1, p2), weight in edge_weights.items():
    if weight >= 1:
        if weight >= 5:
            width = 3.0
            color = "red"
        elif weight >= 3:
            width = 2.0
            color = "orange"
        else:
            width = 1.0
            color = "blue"
        # 論文間エッジ追加（重みに応じて太さを変える）
        G.add_edge(p1, p2, edge_type="paper-paper", weight=weight, width=width, color=color)

paper_tag_edges = [
    (u, v) for u, v, d in G.edges(data=True)
    if d.get("edge_type") == "paper-tag"
]

# weightが2以上の論文間エッジだけを取り出す
paper_paper_edges = [
    (u, v) for u, v, d in G.edges(data=True)
    if d.get("edge_type") == "paper-paper" and d.get("weight", 0) >= 2
]

paper_paper_colors = [G.edges[e]["color"] for e in paper_paper_edges]
paper_paper_widths = [G.edges[e]["width"] for e in paper_paper_edges]

# --- 描画 ---
plt.figure(figsize=(16, 12))
#pos = nx.spring_layout(G, seed=42, k=0.45)
pos = nx.kamada_kawai_layout(G)

# エッジごとに色と太さを分けて描画
paper_tag_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "paper-tag"]
paper_paper_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "paper-paper"]
paper_paper_colors = [G.edges[e]["color"] for e in paper_paper_edges]
paper_paper_widths = [G.edges[e]["width"] for e in paper_paper_edges]

# ノード属性で色分け
color_scheme = {
    "paper": "skyblue",
    "genes": "green",
    "experimental_methods": "orange",
    "model_organisms": "red",
    "software_tools": "purple",
    "reagents_chemicals": "brown"
}

def get_node_color(n):
    node_type = G.nodes[n]["type"]
    return color_scheme.get(node_type, "gray")  # 不明なら灰色

node_colors = [get_node_color(n) for n in G.nodes()]
node_sizes = [
    1200 if G.nodes[n]["type"] == "paper" else 400
    for n in G.nodes()
]

# (1) 論文↔タグのエッジ（灰色で細く）
nx.draw_networkx_edges(
    G, pos,
    edgelist=paper_tag_edges,
    edge_color="gray",
    width=0.5
)

# (2) 論文↔論文のエッジ（重みに応じて色と太さを変化）
nx.draw_networkx_edges(
    G, pos,
    edgelist=paper_paper_edges,
    edge_color=paper_paper_colors,
    width=paper_paper_widths
)

# (3) ノード描画
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=node_sizes
)
# まず全ノードの次数を取得し、条件を満たすノードだけ dict に格納
deg = G.degree()
label_dict = {}
for n, d in deg:
    if d >= 5:  # 次数5以上のノードのみラベル付け
        label_dict[n] = n

# (4) ラベル描画
nx.draw_networkx_labels(
    G, pos,
    font_size=7
)

plt.title("Network: (論文 ↔ 複数タグ & 論文 ↔ 論文)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()