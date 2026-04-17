import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# --- 設定 ---
json_dir = "output2"
target_field = "genes"
normalize_tags = True
max_title_len = 30

# --- 正規化関数 ---
def normalize_tag(tag):
    tag = tag.lower()
    tag = re.sub(r"[^a-z0-9]", "", tag)
    return tag

# --- 準備 ---
G = nx.Graph()
paper_tags = {}  # 論文 → タグリスト
tag_to_papers = defaultdict(set)

# --- ノードと論文↔タグのエッジ構築 ---
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_title = data.get("title", filename.replace(".json", ""))
        short_title = full_title[:max_title_len] + "..." if len(full_title) > max_title_len else full_title
        tags = data.get(target_field, [])
        if normalize_tags:
            tags = [normalize_tag(t) for t in tags]

        G.add_node(short_title, type="paper", full_title=full_title)
        paper_tags[short_title] = tags

        for tag in tags:
            tag_node = f"{target_field.upper()}::{tag}"
            G.add_node(tag_node, type="tag")
            G.add_edge(short_title, tag_node, edge_type="paper-tag")
            tag_to_papers[tag].add(short_title)

# --- 論文間の共通タグをもとにエッジ追加（重みあり） ---
edge_weights = defaultdict(int)
for tag, papers in tag_to_papers.items():
    papers = list(papers)
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            pair = tuple(sorted((papers[i], papers[j])))
            edge_weights[pair] += 1

# 論文間エッジ追加（重みに応じて太さを変える）
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
        G.add_edge(p1, p2, edge_type="paper-paper", weight=weight, width=width, color=color)

# --- 可視化 ---
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42, k=0.45)

# エッジごとに色と太さを分けて描画
paper_tag_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "paper-tag"]
paper_paper_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "paper-paper"]
paper_paper_colors = [G.edges[e]["color"] for e in paper_paper_edges]
paper_paper_widths = [G.edges[e]["width"] for e in paper_paper_edges]

# ノード属性で色分け
node_colors = ["skyblue" if G.nodes[n]["type"] == "paper" else "lightgreen" for n in G.nodes()]
node_sizes = [1200 if G.nodes[n]["type"] == "paper" else 400 for n in G.nodes()]

# 論文↔タグのエッジ（灰色）
nx.draw_networkx_edges(G, pos, edgelist=paper_tag_edges, edge_color="gray", width=0.5)

# 論文↔論文のエッジ（重みつき色）
nx.draw_networkx_edges(G, pos, edgelist=paper_paper_edges, edge_color=paper_paper_colors, width=paper_paper_widths)

# ノード・ラベル
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, font_size=7)

plt.title(f"Network: {target_field} (論文↔タグ & 論文↔論文)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
