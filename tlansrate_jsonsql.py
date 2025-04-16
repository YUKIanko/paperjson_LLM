import json
import sqlite3
import glob
import os

json_dir = "output2"
db_path = "papers.db"

def normalize_json(data):
    def to_list_str(x):
        if isinstance(x, list):
            return [str(i) if not isinstance(i, dict) else json.dumps(i, ensure_ascii=False) for i in x]
        elif isinstance(x, dict):
            return [json.dumps(x, ensure_ascii=False)]
        else:
            return [str(x)]

    normalized = {
        "title": str(data.get("title", "no title")),
        "authors": to_list_str(data.get("authors", [])),
        "references": to_list_str(data.get("references", [])),
        "genes": to_list_str(data.get("genes", [])),
        "proper_nouns": {}
    }

    for key in ["institutions", "experimental_methods", "software_tools", "reagents_chemicals", "model_organisms"]:
        raw_value = data.get("proper_nouns", {}).get(key, [])
        normalized["proper_nouns"][key] = to_list_str(raw_value)

    return normalized

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authors TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER,
    category TEXT,
    keyword TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER,
    cited_work TEXT
)
""")

json_paths = glob.glob(os.path.join(json_dir, "*.json"))
print(f"📂 読み込むJSON数: {len(json_paths)}")

for path in json_paths:
    print(f"→ 処理中: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
            data = normalize_json(raw_data)
        except Exception as e:
            print(f"⚠️ 読み込み失敗: {path} → {e}")
            continue

    authors_str = ", ".join(data["authors"])
    cur.execute("""
        INSERT INTO papers (title, authors)
        VALUES (?, ?)
    """, (data["title"], authors_str))
    paper_id = cur.lastrowid

    def insert_features(category, items):
        for kw in items:
            cur.execute("""
                INSERT INTO features (paper_id, category, keyword)
                VALUES (?, ?, ?)
            """, (paper_id, category, kw))

    insert_features("gene", data["genes"])
    for cat in ["experimental_methods", "reagents_chemicals", "software_tools", "institutions", "model_organisms"]:
        insert_features(cat, data["proper_nouns"].get(cat, []))

    for cited in data["references"]:
        cur.execute("""
            INSERT INTO citations (paper_id, cited_work)
            VALUES (?, ?)
        """, (paper_id, cited))

conn.commit()
conn.close()
print(f"✅ 変換完了：{db_path} に保存されました")
