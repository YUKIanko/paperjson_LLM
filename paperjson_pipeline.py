import os
import json
import re
import fitz  # PyMuPDF
from openai import OpenAI

# LM Studio への接続設定
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ディレクトリパスの設定
input_directory = "pdfs"                  # 入力PDFが格納されたディレクトリ
intermediate_directory = "pypdf_baffuer"  # 抽出テキストおよびセグメント出力の一時保存先
output_directory = "output2"              # 最終的なJSON出力先
merged_segments_filename = "merged_segments.txt"  # 各セグメントの連結ファイル名（中間保存）

# ディレクトリが存在しなければ作成
for d in [intermediate_directory, output_directory]:
    if not os.path.exists(d):
        os.makedirs(d)

def extract_pdf_text(pdf_path):
    """PyMuPDFでPDFから全テキストを抽出する"""
    text_parts = []
    # fitz.open は自動でクローズされないため、withを使う
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # レイアウトに近いテキストを取得（必要なら"text"/"blocks"切替）
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)

def split_into_sections(text):
    """
    テキストを abstract / middle / references の3セクションに分割する。
    目次や見出しのヒューリスティクスに基づく簡易検出を行う。
    戻り値: [abstract_text, middle_text, references_text]
    """
    lines = [ln for ln in text.splitlines()]

    def find_heading(patterns, start=0, end=None, reverse=False):
        rng = range(start, len(lines) if end is None else end)
        if reverse:
            rng = range((len(lines) if end is None else end) - 1, start - 1, -1)
        for i in rng:
            line = lines[i].strip()
            for pat in patterns:
                if re.match(pat, line, flags=re.IGNORECASE):
                    return i
        return None

    # 見出し候補
    abstract_pats = [r"^abstract\b.*$", r"^summary\b.*$"]
    intro_like_pats = [
        r"^introduction\b.*$",
        r"^background\b.*$",
        r"^materials\s+and\s+methods\b.*$",
        r"^methods\b.*$",
        r"^experimental\s+procedures\b.*$",
        r"^results\b.*$",
        r"^main\s+text\b.*$",
        r"^keywords\b.*$"
    ]
    refs_pats = [
        r"^references?\b.*$",
        r"^bibliograph(y|ies)\b.*$",
        r"^literature\s+cited\b.*$",
        r"^works\s+cited\b.*$"
    ]

    # 参照開始位置（文末側から検索）
    refs_start = find_heading(refs_pats, start=0, end=None, reverse=True)

    # abstract開始位置（文頭側） — 先頭20%以内を優先探索
    scan_upto = max(100, int(len(lines) * 0.2))
    abstract_start = find_heading(abstract_pats, start=0, end=min(scan_upto, len(lines)))

    # abstract終了位置: 次の大見出し もしくは refs の直前
    abstract_end = None
    if abstract_start is not None:
        nxt = find_heading(intro_like_pats, start=abstract_start + 1, end=(refs_start if refs_start else None))
        if nxt is not None:
            abstract_end = nxt
        else:
            # 次見出し見つからない場合は、概ね150行を上限
            abstract_end = min(len(lines), (abstract_start + 150 if refs_start is None else min(abstract_start + 150, refs_start)))

    # 抽出
    if abstract_start is not None and abstract_end is not None and abstract_end > abstract_start:
        # タイトル・著者を取りこぼさないよう、Abstract見出しの少し前から含める
        start_idx = max(0, abstract_start - 30)
        abstract_text = "\n".join(lines[start_idx:abstract_end]).strip()
        mid_start_idx = abstract_end
    else:
        # fallback: 先頭から概ね200行をabstract相当とする
        fallback_end = min(len(lines), 200 if refs_start is None else min(200, refs_start))
        abstract_text = "\n".join(lines[:fallback_end]).strip()
        mid_start_idx = fallback_end

    if refs_start is not None and 0 <= refs_start < len(lines):
        references_text = "\n".join(lines[refs_start:]).strip()
        mid_end_idx = refs_start
    else:
        references_text = ""
        mid_end_idx = len(lines)

    middle_text = "\n".join(lines[mid_start_idx:mid_end_idx]).strip()

    return [abstract_text, middle_text, references_text]

#プロンプト生成関数を用途別に分割(セグメントの内容に応じてプロンプトを選択)
# イントロ（最初のセグメント）のプロンプト
def build_intro_prompt(segment_text):
    return (
        "You are a research assistant. Extract the following fields from the text:\n"
        "- Title\n"
        "- Authors\n"
        "- Genes (normalize to HGNC, hyphenate letters/numbers)\n"
        "- Proper Nouns (institutions, experimental_methods, software_tools, reagents_chemicals, model_organisms)\n\n"
        "Respond with valid JSON following this exact schema:\n"
        "{\n"
        '  "title": "",\n'
        '  "authors": [],\n'
        '  "references": [],\n'
        '  "genes": [],\n'
        '  "proper_nouns": {\n'
        '    "institutions": [],\n'
        '    "experimental_methods": [],\n'
        '    "software_tools": [],\n'
        '    "reagents_chemicals": [],\n'
        '    "model_organisms": []\n'
        "  }\n"
        "}\n"
        "If a field is not applicable, return it empty (\"\" or []). No extra explanations.\n\n"
        f"Text:\n{segment_text}"
    )

# 中間部分のプロンプト
def build_middle_prompt(segment_text):
    return (
        "You are a research assistant. Extract ONLY the following fields from the text:\n"
        "- Genes (normalize to HGNC, hyphenate letters/numbers)\n"
        "- Proper Nouns (institutions, experimental_methods, software_tools, reagents_chemicals, model_organisms)\n\n"
        "Respond with valid JSON following this exact schema:\n"
        "{\n"
        '  "title": "",\n'
        '  "authors": [],\n'
        '  "references": [],\n'
        '  "genes": [],\n'
        '  "proper_nouns": {\n'
        '    "institutions": [],\n'
        '    "experimental_methods": [],\n'
        '    "software_tools": [],\n'
        '    "reagents_chemicals": [],\n'
        '    "model_organisms": []\n'
        "  }\n"
        "}\n"
        "Fields not applicable must be empty (\"\" or []). No extra explanations.\n\n"
        f"Text:\n{segment_text}"
    )

# 末尾の文献部分のプロンプト
def build_reference_prompt(segment_text):
    return (
        "You are a research assistant. Extract ONLY the structured REFERENCES from the text below:\n"
        "- authors, title, journal, year, volume, pages, doi\n\n"
        "Respond with valid JSON following this exact schema:\n"
        "{\n"
        '  "title": "",\n'
        '  "authors": [],\n'
        '  "references": [],\n'
        '  "genes": [],\n'
        '  "proper_nouns": {\n'
        '    "institutions": [],\n'
        '    "experimental_methods": [],\n'
        '    "software_tools": [],\n'
        '    "reagents_chemicals": [],\n'
        '    "model_organisms": []\n'
        "  }\n"
        "}\n"
        "All unused fields must be empty (\"\" or []). No explanations.\n\n"
        f"Text:\n{segment_text}"
    )

# LMStudioにセグメントを送り、役割に応じてプロンプトを切り替える
def send_segment_to_llm(segment_text, pdf_base_name, seg_index, total_segments, section_type=None):
    """
    指定セグメントをLLMに送信しJSON抽出を行う。
    section_type: 'abstract' | 'middle' | 'references'（省略時はseg_indexで推定）
    返り値: LLM応答のプレーンJSON文字列（コードブロック除去後）
    """
    # セクション種別の決定
    if section_type is None:
        if seg_index == 0:
            section_type = 'abstract'
        elif seg_index == total_segments - 1:
            section_type = 'references'
        else:
            section_type = 'middle'

    # プロンプト選択
    if section_type == 'abstract':
        prompt = build_intro_prompt(segment_text)
    elif section_type == 'references':
        # テキストが極端に短い場合はmiddle扱いにフォールバック
        prompt = build_reference_prompt(segment_text) if len(segment_text.strip()) > 50 else build_middle_prompt(segment_text)
    else:
        prompt = build_middle_prompt(segment_text)

    # LLMへの送信処理
    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    output_text = response.choices[0].message.content.strip()

    # Markdownコードブロックを除去
    if output_text.startswith("```json"):
        output_text = output_text.replace("```json", "").replace("```", "").strip()

    # セグメントの出力をファイルに保存
    seg_filename = f"{pdf_base_name}_segment_{seg_index}.txt"
    seg_filepath = os.path.join(intermediate_directory, seg_filename)
    with open(seg_filepath, "w", encoding="utf-8") as seg_file:
        seg_file.write(output_text)

    return output_text

def send_merge_prompt(merged_segments_text):
    """
    各セグメントからのJSON出力が連続したテキスト(merged_segments_text)を、
    統合プロンプトとともにLLMに送信して最終的な統一JSONを取得する。
    """
    merge_prompt = (
        "You are a research assistant. Please merge the following JSON responses into one unified JSON object and remove duplicates. "
        "Use the first non-empty title found as the 'title'. Use the first non-empty authors list as 'authors'. "
        "For 'references', concatenate all arrays and deduplicate entries. "
        "The JSON object must adhere strictly to the following schema and contain no extra text, explanations, or formatting (e.g., no markdown code blocks like ```json):\n\n"
        "{\n"
        '  "title": "",\n'
        '  "authors": [],\n'
        '  "references": [],\n'
        '  "genes": [],\n'
        '  "proper_nouns": {\n'
        '    "institutions": [],\n'
        '    "experimental_methods": [],\n'
        '    "software_tools": [],\n'
        '    "reagents_chemicals": [],\n'
        '    "model_organisms": []\n'
        '  }\n'
        "}\n\n"
        "Important:\n"
        "- If any field is empty, use an empty string ('') or empty array ([]).\n"
        "- **Do NOT use markdown code blocks** (e.g., ` ```json `) in your response. Respond with plain JSON only.\n"
        "- Respond with **only the JSON object**.\n\n"
        "Here are the JSON responses to merge:\n\n" + merged_segments_text
    )
    
    response = client.chat.completions.create(
        model="lmstudio-community/dark-science-12b-v0.420-i1",
        messages=[{"role": "user", "content": merge_prompt}]
    )
    final_output_text = response.choices[0].message.content.strip()
    
    # デバッグ用: 応答を保存
    debug_output_path = os.path.join(intermediate_directory, "debug_final_output.txt")
    with open(debug_output_path, "w", encoding="utf-8") as debug_file:
        debug_file.write(final_output_text)

    # Markdownコードブロック記法の除去
    final_output_text = final_output_text.replace("```json", "").replace("```", "").strip()

    # JSONパース処理
    try:
        final_json = json.loads(final_output_text)
    except Exception as e:
        print("最終統合JSONパースエラー:", e)
        print("統合LLM応答内容:", final_output_text)
        final_json = {}

    return final_json

# PDFディレクトリ内の全PDFファイルをループ処理
for filename in os.listdir(input_directory):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_directory, filename)
        pdf_base_name = os.path.splitext(filename)[0]

        # ① PDFからテキスト抽出して中間テキストファイルに保存
        extracted_text = extract_pdf_text(pdf_path)
        intermediate_text_path = os.path.join(intermediate_directory, pdf_base_name + ".txt")
        with open(intermediate_text_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)

        # ② abstract / middle / references の3分割
        with open(intermediate_text_path, "r", encoding="utf-8") as txt_file:
            full_text = txt_file.read()
        segments = split_into_sections(full_text)
        total_segments = len(segments)  # 期待値は3

        # ③ 各セグメントごとにLLMに送信し、出力されたJSONテキストを連結
        merged_segments = ""
        section_names = ['abstract', 'middle', 'references']
        for idx, segment in enumerate(segments):
            section_type = section_names[idx] if idx < len(section_names) else None
            segment_json_text = send_segment_to_llm(
                segment, pdf_base_name, idx, total_segments, section_type=section_type
            )
            merged_segments += segment_json_text + "\n"

        # 連結した中間JSONテキストを保存（デバッグ用）
        merged_segments_file = os.path.join(intermediate_directory, pdf_base_name + "_merged_segments.txt")
        with open(merged_segments_file, "w", encoding="utf-8") as seg_file:
            seg_file.write(merged_segments)

        # ④ 統合プロンプトを送信して、最終的な統一JSONを取得
        final_json = send_merge_prompt(merged_segments)

        # ⑤ 統合した最終JSONを出力ディレクトリに保存（ファイル名は元のPDF名と同じ）
        output_json_path = os.path.join(output_directory, pdf_base_name + ".json")
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(final_json, json_file, ensure_ascii=False, indent=2)

        print(f"{filename} の最終論文JSONが生成され、出力は {output_json_path} に保存されました。")

