import os
import json
import math
import PyPDF2
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
    """PDFから全テキストを抽出する"""
    file_text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                file_text += page_text + "\n"
    return file_text

# テキストを分割するための定数
MAX_CHARS = 6000 #1セグメントの最大文字数
MAX_SEGMENTS = 30 # セグメント数の上限

def split_text(text, max_chars=MAX_CHARS, tail_lines=150):
    lines = text.strip().split("\n")
    #行数が150以下のときは全部tail_textに入れる
    # それ以上のときは、tail_lines行を除いた部分をmain_textに入れ
    if len(lines) <= tail_lines:
        main_text = ""
        tail_text = "\n".join(lines)
    else:
        main_text = "\n".join(lines[:-tail_lines:])
        tail_text = "\n".join(lines[-tail_lines:])
    segments = []
    # main_textをmax_charsごとに分割
    for i in range(0, len(main_text), max_chars):
        segments.append(main_text[i:i + max_chars])

    # 末尾行も追加
    if tail_text:
        segments.append(tail_text)

    return segments

#プロンプト生成関数を用途別に分割(文献情報があるか判定する)
def detect_if_references(segment_text):
    detect_prompt = (
        "You are a research assistant. Examine the following text.\n\n"
        "If the text contains a structured reference list (Author. Year. Title. Journal. Volume: Pages.), "
        "return JSON: {'is_references': 1}, else {'is_references': 0}.\n\n"
        f"Text to check:\n\n{segment_text}"
    )

    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": detect_prompt}]
    )

    result_text = response.choices[0].message.content.strip()
    try:
        result_json = json.loads(result_text)
        return result_json.get("is_references", 0)
    except json.JSONDecodeError:
        return 0

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

#LMstudioとの接続を行い、セグメントごとにプロンプトを送信してJSON出力を取得する
def send_segment_to_llm(segment_text, pdf_base_name, seg_index, total_segments):
    #指定セグメントをLLMに送信しJSON抽出を行う。
    #Parameters:
    #- segment_text: 処理するテキストセグメント
    #- pdf_base_name: PDFファイル名（ベース名）
    #- seg_index: 現在のセグメント番号（0から開始）
    #- total_segments: セグメント総数（全セグメント数）
    #セグメント位置によりプロンプトを切り替えるためにseg_indexとtotal_segmentsを使用。
    # セグメントの位置でプロンプトを分岐
    if seg_index == 0:
        prompt = build_intro_prompt(segment_text)
    elif seg_index == total_segments - 1:
        if detect_if_references(segment_text):
            prompt = build_reference_prompt(segment_text)
        else:
            prompt = build_middle_prompt(segment_text)
    else:
        prompt = build_middle_prompt(segment_text)

    # LLMへの送信処理
    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    output_text = response.choices[0].message.content.strip()

    # markdownコードブロックを除去
    if output_text.startswith("```json"):
        output_text = output_text[7:-3].strip()

    # セグメントの出力をファイルに保存
    seg_filename = f"{pdf_base_name}_segment_{seg_index}.txt"
    seg_filepath = os.path.join(intermediate_directory, seg_filename)
    with open(seg_filepath, "w", encoding="utf-8") as seg_file:
        seg_file.write(output_text)

    return output_text
# 文献情報があるか判定するプロンプトを分離
# 参考文献の有無を判定するプロンプ
def detect_if_references(segment_text):
    detect_prompt = (
        "Determine if the following text contains structured references "
        "(authors, year, title, journal, volume, pages).\n\n"
        "If yes, return: {\"is_references\": 1}, else return: {\"is_references\": 0}.\n\n"
        f"Text:\n{segment_text}"
    )

    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": detect_prompt}]
    )

    result_text = response.choices[0].message.content.strip()
    try:
        result_json = json.loads(result_text)
        return result_json.get("is_references", 0)
    except json.JSONDecodeError:
        return 0
# セグメントの出力をファイルに保存
def send_merge_prompt(merged_segments_text):
    """
    各セグメントからのJSON出力が連続したテキスト(merged_segments_text)を、
    統合プロンプトとともにLLMに送信して最終的な統一JSONを取得する。
    """
    merge_prompt = (
        "You are a research assistant. Please merge the following JSON responses into one unified JSON object and remove duplicates. "
        "Use the first non-empty title found as the 'title'. "
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
        #セグメント警告
        full_text = extracted_text
        segments = split_text(full_text, max_chars=MAX_CHARS)
        if len(segments) > MAX_SEGMENTS:
            print(f"分割数が{MAX_SEGMENTS}を超えました。")
            print(f"分割数: {len(segments)}")
        
        # ② テキストファイルを読み込み、適当な長さに分割
        with open(intermediate_text_path, "r", encoding="utf-8") as txt_file:
            full_text = txt_file.read()
        segments = split_text(full_text, max_chars=3000)
        total_segments = len(segments)
        
        # ③ 各セグメントごとにLLMに送信し、出力されたJSONテキストを連結
        merged_segments = ""
        for idx, segment in enumerate(segments):
            segment_json_text = send_segment_to_llm(segment, pdf_base_name, idx, total_segments)
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
