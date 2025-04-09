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

def split_text(text, max_chars=3000):
    """
    テキストを固定文字数(max_chars)ごとに分割する。
    ※実際はセクション単位に分割するなどの工夫が望ましい。
    """
    segments = []
    text_length = len(text)
    num_segments = math.ceil(text_length / max_chars)
    for i in range(num_segments):
        start = i * max_chars
        end = start + max_chars
        segments.append(text[start:end])
    return segments

def send_segment_to_llm(segment_text, pdf_base_name, seg_index):
    """
    指定されたセグメントのテキストに対し、LLM APIへプロンプトを送信しJSON出力を取得する。
    """
    prompt = (
        "You are a research assistant. Extract only the following fields from the text:\n\n"
        "1. Title: Full title of the paper.\n"
        "2. Authors: A list of all authors.\n"
        "3. References: The formal reference list at the end of the paper only.\n"
        "   - Do NOT include in-text citation markers such as [12], (12), etc.\n"
        "   - Only include structured reference entries (e.g., DOI, authors, journal, volume, pages).\n"
        "4. Genes: Gene names (normalize to official HGNC if possible, insert hyphens between letters and numbers, remove duplicates).\n"
        "5. Proper Nouns, with subcategories:\n"
        "   - Institutions (universities, labs, etc.)\n"
        "   - Experimental Methods (e.g. PCR, sequencing)\n"
        "   - Software & Tools\n"
        "   - Reagents & Chemicals\n"
        "   - Model Organisms & Cell Lines\n\n"
        "Your output must be a single valid JSON object with no extra commentary:\n\n"
        "```json\n"
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
        "```\n\n"
        "If any field is not present, return an empty string or empty array.\n"
        "No explanations or extra text. Only return the JSON.\n\n"
        f"The text to process is:\n\n{segment_text}"
    )
    ...

    
    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    output_text = response.choices[0].message.content.strip()
    
    # コードブロック除去
    if output_text.startswith("```json"):
        output_text = output_text[7:-3].strip()

    return output_text
    
    # 個別セグメントの出力をファイルに保存
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
        
        # ② テキストファイルを読み込み、適当な長さに分割
        with open(intermediate_text_path, "r", encoding="utf-8") as txt_file:
            full_text = txt_file.read()
        segments = split_text(full_text, max_chars=3000)
        
        # ③ 各セグメントごとにLLMに送信し、出力されたJSONテキストを連結
        merged_segments = ""
        for idx, segment in enumerate(segments):
            segment_json_text = send_segment_to_llm(segment, pdf_base_name, idx)
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
