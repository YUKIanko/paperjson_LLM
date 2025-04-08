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

def send_segment_to_llm(segment_text, pdf_base_name, seg_index, intermediate_directory):
    """
    指定されたセグメントのテキストに対し、LLM APIへプロンプトを送信しJSON出力を取得する。
    """
    prompt = (
        "You are a research assistant tasked with extracting and standardizing key metadata from a scientific paper.\n"
        "Please read the provided text carefully and extract only the following fields:\n\n"
        
        "- **Title**: The full title of the paper.\n"
        "- **Authors**: A list of all authors.\n"
        "- **References**: The list of references or citations mentioned in the paper.\n"
        "- **Genes**: Any gene names or specific biological identifiers mentioned in the paper.\n"
        "  - Normalize gene names according to the following rules:\n"
        "    - Convert all gene names to a standardized format.\n"
        "    - Use official HGNC names where applicable.\n"
        "    - For gene names that contain both letters and numbers, insert a hyphen between them (e.g., ABC1 → ABC-1).\n"
        "    - Remove unnecessary whitespace, underscores, or special characters unless they are part of an official name.\n"
        "    - Ensure genes are not duplicated within the same document.\n"
        "- **Proper Nouns**: Scientific entities categorized into the following subcategories:\n"
        "  - **Institutions**: Universities, research centers, laboratories, etc.\n"
        "  - **Experimental Methods**: Gene editing techniques, PCR variations, sequencing methods, biochemical assays.\n"
        "  - **Software & Tools**: Bioinformatics software, automation tools, analytical software.\n"
        "  - **Reagents & Chemicals**: Antibodies, dyes, chemical compounds, experimental reagents.\n"
        "  - **Model Organisms & Cell Lines**: Names of model organisms like Xenopus laevis, HEK293 cells, C57BL/6 mice, zebrafish, etc.\n\n"
        
        "Your output **must be a valid JSON object** with proper indentation and no additional commentary.\n"
        "Ensure that the JSON structure is correctly formatted.\n\n"
        
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
        
        "If any field is not present in the text, return an empty string (or an empty array for list fields).\n"
        "Do not include any explanations or extra text in your output. Only output the JSON object.\n\n"
        "The text to process is:\n\n" + segment_text
    )

    # LLM API を呼び出し
    client = OpenAI()  # OpenAI クライアントの初期化
    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    
    output_text = response.choices[0].message.content.strip()

    # コードブロックが含まれている場合、削除
    if output_text.startswith("```json"):
        output_text = output_text[7:-3].strip()

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
        "You are a research assistant. Please merge the following JSON responses into one JSON object and remove duplicate words from each list. "
        "Use the first non-empty title as the final title. The JSON object must have exactly the following structure with no additional commentary:\n\n"
        "{\n"
        '  "title": "",\n'
        '  "authors": [],\n'
        '  "references": [],\n'
        '  "genes": [],\n'
        '  "proper_nouns": []\n'
        "}\n\n"
        "Here are the JSON responses:\n\n" + merged_segments_text
    )
    
    response = client.chat.completions.create(
        model="lmstudio-community/qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": merge_prompt}]
    )
    final_output_text = response.choices[0].message.content.strip()
    
    # マークダウンのコードブロック記法を除去する
    if final_output_text.startswith("```json"):
        final_output_text = final_output_text[len("```json"):].strip()
    if final_output_text.endswith("```"):
        final_output_text = final_output_text[:-3].strip()
    
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