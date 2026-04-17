import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader


HEADING_PATTERNS = {
    "introduction": [
        r"^introduction\b.*$",
        r"^background\b.*$",
    ],
    "methods": [
        r"^materials?\s+and\s+methods\b.*$",
        r"^methods?\b.*$",
        r"^experimental\s+procedures?\b.*$",
    ],
    "results": [
        r"^results?\b.*$",
        r"^findings\b.*$",
    ],
    "discussion": [
        r"^discussion\b.*$",
        r"^conclusions?\b.*$",
        r"^results\s+and\s+discussion\b.*$",
    ],
    "references": [
        r"^references?\b.*$",
        r"^bibliograph(y|ies)\b.*$",
        r"^literature\s+cited\b.*$",
        r"^works\s+cited\b.*$",
    ],
}

REFERENCE_START_PATTERN = re.compile(r"^\s*(\[\d+\]|\d+\.)\s+")


DEFAULT_LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3.5-9b")
DEFAULT_LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

BODY_HEADING_SYSTEM_PROMPT = """You are given candidate heading windows from the body of a scientific paper.

Each candidate is wrapped like this:
[CANDIDATE_START line=120]
[L118] previous line
[L119] previous line
[L120] candidate heading line
[L121] next line
[L122] next line
[CANDIDATE_END]

Identify only true major body section headings and return exactly one valid JSON object.

Output schema:
{
  "headings": [
    {
      "section": "results",
      "start_line": 120,
      "heading_text": "Results"
    }
  ]
}

Allowed section values:
- introduction
- methods
- results
- discussion
- results_and_discussion
- conclusion

Rules:
- Return JSON only.
- Do not output reasoning.
- Use source line numbers exactly as shown in the input.
- Include only headings that are explicitly present in the text.
- Use the local context around each candidate to reject running headers, page labels, figure captions, table captions, references headings, and inline mentions.
- "Results and Discussion" must be returned as results_and_discussion.
- "Conclusion" and "Conclusions" must be returned as conclusion.
- Do not invent missing headings.
"""


def parse_json_response(response_text: str):
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    for payload in extract_balanced_json_candidates(text):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON found in response: {text[:300]!r}")


def create_lmstudio_client(base_url: str = DEFAULT_LMSTUDIO_BASE_URL) -> OpenAI:
    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=DEFAULT_LMSTUDIO_API_KEY,
    )


def request_chat_completion(
    messages: list[dict],
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
    model: str = DEFAULT_LMSTUDIO_MODEL,
) -> str:
    client = create_lmstudio_client(base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def extract_balanced_json_candidates(text: str):
    candidates = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        for start in range(len(text)):
            if text[start] != opener:
                continue
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : index + 1])
                        break
    candidates.sort(key=len, reverse=True)
    return candidates


def is_candidate_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    if len(stripped.split()) > 12:
        return False
    if REFERENCE_START_PATTERN.match(stripped):
        return False
    if re.search(r"\bdoi\b|www\.|http|@\w+", stripped, flags=re.IGNORECASE):
        return False
    if re.search(r"^fig\.|^figure\s+\d+|^table\s+\d+", stripped, flags=re.IGNORECASE):
        return False
    if re.search(r"^\d+$", stripped):
        return False

    heading_like_patterns = [
        r"^(introduction|background)\b",
        r"^(materials?\s+and\s+methods|methods?|experimental\s+procedures?)\b",
        r"^(results?|findings)\b",
        r"^(discussion|results\s+and\s+discussion)\b",
        r"^(conclusions?)\b",
    ]
    if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in heading_like_patterns):
        return True

    alpha_ratio = sum(char.isalpha() for char in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.6:
        return False
    if stripped.endswith("."):
        return False
    if stripped.isupper():
        return True
    return bool(re.match(r"^[A-Z][A-Za-z0-9,\-–/&()' ]+$", stripped))


def build_candidate_heading_windows(lines: list[str], radius: int = 2) -> str:
    windows = []
    for index, line in enumerate(lines, start=1):
        if not is_candidate_heading_line(line):
            continue
        start = max(1, index - radius)
        end = min(len(lines), index + radius)
        windows.append(f"[CANDIDATE_START line={index}]")
        for line_no in range(start, end + 1):
            windows.append(f"[L{line_no:03d}] {lines[line_no - 1].strip()}")
        windows.append("[CANDIDATE_END]")
    return "\n".join(windows)


def detect_body_headings_with_llm(
    lines: list[str],
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
    model: str = DEFAULT_LMSTUDIO_MODEL,
) -> dict:
    numbered_text = build_candidate_heading_windows(lines)
    if not numbered_text.strip():
        return {"headings": []}

    response_text = request_chat_completion(
        messages=[
            {"role": "system", "content": BODY_HEADING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Return only one JSON object.\n\nCandidate heading windows with local context:\n" + numbered_text,
            },
        ],
        base_url=base_url,
        model=model,
    )
    parsed = parse_json_response(response_text)
    return validate_llm_heading_response(parsed, len(lines))


def validate_llm_heading_response(response: dict, total_lines: int) -> dict:
    allowed_sections = {
        "introduction",
        "methods",
        "results",
        "discussion",
        "results_and_discussion",
        "conclusion",
    }
    headings = response.get("headings")
    if not isinstance(response, dict) or not isinstance(headings, list):
        raise ValueError(f"Expected JSON object with headings list: {response!r}")

    validated = []
    for item in headings:
        if not isinstance(item, dict):
            raise ValueError(f"Heading item must be an object: {item!r}")
        section = item.get("section")
        start_line = item.get("start_line")
        heading_text = item.get("heading_text")
        if section not in allowed_sections:
            raise ValueError(f"Unexpected section value: {item!r}")
        if not isinstance(start_line, int) or not 1 <= start_line <= total_lines:
            raise ValueError(f"Invalid heading line number: {item!r}")
        if not isinstance(heading_text, str) or not heading_text.strip():
            raise ValueError(f"heading_text must be a non-empty string: {item!r}")
        validated.append(
            {
                "section": section,
                "start_line": start_line,
                "heading_text": heading_text.strip(),
            }
        )

    validated.sort(key=lambda item: item["start_line"])
    return {"headings": validated}


def find_heading(lines, patterns, start=0, end=None):
    if end is None:
        end = len(lines)
    for index in range(start, end):
        line = lines[index].strip()
        for pattern in patterns:
            if re.match(pattern, line, flags=re.IGNORECASE):
                return index
    return None


def is_reference_like_line(line):
    text = line.strip()
    if not text:
        return False
    patterns = [
        r"^\d+\.\s+[A-Z]",
        r"^[A-Z][A-Za-z\-']+,\s+[A-Z]\.",
        r"\(\d{4}\)",
        r"\bdoi:\s*10\.\d{4,9}/",
        r"\b10\.\d{4,9}/\S+",
        r"\bet al\.\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def find_references_start(lines):
    heading_idx = find_heading(lines, HEADING_PATTERNS["references"])
    window_size = 12
    min_hits = 4
    search_start = max(0, int(len(lines) * 0.5))
    flags = [is_reference_like_line(line) for line in lines]

    for start in range(search_start, max(search_start, len(lines) - window_size + 1)):
        if sum(flags[start : start + window_size]) >= min_hits:
            return min(heading_idx, max(0, start - 3)) if heading_idx is not None else max(0, start - 3)
    return heading_idx


def find_heading_index_with_patterns(lines: list[str], section_name: str):
    if section_name == "results_and_discussion":
        patterns = [r"^results\s+and\s+discussion\b.*$"]
    elif section_name == "conclusion":
        patterns = [r"^conclusions?\b.*$"]
    else:
        patterns = HEADING_PATTERNS.get(section_name, [])
    return find_heading(lines, patterns)


def build_section_boundaries(lines: list[str], llm_headings: list[dict] | None = None):
    refs_idx = find_references_start(lines)
    boundaries = []
    seen_sections = set()

    for item in llm_headings or []:
        if item["section"] in seen_sections:
            continue
        seen_sections.add(item["section"])
        boundaries.append(
            {
                "section": item["section"],
                "start_index": item["start_line"] - 1,
                "heading_text": item["heading_text"],
                "source": "llm",
            }
        )

    fallback_sections = [
        "introduction",
        "methods",
        "results_and_discussion",
        "results",
        "discussion",
        "conclusion",
    ]
    for section_name in fallback_sections:
        if section_name in seen_sections:
            continue
        start_index = find_heading_index_with_patterns(lines, section_name)
        if start_index is None:
            continue
        boundaries.append(
            {
                "section": section_name,
                "start_index": start_index,
                "heading_text": lines[start_index].strip(),
                "source": "pattern",
            }
        )

    boundaries.sort(key=lambda item: item["start_index"])
    filtered = []
    for item in boundaries:
        if refs_idx is not None and item["start_index"] >= refs_idx:
            continue
        if filtered and filtered[-1]["start_index"] == item["start_index"]:
            continue
        filtered.append(item)
    return filtered, refs_idx


def populate_body_sections(lines: list[str], boundaries: list[dict], refs_idx: int | None):
    sections = {
        "introduction": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "results_and_discussion": "",
        "conclusion": "",
    }
    section_boundaries = []

    for index, boundary in enumerate(boundaries):
        next_index = boundaries[index + 1]["start_index"] if index + 1 < len(boundaries) else len(lines)
        end_index = min(next_index, refs_idx) if refs_idx is not None else next_index
        section_name = boundary["section"]
        sections[section_name] = "\n".join(lines[boundary["start_index"] : end_index]).strip()
        section_boundaries.append(
            {
                "section": section_name,
                "start_line": boundary["start_index"] + 1,
                "end_line": end_index,
                "heading_text": boundary["heading_text"],
                "source": boundary["source"],
            }
        )

    return sections, section_boundaries


def normalize_five_section_output(sections: dict, section_boundaries: list[dict]):
    normalized = {
        "introduction": sections["introduction"],
        "methods": sections["methods"],
        "results": sections["results"],
        "discussion": sections["discussion"],
        "conclusion": sections["conclusion"],
        "results_and_discussion": sections["results_and_discussion"],
        "has_mixed_results_discussion": False,
    }

    normalized_boundaries = []
    for boundary in section_boundaries:
        is_mixed = boundary["section"] == "results_and_discussion" or re.match(
            r"^results\s+and\s+discussion\b",
            boundary["heading_text"],
            flags=re.IGNORECASE,
        )
        if is_mixed:
            normalized["has_mixed_results_discussion"] = True
            normalized_boundaries.append({**boundary, "section": "results_and_discussion", "mixed": True})
            continue
        normalized_boundaries.append({**boundary, "mixed": False})

    return normalized, normalized_boundaries


def read_input_text(input_path):
    if input_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(input_path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    return input_path.read_text(encoding="utf-8")


def split_paper_sections(
    text,
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
    model: str = DEFAULT_LMSTUDIO_MODEL,
    use_llm_heading_detection: bool = False,
):
    lines = text.splitlines()
    if not lines:
        return {
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "results_and_discussion": "",
            "has_mixed_results_discussion": False,
            "section_boundaries": [],
            "llm_heading_candidates": [],
        }

    llm_heading_candidates = []
    if use_llm_heading_detection:
        llm_heading_candidates = detect_body_headings_with_llm(
            lines,
            base_url=base_url,
            model=model,
        )["headings"]

    section_boundaries, refs_idx = build_section_boundaries(lines, llm_heading_candidates)
    body_sections, resolved_boundaries = populate_body_sections(lines, section_boundaries, refs_idx)
    normalized_sections, normalized_boundaries = normalize_five_section_output(
        body_sections,
        resolved_boundaries,
    )

    return {
        **normalized_sections,
        "section_boundaries": normalized_boundaries,
        "llm_heading_candidates": llm_heading_candidates,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="論文本文を5セクションに分割するテスト"
    )
    parser.add_argument("--input", required=True, help="入力ファイル（txt/pdf）")
    parser.add_argument("--output", default="imrad_sections.json", help="分割結果JSONの出力先")
    parser.add_argument("--model", default=DEFAULT_LMSTUDIO_MODEL, help="LM Studio model name")
    parser.add_argument("--base-url", default=DEFAULT_LMSTUDIO_BASE_URL, help="LM Studio OpenAI-compatible base URL")
    parser.add_argument(
        "--llm-detect-body-headings",
        action="store_true",
        help="本文見出しをLM Studioに判定させ、返却された行番号でsection分割する",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    text = read_input_text(input_path)
    sections = split_paper_sections(
        text,
        base_url=args.base_url,
        model=args.model,
        use_llm_heading_detection=args.llm_detect_body_headings,
    )

    output_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
