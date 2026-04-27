#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import OrderedDict

# ====================== CONFIG ======================
README_PATH = Path("README.md")
TOKEN_SCRIPT = Path("token_counter_qwen.py")
TABLE_HEADER = "| Title | Chars | Words | Tokens<br>(Qwen3) |"
TABLE_SEPARATOR = "| --- | --- | --- | --- |"
# ===================================================

def get_txt_files() -> list[Path]:
    return sorted(
        [f for f in Path(".").iterdir() if f.is_file() and f.suffix.lower() == ".txt"],
        key=lambda x: x.name.lower()
    )


def count_chars_words(file_path: Path) -> tuple[int, int]:
    text = file_path.read_text(encoding="utf-8")
    chars = len(text)
    words = len(re.findall(r'\b\w+\b', text))
    return chars, words


def get_token_count(file_path: Path) -> int:
    try:
        result = subprocess.run(
            [sys.executable, str(TOKEN_SCRIPT), str(file_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120
        )
        # Extract the final integer
        match = re.search(r'(\d+)\s*$', result.stdout.strip())
        if match:
            return int(match.group(1))
        numbers = re.findall(r'\d+', result.stdout)
        return int(numbers[-1]) if numbers else 0
    except Exception as e:
        print(f"⚠️ Token count failed for {file_path.name}: {e}")
        return 0


def parse_existing_table(content: str):
    """Returns (before_table_lines, existing_data_dict, after_table_lines)"""
    lines = content.splitlines()
    table_start = -1

    for i, line in enumerate(lines):
        if TABLE_HEADER.strip() in line:
            table_start = i
            break

    if table_start == -1:
        # No table yet → whole file is "before"
        return lines, OrderedDict(), []

    # Find end of table
    table_end = len(lines)
    for i in range(table_start + 2, len(lines)):
        if not lines[i].strip().startswith("|"):
            table_end = i
            break

    before = lines[:table_start]
    table_section = lines[table_start:table_end]
    after = lines[table_end:]

    # Parse existing rows
    data = OrderedDict()
    for line in table_section[2:]:  # skip header + separator
        if not line.strip() or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3:
            title = parts[0]
            data[title] = {
                "chars": parts[1],
                "words": parts[2],
                "tokens": parts[3] if len(parts) > 3 else ""
            }

    return before + table_section[:2], data, after


def main():
    txt_files = get_txt_files()
    if not txt_files:
        print("No .txt files found.")
        return

    # Read current README
    if README_PATH.exists():
        content = README_PATH.read_text(encoding="utf-8")
    else:
        content = "# Books\n\n> Public Domain, full text in plain UTF-8. Sourced from [Project Gutenberg](https://www.gutenberg.org/).\n\n"

    before_table, existing_data, after_table = parse_existing_table(content)

    print(f"Found {len(txt_files)} .txt files. Updating table...\n")

    new_rows = []
    for txt in txt_files:
        title = txt.name
        print(f"→ Processing {title}")

        chars, words = count_chars_words(txt)
        tokens = get_token_count(txt)

        new_rows.append(
            f"| {title} | {chars:,} | {words:,} | {tokens:,} |"
        )

    # Rebuild content
    new_content = "\n".join(before_table) + "\n" + "\n".join(new_rows) + "\n"

    if after_table:
        new_content += "\n" + "\n".join(after_table)

    README_PATH.write_text(new_content.strip() + "\n", encoding="utf-8")
    print(f"\n✅ README.md successfully updated with {len(txt_files)} books!")


if __name__ == "__main__":
    main()
