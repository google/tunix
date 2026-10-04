# Portions copied verbatim from OpenHands (https://github.com/OpenHands/OpenHands)
# commit 0d766ad06b2be64a42e6f0175b9ebcc4a06599d9:
#   openhands/agenthub/opencode_agent/opencode_impl.py (levenshtein and the
#   fuzzy replacers) and openhands/agenthub/opencode_agent/opencode_editor.py
#   (fuzzy_find_match).
#
# The MIT License (MIT)
#
# Copyright © 2025
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""OpenCode fuzzy matching used by the reference str_replace_editor.

The MLPerf reference harness (OpenHands 0d766ad06) runs str_replace through
OpenCodeEditor, which falls back to these replacers when old_str matches
neither exactly nor after strip(). The functions are kept verbatim so that
run_oh_editor_locally reproduces the reference edits; their source is also
shipped into the sandbox by _build_oh_editor_remote_cmd, so this module must
stay self-contained and Python 3.7 compatible.
"""

from __future__ import annotations

import re
from typing import Generator

# Similarity thresholds for block anchor fallback matching
SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.0
MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.3


def levenshtein(a: str, b: str) -> int:
    """Levenshtein distance algorithm implementation."""
    if a == "" or b == "":
        return max(len(a), len(b))

    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )

    return matrix[len(a)][len(b)]


def simple_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Simple exact match replacer."""
    yield find


def line_trimmed_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match by trimmed lines."""
    original_lines = content.split('\n')
    search_lines = find.split('\n')

    if search_lines and search_lines[-1] == '':
        search_lines.pop()

    for i in range(len(original_lines) - len(search_lines) + 1):
        matches = True

        for j in range(len(search_lines)):
            original_trimmed = original_lines[i + j].strip()
            search_trimmed = search_lines[j].strip()

            if original_trimmed != search_trimmed:
                matches = False
                break

        if matches:
            match_start_index = sum(len(original_lines[k]) + 1 for k in range(i))
            match_end_index = match_start_index
            for k in range(len(search_lines)):
                match_end_index += len(original_lines[i + k])
                if k < len(search_lines) - 1:
                    match_end_index += 1

            yield content[match_start_index:match_end_index]


def block_anchor_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match by first and last line anchors with similarity check."""
    original_lines = content.split('\n')
    search_lines = find.split('\n')

    if len(search_lines) < 3:
        return

    if search_lines and search_lines[-1] == '':
        search_lines.pop()

    first_line_search = search_lines[0].strip()
    last_line_search = search_lines[-1].strip()
    search_block_size = len(search_lines)

    # Collect all candidate positions
    candidates = []
    for i in range(len(original_lines)):
        if original_lines[i].strip() != first_line_search:
            continue

        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_line_search:
                candidates.append({'start_line': i, 'end_line': j})
                break

    if not candidates:
        return

    # Handle single candidate
    if len(candidates) == 1:
        start_line = candidates[0]['start_line']
        end_line = candidates[0]['end_line']
        actual_block_size = end_line - start_line + 1

        similarity = 0
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check > 0:
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                original_line = original_lines[start_line + j].strip()
                search_line = search_lines[j].strip()
                max_len = max(len(original_line), len(search_line))
                if max_len == 0:
                    continue
                distance = levenshtein(original_line, search_line)
                similarity += (1 - distance / max_len) / lines_to_check

                if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
                    break
        else:
            similarity = 1.0

        if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            match_start_index = sum(len(original_lines[k]) + 1 for k in range(start_line))
            match_end_index = match_start_index
            for k in range(start_line, end_line + 1):
                match_end_index += len(original_lines[k])
                if k < end_line:
                    match_end_index += 1
            yield content[match_start_index:match_end_index]
        return

    # Multiple candidates - find best match
    best_match = None
    max_similarity = -1

    for candidate in candidates:
        start_line = candidate['start_line']
        end_line = candidate['end_line']
        actual_block_size = end_line - start_line + 1

        similarity = 0
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check > 0:
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                original_line = original_lines[start_line + j].strip()
                search_line = search_lines[j].strip()
                max_len = max(len(original_line), len(search_line))
                if max_len == 0:
                    continue
                distance = levenshtein(original_line, search_line)
                similarity += 1 - distance / max_len
            similarity /= lines_to_check
        else:
            similarity = 1.0

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = candidate

    if max_similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD and best_match:
        start_line = best_match['start_line']
        end_line = best_match['end_line']
        match_start_index = sum(len(original_lines[k]) + 1 for k in range(start_line))
        match_end_index = match_start_index
        for k in range(start_line, end_line + 1):
            match_end_index += len(original_lines[k])
            if k < end_line:
                match_end_index += 1
        yield content[match_start_index:match_end_index]


def whitespace_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with normalized whitespace."""
    def normalize_whitespace(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    normalized_find = normalize_whitespace(find)
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if normalize_whitespace(line) == normalized_find:
            yield line
        else:
            normalized_line = normalize_whitespace(line)
            if normalized_find in normalized_line:
                words = find.strip().split()
                if words:
                    pattern = r'\s+'.join(re.escape(word) for word in words)
                    try:
                        match = re.search(pattern, line)
                        if match:
                            yield match.group(0)
                    except re.error:
                        pass

    # Handle multi-line matches
    find_lines = find.split('\n')
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = lines[i:i + len(find_lines)]
            if normalize_whitespace('\n'.join(block)) == normalized_find:
                yield '\n'.join(block)


def indentation_flexible_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with flexible indentation."""
    def remove_indentation(text: str) -> str:
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return text

        min_indent = min(
            len(l) - len(l.lstrip())
            for l in non_empty_lines
        )

        return '\n'.join(
            line if not line.strip() else line[min_indent:]
            for line in lines
        )

    normalized_find = remove_indentation(find)
    content_lines = content.split('\n')
    find_lines = find.split('\n')

    for i in range(len(content_lines) - len(find_lines) + 1):
        block = '\n'.join(content_lines[i:i + len(find_lines)])
        if remove_indentation(block) == normalized_find:
            yield block


def escape_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with normalized escape sequences."""
    def unescape_string(s: str) -> str:
        replacements = {
            '\\n': '\n', '\\t': '\t', '\\r': '\r',
            "\\'": "'", '\\"': '"', '\\`': '`',
            '\\\\': '\\', '\\$': '$'
        }
        for escaped, unescaped in replacements.items():
            s = s.replace(escaped, unescaped)
        return s

    unescaped_find = unescape_string(find)

    if unescaped_find in content:
        yield unescaped_find

    lines = content.split('\n')
    find_lines = unescaped_find.split('\n')

    for i in range(len(lines) - len(find_lines) + 1):
        block = '\n'.join(lines[i:i + len(find_lines)])
        unescaped_block = unescape_string(block)

        if unescaped_block == unescaped_find:
            yield block


def multi_occurrence_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Yield all exact matches."""
    start_index = 0
    while True:
        index = content.find(find, start_index)
        if index == -1:
            break
        yield find
        start_index = index + len(find)


def trimmed_boundary_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match with trimmed boundaries."""
    trimmed_find = find.strip()

    if trimmed_find == find:
        return

    if trimmed_find in content:
        yield trimmed_find

    lines = content.split('\n')
    find_lines = find.split('\n')

    for i in range(len(lines) - len(find_lines) + 1):
        block = '\n'.join(lines[i:i + len(find_lines)])
        if block.strip() == trimmed_find:
            yield block


def context_aware_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match using context anchors."""
    find_lines = find.split('\n')
    if len(find_lines) < 3:
        return

    if find_lines and find_lines[-1] == '':
        find_lines.pop()

    content_lines = content.split('\n')
    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()

    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue

        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                block_lines = content_lines[i:j + 1]
                block = '\n'.join(block_lines)

                if len(block_lines) == len(find_lines):
                    matching_lines = 0
                    total_non_empty = 0

                    for k in range(1, len(block_lines) - 1):
                        block_line = block_lines[k].strip()
                        find_line = find_lines[k].strip()

                        if block_line or find_line:
                            total_non_empty += 1
                            if block_line == find_line:
                                matching_lines += 1

                    if total_non_empty == 0 or matching_lines / total_non_empty >= 0.5:
                        yield block
                        break
                break


def fuzzy_find_match(content: str, old_str: str) -> str | None:
    """
    Try to find a match using fuzzy matching replacers.
    Returns the matched string if found (unique match), None otherwise.
    """
    replacers = [
        simple_replacer,
        line_trimmed_replacer,
        block_anchor_replacer,
        whitespace_normalized_replacer,
        indentation_flexible_replacer,
        escape_normalized_replacer,
        trimmed_boundary_replacer,
        context_aware_replacer,
        multi_occurrence_replacer,
    ]

    for replacer in replacers:
        matches = list(replacer(content, old_str))
        for search in matches:
            index = content.find(search)
            if index == -1:
                continue

            last_index = content.rfind(search)
            # Only return if unique match
            if index == last_index:
                return search

    return None
