def check_brackets_with_context_line_number(file_path, context_chars=40):
    brackets = {
        '{': '}',
        '[': ']',
        '(': ')'
    }
    open_brackets = set(brackets.keys())
    close_brackets = set(brackets.values())

    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    stack = []
    errors = []

    # 줄 번호를 계산하기 위해 미리 줄 시작 인덱스 리스트 생성
    line_starts = [0]
    for idx, char in enumerate(data):
        if char == '\n':
            line_starts.append(idx + 1)

    def get_line_number(pos):
        # position → 해당 줄 번호(1-based)
        from bisect import bisect_right
        return bisect_right(line_starts, pos)

    for i, char in enumerate(data):
        if char in open_brackets:
            stack.append((char, i))
        elif char in close_brackets:
            if stack and brackets[stack[-1][0]] == char:
                stack.pop()
            else:
                start = max(0, i - context_chars)
                end = min(len(data), i + context_chars)
                context = data[start:end].replace("\n", "\\n")
                errors.append({
                    "type": "unmatched_closing",
                    "char": char,
                    "line_number": get_line_number(i),
                    "context": context
                })

    while stack:
        char, pos = stack.pop()
        start = max(0, pos - context_chars)
        end = min(len(data), pos + context_chars)
        context = data[start:end].replace("\n", "\\n")
        errors.append({
            "type": "unmatched_opening",
            "char": char,
            "line_number": get_line_number(pos),
            "context": context
        })

    return errors


# 사용 예시
file_path = "locomo10.json"  # 검사할 파일 경로
errors_with_context = check_brackets_with_context_line_number(file_path)

if errors_with_context:
    for err in errors_with_context:
        print(f"{err['type']} '{err['char']}' at line {err['line_number']}")
        print(f"Context: {err['context']}\n")
else:
    print("All brackets are correctly matched.")
