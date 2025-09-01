import json
import os
import re
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("The openai package is required to run this script. Install it via 'pip install openai'.")


def predict_speakers(client: OpenAI, speaker_a: str, speaker_b: str, lines: List[str]) -> Dict[str, str]:
    """Ask a GPT model to predict which speaker spoke each line."""
    conversation = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    prompt = (
        f"Given a conversation between two speakers: {speaker_a} and {speaker_b}.\n"
        f"Each line is numbered but speaker names are hidden.\n"
        f"Return a JSON object mapping line numbers (as strings) to the speaker who said it.\n\n"
        f"Conversation:\n{conversation}\n"
    )
    response = client.responses.create(model="gpt-4o-mini", input=prompt)
    text = response.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        mapping = {}
        for line in text.splitlines():
            m = re.match(r"\s*\"?(\d+)\"?\s*[:\-]\s*\"?([A-Za-z]+)\"?", line)
            if m:
                mapping[m.group(1)] = m.group(2)
        return mapping


def main() -> None:
    with open("data/locomo10.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    total_correct = 0
    total_lines = 0

    for conv_idx, sample in enumerate(dataset, start=1):
        conv = sample["conversation"]
        speaker_a = conv["speaker_a"]
        speaker_b = conv["speaker_b"]

        for session_idx in range(1, 11):
            key = f"session_{session_idx}"
            if key not in conv:
                continue

            session_data = conv[key]
            if not session_data:
                continue

            lines = [turn["text"] for turn in session_data]
            gold = [turn["speaker"] for turn in session_data]

            prediction = predict_speakers(client, speaker_a, speaker_b, lines)
            correct = sum(
                1 for i, spk in enumerate(gold, start=1) if prediction.get(str(i)) == spk
            )
            accuracy = correct / len(gold) if gold else 0
            total_correct += correct
            total_lines += len(gold)
            print(
                f"Conversation {conv_idx} Session {session_idx}: {accuracy:.2%} accuracy"
            )

    overall = total_correct / total_lines if total_lines else 0
    print(f"Overall accuracy across all sessions: {overall:.2%}")


if __name__ == "__main__":
    main()
