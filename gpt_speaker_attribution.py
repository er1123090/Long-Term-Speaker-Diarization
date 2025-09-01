import json
import os
import re
import logging
import argparse
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("The openai package is required to run this script. Install it via 'pip install openai'.")


# =====================
# Logger configuration
# =====================
logging.basicConfig(
    filename="locomo10_250901_14-49.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)


def predict_speakers(
    client: OpenAI,
    speaker_a: str,
    speaker_b: str,
    turns: List[Dict[str, str]],
    conv_idx: int,
    session_idx: int,
    hide_names: bool,
) -> Dict[str, str]:
    """Ask a GPT model to predict which speaker spoke each line, with logging.

    Parameters
    ----------
    client: OpenAI
        The OpenAI client used for generation.
    speaker_a, speaker_b: str
        The two participant names.
    turns: List[Dict[str, str]]
        Conversation turns in order.
    conv_idx, session_idx: int
        Identifiers for logging.
    hide_names: bool
        Whether to hide speaker names in the prompt.
    """

    if hide_names:
        conversation = "\n".join(
            f"{i+1}. {turn['text']}" for i, turn in enumerate(turns)
        )
        prompt = (
            f"Given a conversation between two speakers: {speaker_a} and {speaker_b}.\n"
            f"Each line is numbered but speaker names are hidden.\n"
            f"Return a JSON object mapping line numbers (as strings) to the speaker who said it.\n\n"
            f"Conversation:\n{conversation}\n"
        )
    else:
        conversation = "\n".join(
            f"{i+1}. {turn['speaker']}: {turn['text']}" for i, turn in enumerate(turns)
        )
        prompt = (
            f"Given a conversation between two speakers: {speaker_a} and {speaker_b}.\n"
            f"Each line is numbered with speaker names included.\n"
            f"Return a JSON object mapping line numbers (as strings) to the speaker who said it.\n\n"
            f"Conversation:\n{conversation}\n"
        )

    version = "hidden" if hide_names else "names"

    # Log the input prompt with identifiers
    logger.info(
        "Conversation %d Session %d (%s) GPT INPUT:\n%s",
        conv_idx,
        session_idx,
        version,
        prompt,
    )

    response = client.responses.create(model="gpt-4o-mini", input=prompt)
    text = response.output_text.strip()

    # Log the raw output with identifiers
    logger.info(
        "Conversation %d Session %d (%s) GPT OUTPUT:\n%s",
        conv_idx,
        session_idx,
        version,
        text,
    )

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
    parser = argparse.ArgumentParser(
        description="Evaluate GPT speaker attribution accuracy"
    )
    parser.add_argument(
        "--mode",
        choices=["hidden", "names", "both"],
        default="both",
        help="Hide speaker names, show them, or evaluate both modes",
    )
    args = parser.parse_args()
    mode = args.mode

    with open("data/locomo10.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    total_correct_hidden = 0
    total_lines_hidden = 0
    total_correct_names = 0
    total_lines_names = 0

    output_file = open("outputs.txt", "w", encoding="utf-8")

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

            gold = [turn["speaker"] for turn in session_data]

            if mode in ("hidden", "both"):
                prediction_hidden = predict_speakers(
                    client,
                    speaker_a,
                    speaker_b,
                    session_data,
                    conv_idx,
                    session_idx,
                    hide_names=True,
                )
                correct_hidden = sum(
                    1 for i, spk in enumerate(gold, start=1)
                    if prediction_hidden.get(str(i)) == spk
                )
                accuracy_hidden = correct_hidden / len(gold) if gold else 0
                total_correct_hidden += correct_hidden
                total_lines_hidden += len(gold)
                line_hidden = (
                    f"Conversation {conv_idx} Session {session_idx} (hidden): {accuracy_hidden:.2%} accuracy"
                )
                print(line_hidden)
                output_file.write(line_hidden + "\n")

            if mode in ("names", "both"):
                prediction_names = predict_speakers(
                    client,
                    speaker_a,
                    speaker_b,
                    session_data,
                    conv_idx,
                    session_idx,
                    hide_names=False,
                )
                correct_names = sum(
                    1 for i, spk in enumerate(gold, start=1)
                    if prediction_names.get(str(i)) == spk
                )
                accuracy_names = correct_names / len(gold) if gold else 0
                total_correct_names += correct_names
                total_lines_names += len(gold)
                line_names = (
                    f"Conversation {conv_idx} Session {session_idx} (names): {accuracy_names:.2%} accuracy"
                )
                print(line_names)
                output_file.write(line_names + "\n")

    if mode in ("hidden", "both"):
        overall_hidden = (
            total_correct_hidden / total_lines_hidden if total_lines_hidden else 0
        )
        summary_hidden = (
            f"Overall accuracy across all sessions (hidden): {overall_hidden:.2%}"
        )
        print(summary_hidden)
        output_file.write(summary_hidden + "\n")

    if mode in ("names", "both"):
        overall_names = (
            total_correct_names / total_lines_names if total_lines_names else 0
        )
        summary_names = (
            f"Overall accuracy across all sessions (names): {overall_names:.2%}"
        )
        print(summary_names)
        output_file.write(summary_names + "\n")

    output_file.close()


if __name__ == "__main__":
    main()
