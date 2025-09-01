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
    filename="locomo10_250901_14-20.log",
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

    messages: List[Dict[str, str]] = []

    if hide_names:
        system_msg = (
            "You will be given a conversation between two speakers: "
            f"{speaker_a} and {speaker_b}. The conversation messages use the "
            "placeholder roles 'Speaker 1' and 'Speaker 2'. The first message "
            "after this system prompt is message 1, the next is message 2, and "
            "so on. Determine which real speaker said each message and return a "
            "JSON object mapping message numbers (as strings) to the speaker's "
            "name."
        )
        messages.append({"role": "system", "content": system_msg})

        for turn in turns:
            placeholder = "Speaker 1" if turn["speaker"] == speaker_a else "Speaker 2"
            messages.append({"role": placeholder, "content": turn["text"]})
    else:
        system_msg = (
            "You will be given a conversation between two speakers. Each "
            "message's role is the speaker's name. The first message after this "
            "system prompt is message 1, the next is message 2, and so on. "
            "Return a JSON object mapping message numbers (as strings) to the "
            "speaker's name."
        )
        messages.append({"role": "system", "content": system_msg})

        for turn in turns:
            messages.append({"role": turn["speaker"], "content": turn["text"]})

    messages.append(
        {
            "role": "user",
            "content": "Respond with the JSON mapping now.",
        }
    )

    version = "hidden" if hide_names else "names"

    logger.info(
        "Conversation %d Session %d (%s) GPT INPUT:\n%s",
        conv_idx,
        session_idx,
        version,
        json.dumps(messages, ensure_ascii=False, indent=2),
    )

    response = client.responses.create(model="gpt-4o-mini", input=messages)
    text = response.output_text.strip()

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

    results_hidden: Dict[str, Dict[str, Dict[str, object]]] = {}
    results_names: Dict[str, Dict[str, Dict[str, object]]] = {}

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

                conv_key = f"conversation{conv_idx}"
                sess_key = f"session{session_idx}"
                results_hidden.setdefault(conv_key, {})[sess_key] = {
                    "answers": gold,
                    "predict": [
                        prediction_hidden.get(str(i), "")
                        for i in range(1, len(gold) + 1)
                    ],
                    "accuracy": accuracy_hidden,
                }

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

                conv_key = f"conversation{conv_idx}"
                sess_key = f"session{session_idx}"
                results_names.setdefault(conv_key, {})[sess_key] = {
                    "answers": gold,
                    "predict": [
                        prediction_names.get(str(i), "")
                        for i in range(1, len(gold) + 1)
                    ],
                    "accuracy": accuracy_names,
                }

    if mode in ("hidden", "both"):
        overall_hidden = (
            total_correct_hidden / total_lines_hidden if total_lines_hidden else 0
        )
        summary_hidden = (
            f"Overall accuracy across all sessions (hidden): {overall_hidden:.2%}"
        )
        print(summary_hidden)
        results_hidden["overall_accuracy"] = overall_hidden

        with open("outputs_hidden.json", "w", encoding="utf-8") as f_out:
            json.dump(results_hidden, f_out, ensure_ascii=False, indent=2)

    if mode in ("names", "both"):
        overall_names = (
            total_correct_names / total_lines_names if total_lines_names else 0
        )
        summary_names = (
            f"Overall accuracy across all sessions (names): {overall_names:.2%}"
        )
        print(summary_names)
        results_names["overall_accuracy"] = overall_names

        with open("outputs_names.json", "w", encoding="utf-8") as f_out:
            json.dump(results_names, f_out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
