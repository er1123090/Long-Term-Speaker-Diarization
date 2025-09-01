# Long-Term-Speaker-Diarization

## GPT speaker attribution example

The script `gpt_speaker_attribution.py` uses the `data/locomo10.json` dataset to
query a GPT model for the speaker of each line in a session and reports the
accuracy of the model's predictions.

### Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide an OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```
3. Run the evaluation script:
   ```bash
   python gpt_speaker_attribution.py
   ```
