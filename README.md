# Speech-to-Speech Translator with BLEU Evaluation

This project allows users to upload an English audio file, transcribe it, translate it into a selected language, evaluate the translation with a BLEU score, and listen to the translated speech.

## Features
- Speech recognition using OpenAI Whisper
- Translation using Facebook M2M100 model
- BLEU score evaluation with sacrebleu
- Text-to-speech using gTTS
- Interactive interface using Gradio

## Setup

```bash
pip install -r requirements.txt
python app.py
