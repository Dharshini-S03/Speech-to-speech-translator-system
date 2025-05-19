import gradio as gr
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from gtts import gTTS
import pandas as pd
from sacrebleu import sentence_bleu

df = pd.read_csv("/content/a_dataset.csv")

print("Loading Whisper and M2M100 models (this may take a while)...")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

language_map = {
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Tamil": "ta"
}

def speech_to_speech_eval(audio, original_english, target_language_name):
    target_lang = language_map[target_language_name]
    source_lang = "en"  # Assuming audio is English

    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    input_features = whisper_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    predicted_english = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    m2m_tokenizer.src_lang = source_lang
    encoded = m2m_tokenizer(predicted_english, return_tensors="pt")
    generated_tokens = m2m_model.generate(**encoded, forced_bos_token_id=m2m_tokenizer.get_lang_id(target_lang))
    predicted_translation = m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    reference_translation = df.loc[df['English'] == original_english, target_language_name].values[0]

    bleu = sentence_bleu(predicted_translation, [reference_translation])
    bleu_score = round(bleu.score, 2)

    tts = gTTS(text=predicted_translation, lang=target_lang)
    output_audio = "output.mp3"
    tts.save(output_audio)

    return predicted_translation, reference_translation, f"BLEU Score: {bleu_score}", output_audio

english_sentences = df['English'].tolist()
target_langs = list(language_map.keys())

iface = gr.Interface(
    fn=speech_to_speech_eval,
    inputs=[
        gr.Audio(type="filepath", label="Upload English Audio (from dataset sentences)"),
        gr.Dropdown(choices=english_sentences, label="Select the original English sentence"),
        gr.Dropdown(choices=target_langs, label="Translate to Language")
    ],
    outputs=[
        gr.Textbox(label="Predicted Translation"),
        gr.Textbox(label="Reference Translation"),
        gr.Textbox(label="BLEU Score"),
        gr.Audio(type="filepath", label="Translated Speech Audio")
    ],
    title="Speech-to-Speech Translator with BLEU Error Evaluation",
    description="Upload an audio clip (English), select which sentence it is, pick target language, and see translation accuracy."
)

iface.launch()
