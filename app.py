import torch
import torchaudio
import numpy as np
import soundfile as sf
import tempfile
import os
from PIL import Image
from gtts import gTTS
from transformers import pipeline, BlipProcessor, BlipForQuestionAnswering
import gradio as gr

class SpeechToText:
    def __init__(self):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def transcribe(self, audio_path):
        try:
            result = self.pipe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Error in transcription: {e}")
            return None

def process_audio(audio):
    if audio is None:
        return None
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    temp_file = "temp_audio.wav"
    sf.write(temp_file, y, sr)
    return temp_file

class ImageQA:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def answer_question(self, image_path, question):
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, question, return_tensors="pt").to(
                "cuda" if torch.cuda.is_available() else "cpu",
                torch.float16
            )
            out = self.model.generate(**inputs)
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"Error in QA: {e}")
            return "Sorry, I couldn't process that question."

class TextToSpeech:
    def __init__(self):
        self.temp_file = "temp_tts.mp3"

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(self.temp_file)
            return self.temp_file
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

stt = SpeechToText()
qa = ImageQA()
tts = TextToSpeech()

def process_query(audio, image, text_question):
    question = ""

    if audio is not None:
        audio_path = process_audio(audio)
        if audio_path:
            question = stt.transcribe(audio_path)
            os.remove(audio_path)

    if not question and text_question:
        question = text_question

    if not question:
        return "❗ Please ask a question (voice or text)", None

    if image is None:
        return "❗ Please upload an image", None

    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_img.name)

    answer = qa.answer_question(temp_img.name, question)
    os.unlink(temp_img.name)

    tts_path = tts.speak(answer)
    return answer, tts_path

def main():
    with gr.Blocks(
        theme=gr.themes.Base(),
        title="🔥 PicSage - Ask-the-Image AI",
        css="""
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

            body {
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                color: #fff;
                font-family: 'Orbitron', sans-serif;
                margin: 0;
                padding: 0;
            }

            .gradio-container {
                width: 100%;
                min-height: 100vh;
                padding: 40px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                backdrop-filter: blur(10px);
                background: rgba(0, 0, 0, 0.5);
                box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.8);
            }

            .gradio-row {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                width: 100%;
                margin-top: 30px;
            }

            .rounded-lg {
                border-radius: 20px !important;
            }

            .shadow-lg {
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6) !important;
            }

            button:hover {
                transform: scale(1.1);
                transition: all 0.3s ease-in-out;
            }

            .header {
                text-align: center;
                font-size: 3rem;
                font-weight: 800;
                color: #00FFD1;
                text-shadow: 0 0 15px #00FFD1;
            }

            .sub-header {
                text-align: center;
                font-size: 1.2rem;
                color: #aaa;
                margin-bottom: 20px;
            }
        """
    ) as demo:

        gr.Markdown("<div class='header'>🤖 PicSage - Image QA Deluxe</div>")
        gr.Markdown("<div class='sub-header'>Ask any question about your image via text or voice. Get instant answers and listen to them too!</div>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ Record or Type Your Question")
                audio_input = gr.Audio(label="🎤 Record Question (10 sec max)", type="numpy", elem_classes="rounded-lg shadow-lg")
                text_input = gr.Textbox(label="💬 Or Type Your Question", placeholder="e.g., What is in this image?", elem_classes="rounded-lg shadow-lg")

                gr.Markdown("### 🖼️ Upload Your Image")
                image_input = gr.Image(label="📸 Upload Image", type="pil", elem_classes="rounded-lg shadow-lg")

                submit_btn = gr.Button("🚀 Ask the Image!", elem_classes="bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold py-2 px-4 rounded-lg mt-4 shadow hover:scale-105 transition-all duration-300")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 💡 AI's Answer")
                    answer_output = gr.Textbox(
                        label="🧠 Answer",
                        interactive=False,
                        lines=5,
                        elem_classes="rounded-lg shadow-lg text-lg p-4"
                    )
                    gr.Markdown("### 🔊 Listen to the Answer")
                    audio_output = gr.Audio(label="🔈 Spoken Answer", visible=True, elem_classes="rounded-lg shadow-lg")

        with gr.Accordion("❓ Help & Tips", open=False):
            gr.Markdown("""
                - Upload **clear images** for best results.
                - Ask **specific questions** (e.g., "What brand is this shoe?").
                - Use the **record button** if you prefer speaking.
            """)

        submit_btn.click(
            fn=process_query,
            inputs=[audio_input, image_input, text_input],
            outputs=[answer_output, audio_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
