# 🤖 PicSage - Ask-the-Image AI

**PicSage** is a futuristic AI-powered web app that allows users to ask questions about any image using **voice or text**, and receive intelligent answers both in text and spoken form. It combines cutting-edge models from **OpenAI**, **Salesforce**, and **Google** into a seamless Gradio-powered interface.

---

## 🚀 Features

- 🎙️ **Speech-to-Text**: Ask your question by speaking (powered by OpenAI Whisper).
- 💬 **Text Input**: Alternatively, type your question.
- 🖼️ **Visual Question Answering**: Upload any image and get intelligent answers (powered by Salesforce BLIP).
- 🔊 **Text-to-Speech**: Hear the AI-generated answer read out loud (powered by gTTS).
- ⚡ **Interactive Interface**: Built with Gradio and styled with modern CSS.

---

## 🧠 Tech Stack

| Component            | Model/Library                      |
|---------------------|------------------------------------|
| Speech-to-Text      | `openai/whisper-small` (HuggingFace) |
| Image QA            | `Salesforce/blip-vqa-base`         |
| Text-to-Speech      | `gTTS` (Google Text-to-Speech)     |
| UI Framework        | `Gradio`                           |
| Audio/Image Handling| `torchaudio`, `soundfile`, `PIL`   |

---

## 📸 Demo

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-demo-gif.gif" alt="demo" width="600"/>
</p>

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/picsage-ai.git
cd picsage-ai
