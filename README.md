# 🩺 Medical Q&A with RAG

[![Live Space](https://img.shields.io/badge/Live-Demo-green?style=flat&logo=huggingface)](https://kallurayaankit-medical-qa.hf.space)

A Retrieval‑Augmented Generation (RAG) application that answers medical questions using the **Hugging Face Inference API** and a knowledge base built from WHO fact sheets.

---

## 🧠 How It Works

1. **User asks a question** (e.g., “What are the symptoms of diabetes?”).
2. **Chroma vector database** retrieves the most relevant medical facts (from the WHO).
3. **Hugging Face Inference** (free model) generates an answer based on the retrieved context.
4. **FastAPI** serves the response via a public REST endpoint.

---

## 🔗 Live Demo

- **API endpoint:** `https://kallurayaankit-medical-qa.hf.space/ask`  
- **Interactive docs:** `https://kallurayaankit-medical-qa.hf.space/docs`

Try it yourself:

```bash
curl -X POST "https://kallurayaankit-medical-qa.hf.space/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the symptoms of diabetes?"}'