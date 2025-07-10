import os
import glob
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise RuntimeError('.env dosyasında GEMINI_API_KEY tanımlı değil')

genai.configure(api_key=GEMINI_API_KEY)

# FastAPI uygulamasını başlat
app = FastAPI(title="Siber Güvenlik Eğitim Asistanı API")

# Model ve FAISS index ayarları
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DOCS_FOLDER = 'training_documents'

class DocumentChunk:
    def __init__(self, text: str, source: str, idx: int):
        self.text = text
        self.source = source
        self.idx = idx

class RAGRetriever:
    def __init__(self, docs_folder: str, embedding_model_name: str):
        self.docs_folder = docs_folder
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks: List[DocumentChunk] = []
        self.embeddings = None
        self.index = None
        self._load_and_index()

    def _load_and_index(self):
        # Tüm .txt dosyalarını bul
        txt_files = glob.glob(os.path.join(self.docs_folder, '*.txt'))
        paragraphs = []
        meta = []
        for file in txt_files:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Çift satır boşluk ile paragraflara böl
                for idx, para in enumerate([p.strip() for p in content.split('\n\n') if p.strip()]):
                    self.chunks.append(DocumentChunk(para, os.path.basename(file), idx))
                    paragraphs.append(para)
                    meta.append((os.path.basename(file), idx))
        if not paragraphs:
            raise RuntimeError('training_documents/ klasöründe eğitim dokümanı bulunamadı.')
        # Paragrafları gömme vektörlerine dönüştür
        self.embeddings = self.embedding_model.encode(paragraphs, show_progress_bar=True, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        # FAISS index oluştur
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 1):
        # Sorgu için en yakın paragraf(lar)ı bul
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

retriever = RAGRetriever(DOCS_FOLDER, EMBEDDING_MODEL_NAME)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    context: str
    source: str
    source_paragraph_index: int

@app.post('/ask', response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    # En alakalı paragrafı getir
    chunks = retriever.retrieve(req.question, top_k=1)
    if not chunks:
        raise HTTPException(status_code=404, detail="Uygun bir bağlam bulunamadı.")
    context = chunks[0].text
    source = chunks[0].source
    para_idx = chunks[0].idx
    # Gemini için prompt oluştur
    prompt = f"Bir siber güvenlik eğitim asistanısın. Kullanıcının sorusunu aşağıdaki bağlamı kullanarak yanıtla.\n\nBağlam:\n{context}\n\nSoru: {req.question}\n\nCevap:"
    try:
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API hatası: {e}")
    return AskResponse(answer=answer, context=context, source=source, source_paragraph_index=para_idx)

@app.get('/')
def root():
    return {"message": "Siber Güvenlik Eğitim Asistanı API. POST /ask ile örnek JSON: { 'question': 'Phishing nedir?' }"} 