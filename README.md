# DocuRAG

Belge görsellerinden (fatura, sözleşme, dilekçe vb.) çıkarılan metinler üzerinde semantik arama yapabilen, çoklu dil modeli (cloud/local) desteğine uygun esnek bir RAG (Retrieval-Augmented Generation) pipeline'ı.

> 🚧 **Durum: Geliştirme aşamasında.** Aşağıdaki modüller tamamlanmış olup proje aktif olarak genişletilmektedir.

## Şu ana kadar tamamlananlar

- **Multimodal OCR** — Belge görselleri, klasik OCR yerine multimodal bir LLM'e verilerek yapılandırılmış (JSON) metin ve meta veri çıkarımı yapılıyor.
- **Text Splitter** — Framework kullanmadan yazılmış, paragraf → satır → cümle → kelime hiyerarşisini izleyen token bazlı recursive chunking fonksiyonu.
- **Embedding Üretimi** — Çok dilli (Türkçe destekli) `sentence-transformers` modeliyle chunk'lar vektörleştiriliyor.
- **Vektör Veritabanı** — Embedding'ler FAISS'e (`IndexFlatIP`, kosinüs benzerliği) kaydediliyor; doğal dil sorgularıyla en yakın sonuçları (Top-K) döndüren semantik arama fonksiyonu yazıldı.
- **Doğruluk Testi** — Gerçek OCR çıktıları ve ground-truth veri seti üzerinden uçtan uca arama doğruluğu ölçüldü (bkz. `data/processed/search_accuracy_report.json`).

## Devam eden / planlanan çalışmalar

- NLP tabanlı belge sınıflandırma (fatura/sözleşme/dilekçe vb. otomatik etiketleme)
- `config.yaml` üzerinden tek ayarla cloud/local model geçişi (Factory yapısı)
- Tüm modüllerin tek bir uçtan uca pipeline'da birleştirilmesi
- Basit bir chat/arama arayüzü (Streamlit/Gradio)

## Proje yapısı

```
config/          # Model ve vektör DB ayarları (settings.yaml)
data/
  raw_docs/       # Örnek, gizlilik içermeyen test belgeleri
  processed/      # Üretilen embedding, FAISS index ve test raporları
notebooks/        # Adım adım geliştirme/test defterleri
src/              # text_splitter, embedder, vector_store modülleri
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Gerekli API anahtarlarını `.env` dosyasında tanımlayın, model ve vektör DB ayarlarını `config/settings.yaml` üzerinden yapılandırın.
