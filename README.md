# DocuRAG

Belge görsellerinden (fatura, sözleşme, dilekçe vb.) çıkarılan metinler üzerinde semantik arama yapabilen, çoklu dil modeli (cloud/local) desteğine uygun esnek bir RAG (Retrieval-Augmented Generation) pipeline'ı.

> 🚧 **Durum: Geliştirme aşamasında.** Aşağıdaki modüller tamamlanmış olup proje aktif olarak genişletilmektedir.

## Şu ana kadar tamamlananlar

- **Multimodal OCR** — Belge görselleri, klasik OCR yerine multimodal bir LLM'e verilerek yapılandırılmış (JSON) metin ve meta veri çıkarımı yapılıyor.
- **Text Splitter** — Framework kullanmadan yazılmış, paragraf → satır → cümle → kelime hiyerarşisini izleyen token bazlı recursive chunking fonksiyonu.
- **Embedding Üretimi** — Çok dilli (Türkçe destekli) `sentence-transformers` modeliyle chunk'lar vektörleştiriliyor.
- **Vektör Veritabanı** — Embedding'ler FAISS'e (`IndexFlatIP`, kosinüs benzerliği) kaydediliyor; doğal dil sorgularıyla en yakın sonuçları (Top-K) döndüren semantik arama fonksiyonu yazıldı.
- **Doğruluk Testi** — Gerçek OCR çıktıları ve ground-truth veri seti üzerinden uçtan uca arama doğruluğu ölçüldü (bkz. `data/processed/search_accuracy_report.json`).
- **Otomatik Belge Sınıflandırma** — Belge metni bir LLM'e (Claude) verilerek önceden tanımlanmış sınıflardan (fatura/sözleşme/dilekçe/talep formu vb.) uygun olan birden fazlasına (multi-label) atanıyor ve serbest metinli etiketler çıkarılıyor. Ayrı bir "belirsiz" kategorisi açmak yerine, güven skoru bir eşiğin altında kalan belgeler `human_review: true` olarak işaretleniyor. Eşik değeri (0.7), farklı sınıflardan ve kasıtlı belirsiz içerikten gerçek API sonuçlarıyla kalibre edildi; gerçek çoklu-sınıf (multi-label) çıktısı da canlı API ile doğrulandı (bkz. `src/classifier.py`, `data/processed/classification_report.json`, `data/processed/classification_calibration_report.json`).
- **Sınıflandırma Meta Verisinin RAG'a Entegrasyonu** — `siniflar`/`guven`/`etiketler`/`human_review` alanları her chunk'a eklenerek FAISS metadata'sının bir parçası haline getirildi; arama sonuçları artık kategori/etiket/inceleme durumuna göre filtrelenebiliyor. Aynı test belgeleri ve sorgularla uçtan uca zincirin (OCR → RAG → Sınıflandırma) doğruluğu bozulmadan korunduğu doğrulandı. İnsan incelemesi sonrası kategori/etiket düzeltmeleri, FAISS index'i (vektörleri) yeniden kurmadan sadece metadata dosyası güncellenerek kalıcı hale getirilebiliyor (bkz. `src/vector_store.py`, `notebooks/09_classification_metadata_rag.ipynb`).

## Devam eden / planlanan çalışmalar

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
src/              # text_splitter, embedder, vector_store, classifier modülleri
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Gerekli API anahtarlarını `.env` dosyasında tanımlayın, model ve vektör DB ayarlarını `config/settings.yaml` üzerinden yapılandırın.
