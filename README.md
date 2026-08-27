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
- **LLM Factory ile Çoklu Model (Cloud/Local) Yönetimi** — `config/settings.yaml`'daki `llm_settings.active_mode` (cloud/local) ve `provider` (anthropic/openai/huggingface) değerlerine göre doğru istemciyi kuran ortak bir `LLMClient` arayüzü ve `get_llm_client()` Factory fonksiyonu yazıldı (Gün 1 tasarım, bkz. `notebooks/10_llm_factory_test.ipynb`). Cloud (Anthropic/OpenAI, örn. `gpt_4`) ve local (huggingface, örn. `local_llama`) bağlantılarının ikisi de gerçek çalışır durumda; local tarafı `transformers` ile modeli/tokenizer'ı yükleyip inference çalıştırıyor, `embedder.py`'deki model önbellekleme deseniyle tutarlı şekilde tekrar tekrar yeniden yüklenmiyor. `src/classifier.py`, doğrudan Anthropic SDK'sını çağırmak yerine bu Factory'yi kullanacak şekilde entegre edildi; böylece `config/settings.yaml`'da tek bir değer değiştirilerek `classifier.py`'nin kodu hiç dokunulmadan cloud/local arasında geçiş yapılabiliyor (bkz. `src/llm_factory.py`, `notebooks/11_llm_factory_cloud_local_integration_test.ipynb`).
- **Model Geçişlerinin Loglanması ve Uçtan Uca Test Edilmesi** — Her `LLMClient.generate()` çağrısı (hangi sağlayıcı/model, süre, başarı/hata) standart `logging` modülüyle (`llm_factory` logger'ı) loglanıyor; hata durumunda da traceback ile birlikte kaydediliyor. OCR → RAG → Sınıflandırma zinciri, gerçek OCR verisiyle hem cloud (Anthropic) hem local (huggingface) konfigürasyonuyla uçtan uca çalıştırılıp sonuçlar `data/processed/multi_model_chain_report.json`'a kaydedildi: RAG tarafı (embedding/arama) her iki konfigürasyonda da sorunsuz çalıştı; local tarafta küçük bir modelle (`Qwen2.5-0.5B-Instruct`) JSON formatı korunsa da sınıflandırma doğruluğunun cloud'a göre belirgin şekilde düştüğü gözlemlendi (bkz. `notebooks/12_multi_model_e2e_chain_test.ipynb`).
- **Uçtan Uca Pipeline ve OCR'ın `src/`'e Taşınması** — Notebook 01'deki doğrudan Anthropic vision çağrısı `src/ocr.py`'ye taşındı (retry/backoff ve loglama eklendi); `src/pipeline.py`, OCR → chunking → sınıflandırma → embedding → FAISS indeksleme zincirini `python src/pipeline.py ingest <gorsel>` / `search "<sorgu>"` ile tek komutla çalıştırılabilir hale getiriyor (bkz. DOC-31).
- **Hata Toleransı (Retry) ve Determinizm** — `LLMClient.generate()` gecici hatalarda (ağ, rate limit) üstel geri çekilmeyle otomatik yeniden dener; `classifier.classify_document()` LLM geçersiz JSON dönerse modelden düzeltmesini isteyerek yeniden dener ve varsayılan olarak `temperature=0.0` ile deterministik çalışır (aynı belge tekrar çalıştırıldığında aynı sonucu üretir).
- **FAISS CRUD** — `vector_store.py`, `IndexIDMap` kullanacak şekilde yeniden yazıldı: `add_chunks()` var olan index'i sıfırdan kurmadan genişletir, `delete_by_source_doc()` bir belgeye ait vektörleri index'i yeniden kurmadan siler. Eski (düz `IndexFlatIP` + liste metadata) formatında kaydedilmiş index'ler `load_index()` tarafından otomatik olarak bu id-tabanlı formata göçürülür.
- **Otomatik Test Paketi (pytest) ve CI** — `tests/` altında `text_splitter`, `embedder`, `vector_store`, `classifier`, `llm_factory`, `ocr`, `answer` ve `components` için sahte (fake/mock) istemcilerle çalışan, gerçek API/ağ çağrısı yapmayan 91 test yazıldı; `.github/workflows/tests.yml` her push/PR'da bunları otomatik çalıştırıyor (bkz. `requirements-dev.txt`, `pytest.ini`).
- **Streamlit Arayüzü** — Onaylanan tasarıma (krem/mürekkep/petrol yeşili/bronz "sessiz lüks" paleti, "Onay Damgası" güven rozeti) sadık, 4 sayfalı (Arama, Belge Yükleme, İnceleme Kuyruğu, Dashboard) çalışan bir Streamlit uygulaması (`app/`) yazıldı. Tüm sayfalar gerçek veriyle çalışır — mockup'taki hiçbir sahte sayı yok; Dashboard'da henüz izlenmeyen metrikler (haftalık hacim, model maliyeti/gecikme geçmişi) bilerek eklenmedi, bunlar için zaman damgası/maliyet kaydı altyapısı (bkz. aşağıdaki DOC-30 Öncelik 2/3) gerekiyor.
- **Kaynak Gösterimli Cevaplama (Grounded Answering)** — `src/answer.py`, arama sonuçlarından `[1]`/`[2]` gibi tıklanabilir kaynak numaralarıyla işaretlenmiş bir AI özeti üretiyor. Grounding (kaynaklanma) zorunluluğu prompt'a değil backend'e dayanıyor: modelin döndürdüğü her cümle, geçerli bir kaynak indeksine sahip olduğu koddan doğrulanmadan kabul edilmiyor (bkz. `_enforce_grounding`). Arama sayfasında bir kaynağa tıklamak ilgili sonuç kartını kalıcı olarak vurguluyor (sorgu `st.query_params` ile URL'e senkron tutulduğu için tıklamanın tetiklediği tam sayfa yenilemesinde arama durumu korunuyor). Gerçek API ile uçtan uca doğrulandı.

## Devam eden / planlanan çalışmalar

- DOC-30 Öncelik 2-6: Denetim izi (audit trail), aktif öğrenme döngüsü, çoklu model/maliyet karşılaştırma paneli, belge ilişki grafiği, rol bazlı erişim — her biri ayrı bir oturumda ele alınacak.
- Sınıflandırma kalibrasyonunun daha büyük, gerçek API sonuçlarıyla doğrulanmış bir örneklemle genişletilmesi (şu an 9 belgelik küçük bir kalibrasyon örneklemi var).
- (Risk notu) Local mod üretimde kullanılacaksa küçük modeller yerine daha yetkin, instruction-tuned bir modelin (örn. config'teki Llama-3-8B-Instruct) `HF_TOKEN` ile gerçekten doğrulanması gerekiyor.
- (Bilinen mimari sınır) `src/ocr.py`, `llm_factory`'nin cloud/local soyutlamasını kullanmaz — görüntü girdisini yalnızca doğrudan Anthropic (cloud) destekler; local bir görsel-dil modeliyle OCR için `LLMClient` arayüzünün görüntü girdisini de desteklemesi gerekir.
- (Gerçek API ile bulunan bir kısıtlama) `claude-sonnet-5` modeli artık `temperature` parametresini kabul etmiyor ("deprecated" hatası); `AnthropicClient` bu hatayı yakalayıp parametre olmadan otomatik yeniden deniyor (bkz. `src/llm_factory.py`), ancak bu, bu model için `temperature=0.0` ile hedeflenen tam determinizmin artık garanti edilemediği anlamına geliyor.

## Proje yapısı

```
app/              # Streamlit uygulaması (app.py, views/, styles.py, components.py)
config/          # Model ve vektör DB ayarları (settings.yaml)
data/
  raw_docs/       # Örnek, gizlilik içermeyen test belgeleri
  processed/      # Üretilen embedding, FAISS index ve test raporları
notebooks/        # Adım adım geliştirme/test defterleri
src/              # ocr, text_splitter, embedder, vector_store, classifier, llm_factory, answer, pipeline modülleri
tests/            # pytest test paketi (sahte istemcilerle, ağ/API çağrısı yapmadan)
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Gerekli API anahtarlarını `.env` dosyasında tanımlayın (`ANTHROPIC_API_KEY`, `provider: openai` kullanılacaksa `OPENAI_API_KEY`; gated bir huggingface modeli — örn. Llama ailesi — local olarak kullanılacaksa `HF_TOKEN`), model ve vektör DB ayarlarını `config/settings.yaml` üzerinden yapılandırın.

## Uçtan uca çalıştırma

```bash
python src/pipeline.py ingest data/raw_docs/test_talep_01.png
python src/pipeline.py search "laptop talebi"
```

## Streamlit uygulamasını çalıştırma

```bash
streamlit run app/app.py
```

Arama, Belge Yükleme, İnceleme Kuyruğu ve Dashboard sayfalarına sol menüden erişilir.

## Testler

```bash
pip install -r requirements-dev.txt
pytest
```
