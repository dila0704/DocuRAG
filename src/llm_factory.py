"""
config/settings.yaml uzerinden secilen LLM saglayicisina (cloud/local) baglanacak
Factory yapisi (DOC-26 Gun 1 tasarim + DOC-27 Gun 2 cloud/local baglanti entegrasyonu).

Amac: settings.yaml'daki `llm_settings.active_mode` (cloud/local) ve ilgili
`provider` (anthropic/openai/huggingface) degerleri degistirilerek, ust
seviye moduller (orn. classifier.py) hicbir kod degisikligi yapmadan farkli
bir LLM'e gecebilsin. Bu modul tek bir ortak arayuz (LLMClient) tanimlar;
classify_document() gibi cagiran kod sadece `generate()` metodunu bilir,
arka planda Anthropic mi OpenAI mi yoksa yerel bir huggingface modeli mi
calistigini bilmesine gerek kalmaz.

Uc saglayici da calisir durumda:
- AnthropicClient / OpenAIClient: ilgili bulut API'sine baglanir.
- LocalHFClient: transformers ile modeli/tokenizer'i indirip (ilk
  calistirmada) yerelde calistirir; embedder.py'deki _model_cache
  deseniyle tutarli sekilde module-seviyesinde onbelleklenir. Gated bir
  model (orn. varsayilan config'teki meta-llama/Meta-Llama-3-8B-Instruct)
  icin HuggingFace erisim onayi ve .env'de HF_TOKEN gerekir.

Loglama (DOC-28): hangi saglayici/model'in secildigi (get_llm_client) ve
her generate() cagrisinin basarili/basarisiz oldugu + suresi standart
`logging` modulu ile loglanir (logger adi: "llm_factory"). Bu modul
`logging.basicConfig()` COAGIRMAZ (kutuphane konvansiyonu); logging'i
gormek isteyen cagiran kod (notebook/uygulama) kendi handler'ini
kurmali -- bkz. notebooks/12_multi_model_e2e_chain_test.ipynb.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm_factory")
logger.addHandler(logging.NullHandler())

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
DEFAULT_USAGE_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "usage_log.jsonl"

# Yaklasik, elle bakimi yapilan liste fiyatlandirmasi (USD / 1M token). SADECE
# maliyet/gecikme panelinde kaba bir tahmin gostermek icindir -- gercek
# faturalandirma icin saglayicinin resmi fiyatlandirma sayfasina bakilmali.
# Tabloda olmayan bir model icin maliyet UYDURULMAZ, _estimate_cost_usd() None
# doner (DOC-30 Oncelik 3, dashboard.py).
_PRICING_TABLE: dict[str, tuple[float, float]] = {
    # model_name -> (giris $/1M token, cikis $/1M token)
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "gpt_4": (2.5, 10.0),
    "gpt-4o": (2.5, 10.0),
}


def _estimate_cost_usd(model_name: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """model_name _PRICING_TABLE'da yoksa ya da token sayilari eksikse None doner
    (sahte/uydurma bir maliyet gosterilmez)."""
    if input_tokens is None or output_tokens is None:
        return None
    pricing = _PRICING_TABLE.get(model_name)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def _append_usage_log(
    provider: str,
    model_name: str,
    usage: dict | None,
    cost_usd: float | None,
    duration_s: float,
    path: Path | None = None,
) -> None:
    """Basarili her generate() cagrisi icin data/processed/usage_log.jsonl'a
    tek satir JSON append eder (maliyet/gecikme panelinin gercek veri kaynagi,
    bkz. app/views/dashboard.py). Loglama basarisiz olursa (disk/izin sorunu)
    ana LLM cagrisini KESMEZ, sadece uyari loglanir.

    `path` verilmezse modul degiskeni DEFAULT_USAGE_LOG_PATH CAGRI ANINDA
    okunur (fonksiyon imzasinda varsayilan olarak BAGLANMAZ) -- boylece
    testler `monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", ...)`
    ile gercek veri dosyasina yazmadan izole calisabilir."""
    path = path or DEFAULT_USAGE_LOG_PATH
    record = {
        "timestamp": time.time(),
        "provider": provider,
        "model_name": model_name,
        "input_tokens": usage.get("input_tokens") if usage else None,
        "output_tokens": usage.get("output_tokens") if usage else None,
        "cost_usd": cost_usd,
        "duration_s": duration_s,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("usage_log.jsonl yazilamadi (loglama atlandi, LLM cagrisi etkilenmez).", exc_info=True)


class LLMClient(ABC):
    """Tum saglayicilarin (cloud/local) uydugu ortak arayuz.

    Yeni bir saglayici eklemek icin bu sinifi implemente edip
    `_PROVIDER_REGISTRY`'e kaydetmek yeterlidir; get_llm_client() ve onu
    cagiran moduller degismez.

    `generate()` somut (concrete) bir metottur: loglama/sure olcumu/hata
    yakalama burada tek bir yerde uygulanir (DOC-28), her saglayici sadece
    `_generate()`'i implemente eder. Boylece uc saglayicida da ayni
    loglama davranisi garanti edilir, kod tekrari olmaz.
    """

    model_name: str

    # generate() basarisiz olursa varsayilan olarak bu kadar EK deneme yapilir
    # (toplam deneme sayisi = 1 + DEFAULT_MAX_RETRIES). Gecici hatalar icin
    # (rate limit, agdaki kisa kesinti, 5xx) dusunulmustur (DOC-31); programatik
    # hatalar (orn. yanlis parametre) da retry edilir cunku _generate()
    # saglayiciya gore hangi exception turunun "gecici" oldugunu ayirt etmek
    # guvenilir degildir -- son deneme de basarisiz olursa orijinal hata
    # oldugu gibi yukari firlatilir.
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_BACKOFF_BASE = 1.0  # saniye; deneme n icin bekleme = base * 2**(n-1)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int | None = None,
        retry_backoff_base: float | None = None,
    ) -> str:
        """system_prompt + user_message verip modelin metin yanitini dondurur.

        Args:
            temperature: 0.0 en deterministik (siniflandirma gibi
                tekrarlanabilirlik gereken gorevler icin varsayilan);
                saglayici destekliyorsa yukari cekilerek daha "yaratici"
                yanitlar alinabilir.
            max_retries: _generate() basarisiz olursa yapilacak ek deneme
                sayisi (varsayilan DEFAULT_MAX_RETRIES).
            retry_backoff_base: denemeler arasi ussel bekleme suresinin
                tabani, saniye (varsayilan DEFAULT_RETRY_BACKOFF_BASE).
        """
        max_retries = self.DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        retry_backoff_base = (
            self.DEFAULT_RETRY_BACKOFF_BASE if retry_backoff_base is None else retry_backoff_base
        )
        provider = type(self).__name__
        total_attempts = max_retries + 1

        for attempt in range(total_attempts):
            t0 = time.time()
            logger.info(
                "generate basladi: provider=%s model=%s max_tokens=%d temperature=%.2f deneme=%d/%d",
                provider, self.model_name, max_tokens, temperature, attempt + 1, total_attempts,
            )
            try:
                text = self._generate(system_prompt, user_message, max_tokens, temperature)
            except Exception:
                logger.exception(
                    "generate basarisiz (deneme %d/%d): provider=%s model=%s sure=%.2fsn",
                    attempt + 1, total_attempts, provider, self.model_name, time.time() - t0,
                )
                if attempt < max_retries:
                    delay = retry_backoff_base * (2 ** attempt)
                    logger.warning("generate %.2fsn sonra yeniden denenecek.", delay)
                    time.sleep(delay)
                    continue
                raise

            duration = time.time() - t0
            logger.info(
                "generate tamamlandi: provider=%s model=%s sure=%.2fsn yanit_uzunlugu=%d",
                provider, self.model_name, duration, len(text),
            )
            usage = getattr(self, "_last_usage", None)
            cost_usd = _estimate_cost_usd(self.model_name, *(
                (usage.get("input_tokens"), usage.get("output_tokens")) if usage else (None, None)
            ))
            _append_usage_log(provider, self.model_name, usage, cost_usd, duration)
            return text

        raise AssertionError("generate: erisilmemesi gereken kod yolu")  # pragma: no cover

    @abstractmethod
    def _generate(self, system_prompt: str, user_message: str, max_tokens: int, temperature: float) -> str:
        """Saglayiciya ozgu asil uretim mantigi. generate() tarafindan
        loglama/sure olcumu/retry sarmalanmis sekilde cagrilir."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Cloud saglayici: Anthropic (Claude). classifier.py'deki mevcut
    _get_client()/messages.create() deseniyle ayni sekilde calisir."""

    def __init__(self, model_name: str, api_key: str | None = None):
        import anthropic

        self.model_name = model_name
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _generate(self, system_prompt: str, user_message: str, max_tokens: int, temperature: float) -> str:
        import anthropic

        request_kwargs = dict(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        try:
            response = self._client.messages.create(temperature=temperature, **request_kwargs)
        except anthropic.BadRequestError as exc:
            # Bazi (daha yeni) modeller "temperature" parametresini artik
            # desteklemiyor ve bunu 400 hatasiyla reddediyor (gercek API ile
            # gozlemlendi: claude-sonnet-5). Bu durumda parametre olmadan
            # bir kez daha denenir; genel retry mantigina (generate())
            # birakilmaz cunku ayni istek her seferinde ayni sekilde
            # basarisiz olurdu.
            if "temperature" in str(exc).lower() and "deprecated" in str(exc).lower():
                logger.info(
                    "AnthropicClient: model=%s 'temperature' parametresini desteklemiyor, parametre olmadan tekrar deneniyor.",
                    self.model_name,
                )
                response = self._client.messages.create(**request_kwargs)
            else:
                raise
        usage_obj = getattr(response, "usage", None)
        self._last_usage = (
            {"input_tokens": getattr(usage_obj, "input_tokens", None), "output_tokens": getattr(usage_obj, "output_tokens", None)}
            if usage_obj is not None else None
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIClient(LLMClient):
    """Cloud saglayici: OpenAI (orn. gpt_4). Ticket ornegindeki `model: gpt_4`
    senaryosunun karsiligi."""

    def __init__(self, model_name: str, api_key: str | None = None):
        import openai

        self.model_name = model_name
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _generate(self, system_prompt: str, user_message: str, max_tokens: int, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        usage_obj = getattr(response, "usage", None)
        self._last_usage = (
            {"input_tokens": getattr(usage_obj, "prompt_tokens", None), "output_tokens": getattr(usage_obj, "completion_tokens", None)}
            if usage_obj is not None else None
        )
        return response.choices[0].message.content or ""


_local_model_cache: dict[str, tuple] = {}


def _get_local_model(model_name: str):
    """transformers model/tokenizer ciftini lazy-load edip module-seviyesinde
    onbellekler (embedder.py'deki _get_model()/_model_cache deseniyle
    tutarli); ayni model_name ile tekrar cagrildiginda diskten/agdan
    yeniden yuklemez.

    HF_TOKEN .env'de tanimliysa gated modeller (orn. meta-llama ailesi)
    icin de kullanilir; tanimsizsa herkese acik modellerle calisir.
    """
    if model_name not in _local_model_cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.getenv("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, dtype="auto", low_cpu_mem_usage=True,
        )
        model.eval()
        _local_model_cache[model_name] = (tokenizer, model)
    return _local_model_cache[model_name]


_FALLBACK_CONTEXT_WINDOW = 4096  # tokenizer/config'ten guvenilir bir deger alinamazsa kullanilan taban.


def _max_prompt_tokens(tokenizer, model, reserved_for_output: int) -> int:
    """Modelin gercek baglam penceresini bulup `reserved_for_output` (max_new_tokens)
    kadarini uretim icin ayirarak, prompt'a ayrilabilecek guvenli token sinirini dondurur.

    `tokenizer.model_max_length`, model config'inde tanimsizsa transformers
    tarafindan cok buyuk bir sentinel degerle (orn. 1e30) doldurulur; bu
    durumda `model.config.max_position_embeddings`'e, o da yoksa sabit bir
    tabana (_FALLBACK_CONTEXT_WINDOW) dusulur.
    """
    context_window = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(context_window, int) or context_window <= 0:
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length <= 1_000_000:
            context_window = model_max_length
        else:
            context_window = _FALLBACK_CONTEXT_WINDOW

    return max(context_window - reserved_for_output, 1)


class LocalHFClient(LLMClient):
    """Local saglayici: huggingface (orn. `local_llama` -> Meta-Llama-3-8B-Instruct).

    Model, tokenizer'in destekledigi chat template'i (varsa) kullanilarak
    system_prompt + user_message'i tek bir prompt'a cevirir, greedy (
    do_sample=False) sekilde uretir ve sadece yeni uretilen kismi
    (girdi promptu haric) metne cevirip dondurur. Boylece AnthropicClient
    ve OpenAIClient ile ayni `generate()` sozlesmesini saglar.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _generate(self, system_prompt: str, user_message: str, max_tokens: int, temperature: float) -> str:
        import torch

        tokenizer, model = _get_local_model(self.model_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system_prompt}\n\n{user_message}\n\nCevap:"

        max_input_tokens = _max_prompt_tokens(tokenizer, model, reserved_for_output=max_tokens)
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        if prompt_tokens > max_input_tokens:
            logger.warning(
                "LocalHFClient: prompt (%d token) baglam penceresini asiyor, "
                "%d token'a kesiliyor (model=%s, max_new_tokens=%d icin ayrilan alan dusuldu).",
                prompt_tokens, max_input_tokens, self.model_name, max_tokens,
            )

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
        ).to(model.device)
        # temperature=0 -> greedy decoding (deterministik, do_sample=False).
        # temperature>0 -> sample'lama acilir; cloud saglayicilarla ayni
        # "temperature" sozlesmesi burada da korunur.
        generation_kwargs = (
            {"do_sample": False}
            if temperature <= 0
            else {"do_sample": True, "temperature": temperature}
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        # Local calistirma icin USD maliyeti anlamsiz (kendi donanimimizda
        # calisiyor) -- yine de token sayilarini loglariz (dashboard'daki
        # gecikme/hacim karsilastirmasi icin), _estimate_cost_usd zaten
        # _PRICING_TABLE'da olmayan modeller icin None doner.
        self._last_usage = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(generated_ids.shape[0]),
        }
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# Saglayici adi (config/settings.yaml -> provider) -> LLMClient implementasyonu.
# Yeni bir saglayici desteklemek icin sadece burasi genisletilir.
_PROVIDER_REGISTRY: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "huggingface": LocalHFClient,
}


def _merge_llm_settings(base: dict, override: dict) -> dict:
    """`override`'daki alanlari `base` uzerine yazar; ic ice sozlukler (orn.
    cloud_model) bir seviye derinlikte birlestirilir (override'da olmayan
    alt-alanlar base'den korunur)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_llm_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """settings.yaml'i okuyup `llm_settings` blogunu dondurur.

    Ayni dizinde bir 'settings.local.yaml' (bkz. save_llm_settings_override,
    app/views/settings.py) varsa, icindeki alanlar base uzerine merge edilir.
    Boylece Ayarlar sayfasindan yapilan degisiklikler devreye girer ama
    settings.yaml'daki yorumlar/dokumantasyon HICBIR ZAMAN otomatik olarak
    ezilip silinmez (override dosyasi tamamen ayri, yorumsuz, otomatik
    uretilen bir dosyadir). Override dosyasi yoksa davranis tamamen aynidir.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_settings = config["llm_settings"]

    override_path = Path(config_path).parent / "settings.local.yaml"
    if override_path.exists():
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        llm_settings = _merge_llm_settings(llm_settings, override)
    return llm_settings


def save_llm_settings_override(updates: dict, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """settings.yaml'in yanina, yorumsuz/otomatik-uretilmis bir
    'settings.local.yaml' yazar -- base settings.yaml'a ASLA dokunulmaz
    (Turkce yorumlar boylece hicbir zaman kaybolmaz). `updates` ornegin
    {"active_mode": "local", "local_model": {"provider": "huggingface",
    "model_name": "..."}} seklinde olabilir.

    Uygulamanin yeniden baslatilmasi GEREKMEZ: get_llm_client() her cagrida
    load_llm_config() ile config'i taze okur (bkz. classifier.classify_document,
    answer.generate_grounded_answer).
    """
    override_path = Path(config_path).parent / "settings.local.yaml"
    with open(override_path, "w", encoding="utf-8") as f:
        f.write("# Bu dosya Ayarlar sayfasindan otomatik uretildi, elle duzenlemeyin.\n")
        f.write("# settings.yaml'in ustune SADECE burada tanimli alanlari gecici olarak ezer.\n")
        f.write("# Varsayilana donmek icin bu dosyayi silin.\n")
        yaml.safe_dump(updates, f, allow_unicode=True, sort_keys=False)


def get_llm_client(llm_settings: dict | None = None) -> LLMClient:
    """`llm_settings.active_mode`'a (cloud/local) gore uygun LLMClient'i
    insa edip dondurur -- Factory'nin kendisi budur.

    settings.yaml ornegi:
        llm_settings:
          active_mode: "cloud"          # veya "local"
          cloud_model:
            provider: "anthropic"        # veya "openai" (orn. gpt_4)
            model_name: "claude-sonnet-5"
          local_model:
            provider: "huggingface"      # orn. local_llama
            model_name: "meta-llama/Meta-Llama-3-8B-Instruct"

    Args:
        llm_settings: load_llm_config() ciktisiyla ayni sekilde
            {"active_mode": ..., "cloud_model": {...}, "local_model": {...}}
            seklinde bir sozluk. None ise config/settings.yaml'dan okunur.

    Returns:
        Secilen saglayiciya gore AnthropicClient/OpenAIClient/LocalHFClient
        ornegi (hepsi LLMClient arayuzunu uygular).

    Raises:
        ValueError: active_mode icin ilgili blok (cloud_model/local_model)
            tanimli degilse veya provider _PROVIDER_REGISTRY'de yoksa.
    """
    llm_settings = llm_settings if llm_settings is not None else load_llm_config()

    active_mode = llm_settings.get("active_mode", "cloud")
    mode_key = "cloud_model" if active_mode == "cloud" else "local_model"
    model_config = llm_settings.get(mode_key)
    if not model_config:
        raise ValueError(
            f"config/settings.yaml icinde active_mode='{active_mode}' icin "
            f"'{mode_key}' tanimli degil."
        )

    provider = model_config.get("provider")
    client_cls = _PROVIDER_REGISTRY.get(provider)
    if client_cls is None:
        raise ValueError(
            f"Bilinmeyen LLM saglayicisi: '{provider}'. "
            f"Desteklenenler: {list(_PROVIDER_REGISTRY)}"
        )

    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError(f"config/settings.yaml icinde '{mode_key}.model_name' tanimli degil.")

    logger.info(
        "get_llm_client: active_mode=%s provider=%s model_name=%s -> %s",
        active_mode, provider, model_name, client_cls.__name__,
    )
    return client_cls(model_name)
