"""
src/ altindaki moduller notebook'larla tutarli olacak sekilde "duz" (flat)
import ile yazildi (orn. `import text_splitter`, `from llm_factory import ...`),
paket (src.text_splitter) olarak degil. Testlerin de ayni sekilde import
edebilmesi icin src/ dizinini sys.path'e ekliyoruz.
"""
import os
import sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import pytest  # noqa: E402
from llm_factory import LLMClient  # noqa: E402


class FakeLLMClient(LLMClient):
    """Sirali sahte yanitlar donduren, cagrilari kaydeden test istemcisi.

    test_classifier.py / test_answer.py / test_query_rewriter.py gibi birden
    fazla test modulunde birebir ayni sekilde tanimlaniyordu; tekrari onlemek
    icin tek bir yerde tutulur.
    """

    def __init__(self, responses, model_name: str = "fake"):
        self.model_name = model_name
        self._responses = list(responses)
        self.calls: list[dict] = []

    def _generate(self, system_prompt, user_message, max_tokens, temperature):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_client():
    """FakeLLMClient sinifinin kendisini dondurur; testler `fake_llm_client([...])`
    seklinde cagirip bir istemci ornegi olusturur (dogrudan sinif importuyla ayni
    kullanim, ama fixture uzerinden - monkeypatch/diger fixture'larla tutarli)."""
    return FakeLLMClient


@pytest.fixture(autouse=True)
def _isolate_usage_log(tmp_path, monkeypatch):
    """LLMClient.generate() basarili her cagridan sonra data/processed/usage_log.jsonl'a
    yazar (bkz. llm_factory._append_usage_log). Testler (FakeLLMClient dahil,
    cunku generate() concrete metod ve override edilmiyor) bu gercek veri
    dosyasini KIRLETMESIN diye her test icin gecici bir dosyaya yonlendirilir."""
    import llm_factory

    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
