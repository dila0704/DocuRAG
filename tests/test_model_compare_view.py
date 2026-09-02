"""app/views/model_compare.py'nin hatasiz render edildigini dogrular.

streamlit.testing.v1.AppTest, sayfayi kendi calistirma baglaminda import
ediyor -- conftest.py'nin sys.path.insert() ile yaptigi ekleme bu baglama
sizmiyor, bu yuzden PYTHONPATH ortam degiskeni ile src/ ve app/ ayrica
eklenir (bkz. asagidaki fixture)."""
from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

VIEW_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "views", "model_compare.py")


@pytest.fixture(autouse=True)
def _pythonpath_for_apptest(monkeypatch):
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src_dir, app_dir, existing])))


def test_model_compare_view_renders_without_index(monkeypatch):
    """Henuz hicbir belge indekslenmemisken (FileNotFoundError yakalanmali)
    sayfa hata firlatmadan bos-durum halini gostermeli."""
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), ".."))
    at = AppTest.from_file(VIEW_PATH)
    at.run(timeout=30)
    assert not at.exception
    assert len(at.markdown) >= 1
