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
