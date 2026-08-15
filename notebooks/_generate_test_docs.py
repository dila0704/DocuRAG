"""
Gun 3 - Test belgesi uretici.
data/raw_docs altina cesitli talep formu gorselleri uretir ve
data/processed/ground_truth.json icine beklenen (dogru) alanlari yazar.
Tek seferlik yardimci script; notebook degildir.
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_docs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FONT_PATH = r"C:\Windows\Fonts\consola.ttf"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, 20)

BG_COLOR = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)


def render_form(filename, lines, width=1000, height=420):
    img = Image.new("RGB", (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    y = 15
    for line in lines:
        draw.text((15, y), line, font=font, fill=TEXT_COLOR)
        y += 30
    path = os.path.join(RAW_DIR, filename)
    img.save(path)
    print(f"Olusturuldu: {path}")


# ------------------------------------------------------------------
# Belge 2: Normal / eksiksiz durum, farkli departman ve konu
# ------------------------------------------------------------------
render_form(
    "test_talep_02.png",
    [
        "DONANIM TALEP FORMU",
        "",
        "Talep Eden: Ahmet Yilmaz",
        "Tarih: 13.08.2026",
        "Departman: Insan Kaynaklari",
        "Konu: Yeni Laptop Talebi",
        "",
        "Aciklama:",
        "Mevcut laptopumun performans sorunlari nedeniyle is verimliligimi",
        "etkilemektedir. Tarafima yeni bir laptop tahsis edilmesini rica ederim.",
        "",
        "Imza:",
    ],
)
GT_02 = {
    "talep_eden": "Ahmet Yilmaz",
    "tarih": "2026-08-13",
    "departman": "Insan Kaynaklari",
    "konu": "Yeni Laptop Talebi",
    "aciklama": (
        "Mevcut laptopumun performans sorunlari nedeniyle is verimliligimi "
        "etkilemektedir. Tarafima yeni bir laptop tahsis edilmesini rica ederim."
    ),
}

# ------------------------------------------------------------------
# Belge 3: Eksik alan durumu - Departman satiri tamamen yok
# ------------------------------------------------------------------
render_form(
    "test_talep_03.png",
    [
        "DONANIM TALEP FORMU",
        "",
        "Talep Eden: Zeynep Kaya",
        "Tarih: 14.08.2026",
        "Konu: Klavye Degisimi",
        "",
        "Aciklama:",
        "Klavyemdeki birkac tus arizali. Yeni bir klavye talep ediyorum.",
        "",
        "Imza:",
    ],
)
GT_03 = {
    "talep_eden": "Zeynep Kaya",
    "tarih": "2026-08-14",
    "departman": None,
    "konu": "Klavye Degisimi",
    "aciklama": "Klavyemdeki birkac tus arizali. Yeni bir klavye talep ediyorum.",
}

# ------------------------------------------------------------------
# Belge 4: Farkli tarih formati (gun ay yil - Turkce ay adi)
# ------------------------------------------------------------------
render_form(
    "test_talep_04.png",
    [
        "DONANIM TALEP FORMU",
        "",
        "Talep Eden: Mehmet Demir",
        "Tarih: 15 Agustos 2026",
        "Departman: Muhasebe",
        "Konu: Yazici Talebi",
        "",
        "Aciklama:",
        "Departmanimizda ortak kullanilan yazici arizalandi. Yeni bir yazici",
        "temin edilmesini rica ederim.",
        "",
        "Imza:",
    ],
)
GT_04 = {
    "talep_eden": "Mehmet Demir",
    "tarih": "2026-08-15",
    "departman": "Muhasebe",
    "konu": "Yazici Talebi",
    "aciklama": (
        "Departmanimizda ortak kullanilan yazici arizalandi. Yeni bir yazici "
        "temin edilmesini rica ederim."
    ),
}

# ------------------------------------------------------------------
# Belge 5: Farkli tarih formati (YYYY/MM/DD) + kisa aciklama
# ------------------------------------------------------------------
render_form(
    "test_talep_05.png",
    [
        "DONANIM TALEP FORMU",
        "",
        "Talep Eden: Elif Sahin",
        "Tarih: 2026/08/16",
        "Departman: Pazarlama",
        "Konu: Ek Ekran Talebi",
        "",
        "Aciklama:",
        "Tasarim islerinde verimliligi artirmak icin ikinci bir ekrana",
        "ihtiyacim var.",
        "",
        "Imza:",
    ],
)
GT_05 = {
    "talep_eden": "Elif Sahin",
    "tarih": "2026-08-16",
    "departman": "Pazarlama",
    "konu": "Ek Ekran Talebi",
    "aciklama": "Tasarim islerinde verimliligi artirmak icin ikinci bir ekrana ihtiyacim var.",
}

# ------------------------------------------------------------------
# Belge 1 icin ground truth (Gun 1/2'de kullanilan mevcut belge)
# ------------------------------------------------------------------
GT_01 = {
    "talep_eden": "Dila Alpay",
    "tarih": "2026-08-12",
    "departman": "Yazılım Geliştirme",
    "konu": "Ek Monitör Talebi",
    "aciklama": (
        "Geliştirmekte olduğum yapay zeka destekli doküman analiz modülünün "
        "test işlemleri sırasında, log takibi ve kod yazımını eşzamanlı "
        "yürütebilmek amacıyla tarafıma 1 adet 27 inç monitör tahsis "
        "edilmesini rica ederim."
    ),
}

ground_truth = {
    "test_talep_01.png": GT_01,
    "test_talep_02.png": GT_02,
    "test_talep_03.png": GT_03,
    "test_talep_04.png": GT_04,
    "test_talep_05.png": GT_05,
}

gt_path = os.path.join(PROCESSED_DIR, "ground_truth.json")
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)
print(f"\nGround truth yazildi: {gt_path}")
