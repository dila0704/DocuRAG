"""
Canli sunum icin KASITLI olarak hasarli/dusuk kaliteli bir "taranmis belge"
gorseli uretir (bkz. README "Sirada Ne Var?" tartismasi sonrasi eklenen UX
onerisi: "canli hata enjeksiyonu" demo senaryosu).

notebooks/_generate_test_docs.py ile AYNI PIL tabanli uretim desenini
kullanir, ama bilerek kotu okunabilirlik uretir: dusuk kontrastli satirlar,
bulanikliktirma (GaussianBlur) ve bir kismini ortan siyah bir "hasar" kutusu.
Amac: bu belgeyi Belge Yukleme sayfasindan elle yukleyip dusuk guven ->
Inceleme Kuyrugu akisini canli gostermek -- OTOMATIK OLARAK INDEKSE
EKLENMEZ, sadece dosyayi uretir.

Tek seferlik yardimci script; notebook degildir (digerleriyle ayni desen).
"""
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_docs")
FONT_PATH = r"C:\Windows\Fonts\consola.ttf"

os.makedirs(RAW_DIR, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, 20)

BG_COLOR = (30, 30, 30)
TEXT_COLOR_NORMAL = (220, 220, 220)
TEXT_COLOR_FAINT = (55, 52, 48)  # arka plana neredeyse gomulu, kasitli dusuk kontrast

WIDTH, HEIGHT = 1000, 420

# (satir, kontrast) -- "faint" olanlar OCR icin neredeyse okunamaz seviyede.
LINES = [
    ("DONANIM TALEP FORMU", "normal"),
    ("", "normal"),
    ("Talep Eden: ..............", "faint"),
    ("Tarih: 2026/0.../..", "faint"),
    ("Departman: Bilgi Islem", "normal"),
    ("Konu: ................", "faint"),
    ("", "normal"),
    ("Aciklama:", "normal"),
    ("... miktar bilgisi okunamiyor ...", "faint"),
    ("talep edilen urun kismi hasarli", "normal"),
    ("", "normal"),
    ("Imza: [lekeli/okunamiyor]", "faint"),
]

img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
draw = ImageDraw.Draw(img)
y = 15
for line, contrast in LINES:
    color = TEXT_COLOR_NORMAL if contrast == "normal" else TEXT_COLOR_FAINT
    draw.text((15, y), line, font=font, fill=color)
    y += 30

# Belgenin bir kismini "fiziksel hasar/leke" gibi tamamen karartilmis bir
# dikdortgenle ortuyoruz -- gercek bir kotu taramada sikca goruleni taklit
# eder (Tesseract/Claude vision hicbir sey okuyamaz bu bolgede).
draw.rectangle([550, 180, 950, 260], fill=(12, 12, 12))

# Hafif bulaniklastirma: dusuk-kontrastli metni daha da okunmaz kilar
# (gercek bir odak dısı/dusuk cozunurluklu tarama hissi verir).
img = img.filter(ImageFilter.GaussianBlur(radius=1.4))

out_path = os.path.join(RAW_DIR, "demo_hasarli_belge.png")
img.save(out_path)
print(f"Olusturuldu (demo icin, INDEKSE EKLENMEDI): {out_path}")
