"""
DOC-34: kalibrasyon/degerlendirme orneklemi genisletme turunda uretilen 10
yeni belge gorseli (data/raw_docs/eval_*.png). Amac: search accuracy
degerlendirme setini (bkz. notebooks/07 + notebooks/14) 5 belgelik/tek
kategorili (yalnizca talep formu) bir orneklemden, 5 KATEGORIYE (fatura,
sozlesme, dilekce, talep formu) yayilan 15 belgelik daha gercekci bir
orneklemine cikarmak -- kucuk/homojen bir eval setinin Hit@1/MRR gibi
metrikleri olduğundan iyi gostermesi riskine karsi (bkz. notebooks/14
sonuc bolumu).

notebooks/_generate_test_docs.py ile AYNI PIL tabanli uretim desenini
kullanir. Tek seferlik yardimci script; notebook degildir.
"""
import os

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_docs")
FONT_PATH = r"C:\Windows\Fonts\consola.ttf"

os.makedirs(RAW_DIR, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, 20)
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)


def render_form(filename, lines, width=1000):
    height = max(200, 30 * len(lines) + 30)
    img = Image.new("RGB", (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    y = 15
    for line in lines:
        draw.text((15, y), line, font=font, fill=TEXT_COLOR)
        y += 30
    path = os.path.join(RAW_DIR, filename)
    img.save(path)
    print(f"Olusturuldu: {path}")


DOCS = {
    "eval_fatura_ofis.png": [
        "FATURA", "",
        "Fatura No: FTR-2026-02001", "Tarih: 05.08.2026",
        "Satici: Kirtasiye Dunyasi Ltd.", "Alici: Docurag Yazilim Ltd. Sti.", "",
        "Urun: A4 Kagit, Toner Kartusu, Dosya Klasoru",
        "Toplam Tutar: 2.350 TL",
    ],
    "eval_fatura_yazilim.png": [
        "FATURA", "",
        "Fatura No: FTR-2026-02002", "Tarih: 07.08.2026",
        "Satici: CloudSoft Yazilim A.S.", "Alici: Docurag Yazilim Ltd. Sti.", "",
        "Urun/Hizmet: Yillik Bulut Depolama Lisansi",
        "Toplam Tutar: 18.000 TL",
    ],
    "eval_fatura_kargo.png": [
        "FATURA", "",
        "Fatura No: FTR-2026-02003", "Tarih: 09.08.2026",
        "Satici: HizliKargo Lojistik", "Alici: Docurag Yazilim Ltd. Sti.", "",
        "Hizmet: Yurt Ici Nakliye (50 Koli)",
        "Toplam Tutar: 4.750 TL",
    ],
    "eval_sozlesme_danismanlik.png": [
        "DANISMANLIK SOZLESMESI", "",
        "Taraflar: Docurag Yazilim Ltd. Sti. - Pinar Aydogan",
        "Tarih: 01.08.2026",
        "Konu: Veri gizliligi ve KVKK uyum danismanligi",
        "Sure: 6 ay", "Ucret: Aylik 15.000 TL",
    ],
    "eval_sozlesme_kira.png": [
        "KIRA SOZLESMESI", "",
        "Taraflar: Mulk Sahibi Kaan Ergin - Kiraci Docurag Yazilim Ltd. Sti.",
        "Tarih: 15.07.2026",
        "Konu: Ofis Kirasi (Kadikoy, 120 m2)",
        "Sure: 24 ay", "Aylik Kira: 45.000 TL",
    ],
    "eval_sozlesme_gizlilik.png": [
        "GIZLILIK SOZLESMESI (NDA)", "",
        "Taraflar: Docurag Yazilim Ltd. Sti. - BetaTest A.S.",
        "Tarih: 03.08.2026",
        "Konu: Beta test surecinde paylasilan bilgilerin gizliligi",
        "Sure: 3 yil",
    ],
    "eval_dilekce_izin.png": [
        "DILEKCE", "",
        "Muhatap: Insan Kaynaklari Mudurlugu",
        "Tarafimca 05.09.2026-09.09.2026 tarihleri arasinda yillik izin",
        "kullanilmak istenmektedir. Geregini rica ederim.", "",
        "Ad Soyad: Burak Sen", "Tarih: 20.08.2026",
    ],
    "eval_dilekce_sikayet.png": [
        "DILEKCE", "",
        "Muhatap: Genel Mudurluk",
        "Ofis klima sisteminin uzun suredir arizali olmasi nedeniyle",
        "calisma kosullarinin olumsuz etkilendigini bildiririm.",
        "Geregini rica ederim.", "",
        "Ad Soyad: Selin Kara", "Tarih: 18.08.2026",
    ],
    "eval_talep_ofis_malzeme.png": [
        "OFIS MALZEMESI TALEP FORMU", "",
        "Talep Eden: Kerem Aksoy", "Tarih: 11.08.2026",
        "Departman: Finans", "Konu: Yazici Toneri Talebi", "",
        "Aciklama:",
        "Departmanimizdaki yazicinin toneri bitmek uzere, yeni toner",
        "temin edilmesini rica ederim.",
    ],
    "eval_talep_uzaktan_calisma.png": [
        "UZAKTAN CALISMA TALEP FORMU", "",
        "Talep Eden: Gizem Yildirim", "Tarih: 19.08.2026",
        "Departman: Pazarlama", "Konu: Uzaktan Calisma Talebi", "",
        "Aciklama:",
        "Saglik nedeniyle onumuzdeki 2 hafta boyunca evden calismak",
        "istiyorum, onayinizi rica ederim.",
    ],
}

for filename, lines in DOCS.items():
    render_form(filename, lines)

print(f"\n{len(DOCS)} yeni degerlendirme belgesi uretildi.")
