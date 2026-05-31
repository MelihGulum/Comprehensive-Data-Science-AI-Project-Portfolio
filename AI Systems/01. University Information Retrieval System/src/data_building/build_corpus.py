import os
import re
import json
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import pdfplumber

try:
    from zemberek import TurkishMorphology
except ImportError:
    TurkishMorphology = None
    logging.warning("⚠️ Zemberek not installed. Run `pip install zemberek-python` for stemming.")


# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_PDF_DIR = Path("data/raw/pdfs")
RAW_TEXT_DIR = Path("data/raw/text")
PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_FILE = INTERIM_DIR / "corpus.json"
METADATA_FILE = INTERIM_DIR / "metadata.csv"
CORPUS_PREPROCESSED_FILE = PROCESSED_DIR / "corpus_preprocessed.json"

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------
# STOPWORDS
# -----------------------------
TURKISH_STOPWORDS = set([
    "ve", "ile", "ama", "ancak", "fakat", "lakin", "veya", "ya", "yahut", "çünkü", "ise", "ki", "dahi", "için", "olarak",
    "bu", "şu", "o", "bunlar", "şunlar", "onlar", "her", "bazı", "tüm", "hiç", "kendi",
    "mi", "mı", "mu", "mü",
    "çok", "daha", "en", "az", "son", "önce", "sonra", "kadar", "gibi",
    "fıkra", "bent", "sayılı", "tarihli", "uyarınca", "gereğince", "kapsamında",
    "ilişkin", "dair", "hakkında", "yapılır", "edilir", "olur", "olduğu", "amacıyla", "şekilde", "hususunda",
    "etmek", "olmak", "yapmak", "bulunmak", "göstermek", "sunmak", "belirlemek",
])

# -----------------------------
# REGULATION CATEGORIES (Kod2’den entegre)
# -----------------------------
REGULATION_CATEGORIES: Dict[str, Dict[str, str]] = {
    "2025-2026 Eğitim Öğretim Yılı Akademik Takvimi (Tıp ve Hukuk Fakülteleri Hariç Tüm Birimler)":
        {"category": "Akademik Takvim", "faculty": "Tıp ve Hukuk Hariç Tüm Fakülteler", "degree_level": "Tümü"},
    "2025-2026 Eğitim Öğretim Yılı Uluslararası Öğrenci Harçları":
        {"category": "Öğrenci Harçları", "faculty": "Tümü", "degree_level": "Tümü"},
    "2025-2026 Eğitim Öğretim Yılı Öğrenci Katkı Payı-Öğrenim Ücret Tablosu":
        {"category": "Öğrenim Ücretleri", "faculty": "Tümü", "degree_level": "Tümü"},
    "2025-2026 Eğitim-Öğretim Yılı Bahar YY_ Önemli Tarihler (1)":
        {"category": "Önemli Tarihler", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-008": {"category": "AKTS Uygulama Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-042": {"category": "Öğrenci İşleri Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-043": {"category": "Öğrenci Konseyi Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-047": {"category": "Muafiyet ve İntibak İşlemleri", "faculty": "Tümü", "degree_level": "Önlisans/Lisans"},
    "EYS-YNG-066": {"category": "Uzaktan Eğitim Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-068": {"category": "Hazırlık ve Yabancı Dille Öğretim", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-071": {"category": "Yaz Okulu Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-072": {"category": "Yemek Bursu Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-084": {"category": "İşletmede Mesleki Eğitim Yönergesi", "faculty": "Tümü", "degree_level": "Lisans"},
    "EYS-YNG-085": {"category": "Çift Anadal Programı Yönergesi", "faculty": "Tümü", "degree_level": "Lisans"},
    "EYS-YNG-087": {"category": "Yandal Programı Yönergesi", "faculty": "Tümü", "degree_level": "Lisans"},
    "EYS-YNG-094": {"category": "Eğitim-Öğretim Komisyonu Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-098": {"category": "Ödül ve Teşvik Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-100": {"category": "Özel Gereksinimli Öğrenciler Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-122": {"category": "Telafi Dersi Uygulama Yönergesi", "faculty": "Tümü", "degree_level": "Lisans"},
    "EYS-YNG-135": {"category": "Diploma ve Diğer Belgeler", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-136": {"category": "Yurt Dışından Öğrenci Kabul Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-137": {"category": "Erasmus+ Yönergesi", "faculty": "Tümü", "degree_level": "Tümü"},
    "EYS-YNG-139": {"category": "Yatay Geçiş Yönergesi", "faculty": "Tümü", "degree_level": "Önlisans/Lisans"},
    "EYS-YNT-005": {"category": "Eğitim-Öğretim ve Sınav Yönetmeliği", "faculty": "Veteriner Fakültesi", "degree_level": "Lisans"},
    "EYS-YNT-008": {"category": "Eğitim-Öğretim Yönetmeliği", "faculty": "Tıp Fakültesi", "degree_level": "Lisans"},
    "EYS-YNT-012": {"category": "Temel Akademik Yönetmelik", "faculty": "Tümü", "degree_level": "Önlisans/Lisans"},
    "EYS-YNT-013": {"category": "Temel Akademik Yönetmelik", "faculty": "Tümü", "degree_level": "Lisansüstü"},
    "EYS-YNT-018": {"category": "Eğitim-Öğretim Yönetmeliği", "faculty": "Diş Hekimliği Fakültesi", "degree_level": "Lisans"},
    "EYS-YNT-019": {"category": "Eğitim-Öğretim ve Sınav Yönetmeliği", "faculty": "Hukuk Fakültesi", "degree_level": "Lisans"},
    "Hukuk 2025-2026 Akademik Takvimi (1)": {"category": "Akademik Takvim", "faculty": "Hukuk Fakültesi", "degree_level": "Tümü"},
    "Sağlık Bilimleri Enstitüsü Lisansüstü Eğitim Öğretim Süreç Takvimi(Güz-Bahar Dönemi) (1)":
        {"category": "Akademik Takvim", "faculty": "Sağlık Bilimleri Enstitüsü", "degree_level": "Lisansüstü"},
    "Tıp Fakültesi Akademik Takvim": {"category": "Akademik Takvim", "faculty": "Tıp Fakültesi", "degree_level": "Lisans"},
    "YÜKSEKÖĞRETİM KURUMLARI ÖĞRENCİ DİSİPLİN YÖNETMELİĞİ":
        {"category": "Öğrenci Disiplin Yönetmeliği", "faculty": "Tümü", "degree_level": "Tümü"},
    "Çift Anadal ve Yandal Programı Takvimi": {"category": "Çift Anadal / Yandal Takvimi", "faculty": "Tümü", "degree_level": "Lisans"},
}
# -----------------------------
# Header & Intro Cleanup (Kod2’den)
# -----------------------------
HEADER_KEYWORDS = [
    r"Doküman No:", r"Hazırlama Tarihi:", r"TNKÜ", r"ÖĞRENCİ İŞLERİ DAİRE",
    r"BAŞKANLIĞI ÇALIŞMA", r"USUL VE ESASLARI", r"YÖNERGESİ",
    r"Revizyon Tarihi:", r"Revizyon No:", r"Toplam Sayfa Sayısı",
    r"Resmi Gazete Tarihi:"
]

REDUNDANT_TITLES = ["Amaç", "Kapsam", "Dayanak", "Tanımlar", "Tanımlar:",
                    "Amaç, Kapsam, Dayanak ve Tanımlar",
                    "Amaç-Kapsam-Dayanak-Tanımlar",
                    "Amaç ve kapsam", "Amaç ve kapsam:",
                    "Diğer Hükümler", "Çeşitli ve Son Hükümler"]

# -----------------------------
# PHRASE NORMALIZATION (Sparse için)
# -----------------------------
def normalize_phrases_sparse(text: str) -> str:
    phrase_map = {
        r'genel\s+not\s+ortalamas[ıi]\w*': 'gno',
        r'azami\s+öğrenim\s+süres[ıi]\w*': 'azami_sure',
        r'yatay\s+geçiş': 'yatay_gecis',
        r'dikey\s+geçiş': 'dikey_gecis',
        r'kayıt\s+silme': 'kayit_silme',
        r'(tekirdağ\s+namık\s+kemal\s+üniversite\w*|namık\s+kemal\s+üniversite\w*|\bnkü\b|\btnkü\b)':
            'tnku tekirdag_namik_kemal_universitesi',
    }

    for pattern, replacement in phrase_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.UNICODE)

    return text


def clean_headers(text: str) -> str:
    for kw in HEADER_KEYWORDS:
        text = re.sub(rf"^{kw}.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return re.sub(r"\n\s*\n", "\n", text).strip()


def remove_redundant_intro_titles(text: str) -> str:
    pattern = r"^\s*(" + "|".join(REDUNDANT_TITLES) + r")\s*$"
    text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return re.sub(r"\n\s*\n", "\n", text).strip()


def remove_footnotes(text: str) -> str:
    # Mevcut temizlik
    text = re.sub(r'(?im)^\s*(Web|Email|E-?mail|Telefon|Faks)\s*:.*$', '', text)
    # Üst simgeler ve dipnot numaraları
    text = re.sub(r'\d+\s*[\^*]', '', text)
    text = re.sub(r'[\u00B9\u00B2\u00B3\u2070-\u2079]', '', text)  # üst simge unicode
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()


# -----------------------------
# PDF to TEXT
# -----------------------------
def pdf_to_text(pdf_path: Path, save_raw_text: bool = True) -> str:
    logging.info(f"Processing PDF-to-Text: {pdf_path.name}")
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_pages.append(page.extract_text() or "")
    full_text = "\n".join(text_pages)
    if save_raw_text:
        txt_file = RAW_TEXT_DIR / f"{pdf_path.stem}.txt"
        txt_file.write_text(full_text, encoding="utf-8")
    return full_text


# -----------------------------
# Turkish Stemmer
# -----------------------------
morphology = TurkishMorphology.create_with_defaults() if TurkishMorphology else None


def turkish_stem(text: str) -> str:
    text = text.lower()
    # Harf veya sayı bloklarını yakala
    tokens = re.findall(r'[a-zçğıöşü]+|\d+(?:[\.\-]\d+)*', text, flags=re.UNICODE)
    stems = []

    for token in tokens:
        # Harfse kök bul
        if re.match(r'^[a-zçğıöşü]+$', token, flags=re.UNICODE):
            if token in TURKISH_STOPWORDS:
                continue

            if morphology:
                try:
                    analyses = list(morphology.analyze(token))
                    lemma = analyses[0].get_stem() if analyses else token
                except Exception:
                    lemma = token
            else:
                lemma = token
            stems.append(lemma)
        else:
            # Sayıları olduğu gibi bırak
            stems.append(token)

    return " ".join(stems)


# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_text_keep_structure(text: str) -> str:
    """
    Metni temizler ama sayıları, parantez içindeki sayıları ve tarihleri korur.
    """
    # Unicode normalize ve lowercase
    text = unicodedata.normalize("NFKC", text).lower()
    text = ''.join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    # Satır sonu tireleri ve fazla newline/alt çizgi temizle
    text = re.sub(r'-\n', '', text)       # satır sonu tireleri kaldır
    text = re.sub(r'\n+', '\n', text)    # birden fazla newline → tek newline
    text = re.sub(r'_+', ' ', text)      # alt çizgileri boşluk yap

    # URL ve email temizle
    text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)

    # Tarihleri, parantez içindeki sayıları ve diğer sayıları koru
    preserved = re.findall(
        r'\d{1,2}\.\d{1,2}\.\d{2,4}|'  # 25.08.2026
        r'\d{4}-\d{4}|'  # 2025-2026
        r'\d{1,2}-\d{1,2}\s+[a-zçğıöşü]+\s+\d{4}|'  # 08-09 haziran 2026
        r'\(\d+\)|'  # (1)
        r'\d+\.?\d*|'  # normal sayılar
        r'[a-zçğıöşü]+|'  # kelimeler
        r'[%]',  # sadece yüzdeyi ayrı bırak
        text,
        flags=re.UNICODE
    )

    # Kelimeleri tekrar birleştir
    text = ' '.join(preserved)

    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def apply_stemming_on_words(text: str, remove_stopwords=True) -> str:
    """
    Metni temizler ve kelimeleri kökler, ama tarihleri ve parantez içi sayıları korur.
    """
    # 1. Tarihleri ve parantez içi sayıları placeholder ile değiştir
    placeholders = {}
    for i, token in enumerate(re.findall(r'\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|\(\d+\)', text)):
        key = f"__PLACEHOLDER_{i}__"
        text = text.replace(token, key)
        placeholders[key] = token

    # 2. Tokenize (placeholder’ları koru)
    tokens = re.findall(r'__PLACEHOLDER_\d+__|[a-zçğıöşü]+|\d+\.?\d*|[%\.-]', text, flags=re.UNICODE)
    stems = []

    for token in tokens:
        if token in placeholders:
            stems.append(placeholders[token])  # placeholder’ı orijinal değeriyle değiştir
            continue

        token_lower = token.lower()
        if remove_stopwords and token_lower in TURKISH_STOPWORDS:
            continue

        if morphology and re.match(r'^[a-zçğıöşü]+$', token_lower):
            try:
                analyses = list(morphology.analyze(token_lower))
                lemma = analyses[0].get_stem() if analyses else token_lower
            except Exception:
                lemma = token_lower
            stems.append(lemma)
        else:
            stems.append(token)

    return " ".join(stems)


def clean_calendar_table_pdf(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'_+', ' ', text)

    placeholders = {}

    patterns = [
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',
        r'\d{4}-\d{4}',
        r'\d{1,2}-\d{1,2}\s+[a-zçğıöşü]+\s+\d{4}'
    ]

    for i, pat in enumerate(patterns):
        for match in re.findall(pat, text):
            key = f"__DATE_{i}_{len(placeholders)}__"
            text = text.replace(match, key)
            placeholders[key] = match
    text = re.sub(r'\n+', ' | ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    for key, value in placeholders.items():
        text = text.replace(key, value)

    return text.strip()


# -----------------------------
# DENSE CLEAN (Embedding için minimal temizlik)
# -----------------------------
def clean_dense_text(text: str) -> str:
    """
    Embedding modelleri için minimal temizlik.
    Stemming yok, stopword silme yok.
    """
    text = unicodedata.normalize("NFKC", text).lower()

    # Accent normalize
    text = ''.join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    # URL / email temizle
    text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)

    # Fazla boşluk temizle
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def preprocess_title(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if t not in TURKISH_STOPWORDS and len(t) > 1]


# -----------------------------
# RAG Split
# -----------------------------
# PARAGRAPH_PATTERN = re.compile(r"^\s*\((\d+)\)", re.MULTILINE)
# PARAGRAPH_PATTERN = re.compile(r"^\s*(?:\((\d+)\)|(\d+)\.(?=\s+[a-zçğıöşü]))",    re.MULTILINE)
PARAGRAPH_PATTERN = re.compile(r'^\s*(\(?\d+\)?[\.\-\)]?)\s+',re.MULTILINE)

ARTICLE_PATTERN = re.compile(
    r"^\s*(GEÇİCİ\s+MADDE|EK\s+MADDE|MADDE)\s+(\d+[A-Z]?[-/]?\d*)",
    re.IGNORECASE | re.MULTILINE
)

CLAUSE_PATTERN = re.compile(
    r'^\s*(?:\(([a-zçğıöşü])\)|(\d+)[\.\-])\s+',
    re.IGNORECASE | re.MULTILINE
)

ROMAN_MAP = {
    "I": "1", "II": "2", "III": "3",
    "IV": "4", "V": "5", "VI": "6",
    "VII": "7", "VIII": "8", "IX": "9",
    "X": "10"
}

WORD_MAP = {
    "birinci": "1",
    "ikinci": "2",
    "üçüncü": "3",
    "dördüncü": "4",
    "beşinci": "5",
    "altıncı": "6",
    "yedinci": "7",
    "sekizinci": "8",
    "dokuzuncu": "9",
    "onuncu": "10"
}


def split_regulation_rag(text: str) -> List[Dict]:
    chunks = []
    text = text.replace("\f", "\n")
    text = clean_headers(text)
    text = remove_redundant_intro_titles(text)

    article_splits = list(ARTICLE_PATTERN.finditer(text))
    if not article_splits:
        return [{
            "article_type": "GENEL",
            "article_no": "-",
            "paragraph_no": "-",
            "clause": "-",
            "text": text
        }]

    for i, match in enumerate(article_splits):
        article_type = match.group(1).upper() if match.group(1) else "GENEL"
        article_no = match.group(2) if match.group(2) else "-"
        article_no = article_no.rstrip("-")

        start = match.start()
        end = article_splits[i + 1].start() if i + 1 < len(article_splits) else len(text)
        article_text = text[start:end].strip()
        article_lines = article_text.splitlines()
        article_body = "\n".join(article_lines[1:]).strip() if article_lines else ""

        paragraph_splits = list(PARAGRAPH_PATTERN.finditer(article_body))
        if not paragraph_splits:
            chunks.append({
                "article_type": article_type,
                "article_no": article_no,
                "paragraph_no": "-",
                "clause": "-",
                "text": article_text
            })
            continue

        for j, p_match in enumerate(paragraph_splits):
            raw_no = (p_match.group(1) or
                      p_match.group(2) or
                      p_match.group(3) or
                      p_match.group(4) or
                      "-"
                      )
            if raw_no != "-":
                paragraph_no = (
                        ROMAN_MAP.get(raw_no.upper()) or
                        WORD_MAP.get(raw_no.lower()) or
                        raw_no
                )
            else:
                paragraph_no = "-"
            p_start = p_match.start()
            p_end = paragraph_splits[j + 1].start() if j + 1 < len(paragraph_splits) else len(article_body)
            paragraph_text = article_body[p_start:p_end].strip()

            clause_splits = list(CLAUSE_PATTERN.finditer(paragraph_text))
            if not clause_splits:
                chunks.append({
                    "article_type": article_type,
                    "article_no": article_no,
                    "paragraph_no": paragraph_no,
                    "clause": "-",
                    "text": f"{article_type} {article_no} {paragraph_text}"
                })
                continue

            for k, c_match in enumerate(clause_splits):
                clause_letter = c_match.group(1).lower() if c_match.group(1) else "-"
                c_start = c_match.start()
                c_end = clause_splits[k + 1].start() if k + 1 < len(clause_splits) else len(paragraph_text)
                clause_text = paragraph_text[c_start:c_end].strip()
                clause_text = re.sub(r'^\(?([a-z])\)?\s*', '', clause_text, flags=re.IGNORECASE)

                chunks.append({
                    "article_type": article_type,
                    "article_no": article_no,
                    "paragraph_no": paragraph_no,
                    "clause": clause_letter,
                    "text": f"{article_type} {article_no} ({paragraph_no}) {clause_text}"
                })
    return chunks


# -----------------------------
# PARAGRAPH SPLIT
# -----------------------------
def split_into_paragraphs(text: str, min_words: int = 30) -> List[str]:
    text = re.sub(r'\n{2,}', '\n\n', text)
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n(?=[A-ZÇŞĞÜÖİ\s]+BÖLÜM|MADDE)', text) if p.strip()]
    merged_paragraphs, buffer = [], ""
    for para in raw_paragraphs:
        if len(para.split()) < min_words:
            buffer = f"{buffer} {para}".strip()
        else:
            if buffer:
                merged_paragraphs.append(buffer)
                buffer = ""
            merged_paragraphs.append(para)
    if buffer:
        merged_paragraphs.append(buffer)
    return merged_paragraphs


# -----------------------------
# GET CATEGORY METADATA
# -----------------------------
def get_category_metadata(reg_code: str) -> Dict[str, str]:
    if reg_code in REGULATION_CATEGORIES:
        return REGULATION_CATEGORIES[reg_code]
    return {"category": "UNKNOWN", "faculty": "All", "degree_level": "All"}


# -----------------------------
# MAIN PROCESS
# -----------------------------
def process_pdfs(
    raw_pdf_dir: Path,
    raw_text_dir: Path,
    processed_dir: Path,
    interim_dir: Path,
    corpus_file: Path,
    corpus_preprocessed_file: Path,
    metadata_file: Path,
    save_raw_text: bool = True
) -> None:
    """
    Process PDFs into structured corpus and preprocessed corpus.
    Parameters:
        raw_pdf_dir: Path to folder containing raw PDFs.
        raw_text_dir: Path to save intermediate raw text files.
        processed_dir: Path to save processed outputs.
        interim_dir: Path to save interim outputs.
        corpus_file: Path to save full corpus JSON.
        corpus_preprocessed_file: Path to save preprocessed corpus JSON.
        metadata_file: Path to save metadata CSV.
        save_raw_text: Whether to save raw text from PDFs.
    """
    # Ensure directories exist
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    corpus, corpus_preprocessed, metadata_rows = [], [], []
    doc_counter = 1

    for pdf_file in raw_pdf_dir.glob("*.pdf"):
        full_text = pdf_to_text(pdf_file, save_raw_text=save_raw_text)
        if not full_text.strip():
            logging.warning(f"Skipping {pdf_file.name}: empty text")
            continue

        reg_code = pdf_file.stem.split("_")[0]
        regulation_title = pdf_file.stem
        meta_info = get_category_metadata(reg_code)
        title_tokens = preprocess_title(regulation_title)

        try:
            rag_chunks = split_regulation_rag(full_text)
        except Exception as e:
            logging.error(f"RAG split error in {pdf_file.name}: {e}")
            rag_chunks = [{"article_type": "GENEL", "article_no": "-", "paragraph_no": "-", "clause": "-", "text": full_text}]

        for chunk in rag_chunks:
            doc_id = f"DOC_{doc_counter:05d}"
            doc_counter += 1

            entry = {
                "doc_id": doc_id,
                "regulation_code": reg_code,
                "regulation_title": regulation_title,
                "title_tokens": title_tokens,
                "article_type": chunk["article_type"],
                "article_no": chunk["article_no"],
                "paragraph_no": chunk["paragraph_no"],
                "clause": chunk["clause"],
                "text": chunk["text"],
                "category": meta_info["category"],
                "faculty": meta_info["faculty"],
                "degree_level": meta_info["degree_level"]
            }
            corpus.append(entry)

            title_lower = regulation_title.lower()
            is_calendar = any(kw in title_lower for kw in ["akademik takvim", "takvim", "takvimi", "tarihler"])
            is_tabular = any(kw in title_lower for kw in ["harç", "harc", "tablo"])

            # -----------------------------
            # HYBRID CLEANING
            # -----------------------------
            preprocessed_entry = entry.copy()

            para_cleaned = remove_footnotes(chunk["text"])
            para_normalized = normalize_phrases_sparse(para_cleaned)

            if is_calendar or is_tabular or meta_info["category"].lower() == "akademik takvim":
                clean_sparse = clean_calendar_table_pdf(para_normalized)
            else:
                clean_sparse = clean_text_keep_structure(para_normalized)
                clean_sparse = apply_stemming_on_words(clean_sparse)

            clean_dense = clean_dense_text(para_cleaned)

            preprocessed_entry["clean_sparse"] = clean_sparse
            preprocessed_entry["clean_dense"] = clean_dense

            corpus_preprocessed.append(preprocessed_entry)

            metadata_rows.append({
                "doc_id": doc_id,
                "regulation_code": reg_code,
                "regulation_title": regulation_title,
                "category": meta_info["category"],
                "faculty": meta_info["faculty"],
                "degree_level": meta_info["degree_level"],
                "pdf_file": pdf_file.name
            })

    # Save outputs
    corpus_file.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_preprocessed_file.write_text(json.dumps(corpus_preprocessed, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(metadata_rows).to_csv(metadata_file, index=False, encoding="utf-8")
    logging.info(f"Processing complete. Corpus, preprocessed corpus, and metadata saved.")


