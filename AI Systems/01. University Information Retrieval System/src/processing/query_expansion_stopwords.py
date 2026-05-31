# -----------------------------
# STOPWORDS
# -----------------------------
TURKISH_STOPWORDS = {"ve", "ile", "ama", "ancak", "fakat", "lakin", "veya", "ya", "yahut", "çünkü", "ise", "ki", "dahi",
                     "için", "olarak", "bu", "şu", "o", "bunlar", "şunlar", "onlar", "her", "bazı", "tüm", "hiç",
                     "kendi", "mi", "mı", "mu", "mü", "çok", "daha", "en", "az", "son", "önce", "sonra", "kadar",
                     "gibi", "fıkra", "bent", "sayılı", "tarihli", "uyarınca", "gereğince", "kapsamında", "ilişkin",
                     "dair", "hakkında", "yapılır", "edilir", "olur", "olduğu", "amacıyla", "şekilde", "hususunda",
                     "etmek", "olmak", "yapmak", "bulunmak", "göstermek", "sunmak", "belirlemek"}

# -----------------------------
# QUERY EXPANSION
# -----------------------------
HINT_EXPANSION = {
    "ders kaydı": ["akademik takvim", "kayıt tarihleri", "ders ekleme çıkarma"],
    "bahar dönemi başlangıç": ["akademik takvim", "eğitim yılı takvimi"],
    "final sınav": ["yarıyıl sonu sınavı", "sınav takvimi", "sınav tarihleri"],
    "mezuniyet": ["diploma hakkı", "staj bitirme projesi", "mezuniyet koşulları"],
}

SYNONYM_EXPANSION = {
    "şart": ["koşul", "gereklilik"],
    "koşul": ["şart", "gereklilik"],
    "madde": ["hüküm"],
    "başvuru": ["müracaat", "form", "dilekçe"],
    "kayıt silme": ["ilişik kesme", "kayıt iptali", "çıkış"],
    "mezuniyet": ["diploma hakkı", "staj", "bitirme projesi"],
    "ders kaydı": ["kayıt", "ders seçimi", "ekle/çıkar"],
    "sınav": ["ara sınav", "final", "bütünleme", "sınav tarihi"],
    "final": ["yarıyıl sonu sınavı"],
    "bütünleme": ["tek ders sınavı", "bütünleme sınavı"],
    "tarih": ["zaman", "başlangıç", "bitiş", "son tarih", "süre"],
    "kredi": ["ders kredisi", "AKTS"],
    "staj": ["zorunlu staj", "uygulamalı ders"],
    "burs": ["maddi yardım", "katkı"],
    "not": ["başarı notu", "harf notu"],
    "öğrenci belgesi": ["öğrenci kimliği", "transkript"]
}

CONCEPT_EXPANSION = {
    "yatay geçiş": ["kurumlararası geçiş", "merkezi yerleştirme puanı"],
    "dikey geçiş": ["intibak", "dgs"],
    "disiplin": ["disiplin cezası", "uyarı", "kınama"],
    "sınav": ["ara sınav", "final", "bütünleme"],
    "mezuniyet": ["staj", "diploma", "bitirme projesi"],
    "tarih": ["başlangıç", "bitiş", "son tarih", "süre"],
    "ders kaydı": ["ekle/çıkar", "danışman onayı"],
    "başvuru": ["form", "dilekçe", "evrak"],
    "kayıt silme": ["ilişik kesme", "çıkış"]
}

HIERARCHICAL_EXPANSION = {
    "ara sınav": ["sınav"],
    "final": ["sınav"],
    "bütünleme": ["sınav"],
    "kınama": ["disiplin"],
    "uyarı": ["disiplin"],
    "tarih": ["zaman"],
    "başvuru": ["belge"],
    "mezuniyet": ["öğrenci hakları"],
    "staj": ["mezuniyet"],
    "burs": ["öğrenci hakları"],
    "not": ["sınav"]
}
