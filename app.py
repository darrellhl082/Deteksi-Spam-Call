import streamlit as st
import numpy as np
import pandas as pd
import os
import datetime
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="SmartGuard AI", page_icon="🛡️", layout="wide")

# ==========================================
# 2. DATABASE & KONSTANTA
# ==========================================
PHONE_DATABASE = {
    "0800123456":  {"reports": 145, "kategori": "Bank Palsu", "verified": True},
    "0811999000":  {"reports": 89,  "kategori": "Pinjol Ilegal", "verified": True},
    "02150001234": {"reports": 203, "kategori": "Penipuan Hadiah", "verified": True},
    "+6281234999": {"reports": 67,  "kategori": "VOIP Scammer", "verified": False},
    "+1234567890": {"reports": 312, "kategori": "Luar Negeri", "verified": True},
    "0877000111":  {"reports": 12,  "kategori": "Telemarketing", "verified": False},
    "08123456789": {"reports": 0,   "kategori": "Normal", "verified": False},
    "085111222333":{"reports": 178, "kategori": "Penipuan OTP", "verified": True},
    "021-5000111": {"reports": 55,  "kategori": "Penipuan Pajak", "verified": True}
}

EDUKASI_DATABASE = {
    "Bank Palsu": "⚠️ Bank TIDAK PERNAH meminta PIN, OTP, atau password lewat telepon. Jika ada yang mengaku pihak bank & meminta data sensitif → TUTUP TELEPON.",
    "Penipuan Hadiah": "⚠️ Tidak ada hadiah gratis yang meminta biaya administrasi terlebih dahulu. Ciri khas: desakan segera, minta transfer dulu baru hadiah dikirim.",
    "Pinjol Ilegal": "⚠️ Pinjol legal terdaftar di OJK. Jangan berikan foto KTP & selfie ke aplikasi tidak dikenal. Aduan: kontak157@ojk.go.id",
    "Penipuan OTP": "⚠️ OTP adalah kode rahasia pribadi. JANGAN bagikan ke siapapun. Scammer berpura-pura jadi CS, kurir, atau petugas untuk minta OTP.",
    "Penipuan Pajak": "⚠️ Petugas DJP tidak pernah menagih pajak lewat telepon/WhatsApp. Tagihan resmi dikirim melalui surat atau portal pajak.go.id",
    "VOIP Scammer": "⚠️ Scammer sering menggunakan nomor VOIP (+1, +44) untuk menyembunyikan identitas. Jangan transfer uang atau berikan data apapun.",
    "default": "⚠️ Tips Aman: 1. Jangan bagikan OTP/PIN. 2. Verifikasi identitas penelepon. 3. Laporkan ke truecaller.com atau cekrekening.id"
}

HONEYPOT_REPLIES = {
    "Bank Palsu": "Baik, saya perlu konfirmasi dulu. Bisa tolong sebutkan nama lengkap dan nomor pegawai Anda? Saya akan catat untuk verifikasi ke kantor pusat.",
    "Penipuan Hadiah": "Wah menarik! Tapi saya perlu verifikasi dulu. Bisa kirimkan surat resmi bermaterai ke alamat saya? Saya tidak mau salah klaim hadiah.",
    "Pinjol Ilegal": "Saya tertarik, tapi saya perlu nomor izin OJK aplikasi Anda terlebih dahulu sebelum melanjutkan. Bisa disebutkan nomor registrasinya?",
    "default": "Mohon maaf, saya sedang sibuk. Bisa tolong kirimkan detail lengkap via email resmi dengan kop surat perusahaan? Terima kasih."
}

# ==========================================
# 3. LOAD MODEL (CACHING)
# ==========================================
@st.cache_resource
def load_models():
    model_dir = "smartguard_output"
    try:
        model_text = tf.keras.models.load_model(os.path.join(model_dir, "model_mlp_teks.keras"))
        model_meta = tf.keras.models.load_model(os.path.join(model_dir, "model_mlp_metadata.keras"))
        model_audio = tf.keras.models.load_model(os.path.join(model_dir, "model_cnn_audio.keras"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        tfidf = joblib.load(os.path.join(model_dir, "tfidf.pkl"))
        return model_text, model_meta, model_audio, scaler, tfidf, True
    except Exception as e:
        return None, None, None, None, None, False

# ==========================================
# 4. FUNGSI ANALISIS (CORE LOGIC)
# ==========================================
def clean_phone(nomor):
    return nomor.replace("-", "").replace(" ", "").replace("+", "")

def smartguard_analyze(nomor_telepon, pesan_teks, meta_dict, use_audio, models):
    model_text, model_meta, model_audio, scaler, tfidf, is_loaded = models
    
    nomor_clean = clean_phone(nomor_telepon)
    
    # 1. Community Shield
    db_info = {"reports": 0, "kategori": "Tidak dikenal", "verified": False}
    for db_key, val in PHONE_DATABASE.items():
        if clean_phone(db_key) == nomor_clean:
            db_info = val
            break
            
    scores = []
    weights = []
    detail_scores = {"DB": 0, "Meta": 0, "Teks": 0, "Audio": 0}

    # DB Score
    db_score = min(db_info["reports"] / 50.0, 1.0)
    if db_info["reports"] > 0:
        scores.append(db_score)
        weights.append(0.2)
        detail_scores["DB"] = db_score * 100

    # 2. Metadata Score
    if is_loaded:
        meta_features = np.array([[
            meta_dict['call_duration_sec'], meta_dict['call_hour'], meta_dict['is_voip'], 
            meta_dict['is_foreign_prefix'], meta_dict['call_freq_24h'], meta_dict['call_freq_7d'], 
            meta_dict['avg_duration_short'], meta_dict['community_reports'], meta_dict['is_weekend'],
            meta_dict['call_hour_abnormal']
        ]])
        meta_scaled = scaler.transform(meta_features)
        score_meta = float(model_meta.predict(meta_scaled, verbose=0)[0][0])
    else:
        score_meta = 0.1
        if meta_dict['is_voip'] and meta_dict['is_foreign_prefix']: score_meta += 0.4
        if meta_dict['call_hour_abnormal']: score_meta += 0.2
        if meta_dict['call_freq_24h'] > 10: score_meta += 0.2
        score_meta = min(score_meta, 1.0)
        
    scores.append(score_meta)
    weights.append(0.35)
    detail_scores["Meta"] = score_meta * 100

    # 3. Text Score
    score_text = 0.0
    if pesan_teks:
        if is_loaded:
            text_vec = tfidf.transform([pesan_teks]).toarray()
            score_text = float(model_text.predict(text_vec, verbose=0)[0][0])
        else:
            spam_words = ["otp", "pin", "bank", "blokir", "hadiah", "menang", "pinjol", "pajak"]
            if any(w in pesan_teks.lower() for w in spam_words): score_text = 0.85
            else: score_text = 0.1
        scores.append(score_text)
        weights.append(0.25)
        detail_scores["Teks"] = score_text * 100

    # 4. Audio Score
    score_audio = 0.0
    if use_audio:
        if is_loaded:
            mfcc_input = np.random.randn(1, 50, 13) * 1.5 + np.linspace(1.0, 3.0, 13)
            score_audio = float(model_audio.predict(mfcc_input, verbose=0)[0][0])
        else:
            score_audio = 0.7 if meta_dict['is_voip'] else 0.2
        scores.append(score_audio)
        weights.append(0.20)
        detail_scores["Audio"] = score_audio * 100

    # 5. Ensemble
    final_score = np.average(scores, weights=weights[:len(scores)])
    risk_pct = final_score * 100

    # 6. Kategori
    if db_info["reports"] > 0 and db_info["kategori"] not in ["Normal", "Tidak dikenal"]:
        kategori = db_info["kategori"]
    elif meta_dict['is_voip'] and meta_dict['is_foreign_prefix']:
        kategori = "VOIP Scammer"
    elif pesan_teks:
        teks_lower = pesan_teks.lower()
        if any(k in teks_lower for k in ["otp", "pin", "atm", "rekening", "blokir", "bank"]): kategori = "Bank Palsu"
        elif any(k in teks_lower for k in ["hadiah", "menang", "undian", "promo"]): kategori = "Penipuan Hadiah"
        elif any(k in teks_lower for k in ["pinjam", "kredit", "cicilan", "modal"]): kategori = "Pinjol Ilegal"
        elif any(k in teks_lower for k in ["pajak", "polisi", "pengadilan", "hukum"]): kategori = "Penipuan Pajak"
        else: kategori = "Spam Umum"
    else:
        kategori = "Spam Umum" if risk_pct >= 50 else "Normal"

    # 7. Level
    if risk_pct >= 80: level = "KRITIS 🔴"
    elif risk_pct >= 60: level = "SPAM 🟠"
    elif risk_pct >= 40: level = "WASPADA 🟡"
    else: level = "AMAN ✅"

    return {
        "nomor": nomor_telepon,
        "risk_score": round(risk_pct, 1),
        "level": level,
        "kategori": kategori,
        "db_info": db_info,
        "detail_scores": detail_scores,
        "edukasi": EDUKASI_DATABASE.get(kategori, EDUKASI_DATABASE["default"]),
        "honeypot": HONEYPOT_REPLIES.get(kategori, HONEYPOT_REPLIES["default"])
    }

# ==========================================
# 5. UI STREAMLIT (MAIN PAGE LAYOUT)
# ==========================================
def main():
    st.title("🛡️ SMARTGUARD: Context-Aware Spam Call Assistant")
    st.markdown("*Sistem Deteksi Dini & Edukasi Penipuan Telepon Berbasis Ensemble AI (MLP + 1D CNN)*")
    
    models = load_models()
    is_loaded = models[-1]
    
    if not is_loaded:
        st.warning("⚠️ **Model AI tidak ditemukan.** Pastikan folder `smartguard_output/` tersedia. Aplikasi berjalan dalam **Mode Simulasi/Demo**.")

    st.markdown("---")

    # --- FORM INPUT DI MAIN PAGE ---
    st.header("📞 Panel Simulasi Panggilan Masuk")
    
    with st.form("smartguard_input_form"):
        # Baris 1: Identitas & Metadata
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            st.subheader("📱 Identitas & Pesan")
            nomor = st.text_input("Nomor Telepon", value="085111222333", help="Contoh: 085111222333, +1234567890")
            pesan = st.text_area("Isi Pesan / SMS (Opsional)", value="Rekening BCA Anda diblokir! Verifikasi OTP sekarang ke nomor kami", height=120)
            
        with col_right:
            st.subheader("⚙️ Metadata Panggilan")
            meta_col1, meta_col2 = st.columns(2)
            
            with meta_col1:
                dur = st.number_input("Durasi (detik)", 0, 600, 30)
                jam = st.slider("Jam Panggilan (0-23)", 0, 23, 2)
                voip = st.selectbox("Jaringan VOIP?", [0, 1], index=1)
                foreign = st.selectbox("Prefix Luar Negeri?", [0, 1], index=1)
                
            with meta_col2:
                freq_24 = st.number_input("Frek. 24 Jam", 0, 50, 18)
                freq_7 = st.number_input("Frek. 7 Hari", 0, 100, 70)
                reports = st.number_input("Laporan Komunitas", 0, 500, 25)
                weekend = st.selectbox("Akhir Pekan?", [0, 1], index=1)
                
        # Baris 2: Audio & Tombol Aksi
        col_audio, col_spacer, col_btn = st.columns([1, 1, 2])
        with col_audio:
            use_audio = st.checkbox("🎙️ Aktifkan Analisis Audio (Simulasi MFCC)", value=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer kecil
            submitted = st.form_submit_button("🚀 ANALISIS PANGGILAN SEKARANG", type="primary", use_container_width=True)

    st.markdown("---")

    # --- HASIL ANALISIS (MUNCUL SETELAH TOMBOL DITEKAN) ---
    if submitted:
        meta_dict = {
            'call_duration_sec': dur,
            'call_hour': jam,
            'is_voip': voip,
            'is_foreign_prefix': foreign,
            'call_freq_24h': freq_24,
            'call_freq_7d': freq_7,
            'avg_duration_short': 1 if dur < 30 else 0,
            'community_reports': reports,
            'is_weekend': weekend,
            'call_hour_abnormal': 1 if (jam < 6 or jam >= 22) else 0
        }
        
        with st.spinner("🧠 Menganalisis pola panggilan, metadata, teks, dan audio..."):
            import time
            time.sleep(1.5) # Simulasi delay AI
            result = smartguard_analyze(nomor, pesan, meta_dict, use_audio, models)
        
        # Push Notification Simulation
        st.toast(f"📱 SmartGuard Alert: Panggilan dari {nomor} terdeteksi {result['level']}", icon="🛡️")

        # Metrics Row
        st.subheader("📊 Hasil Analisis Risiko")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{result['risk_score']}%", delta="High Risk" if result['risk_score'] >= 60 else "Low Risk", delta_color="inverse")
        m2.metric("Level", result['level'])
        m3.metric("Kategori", result['kategori'])
        
        st.progress(result['risk_score'] / 100, text=f"Indikator Risiko: {result['risk_score']}%")
        st.markdown("---")
        
        # Detail Scores & DB
        col_dash_left, col_dash_right = st.columns([1, 1])
        
        with col_dash_left:
            st.markdown("#### 🧠 Breakdown Skor Model (Ensemble)")
            df_scores = pd.DataFrame({
                "Sumber Data": ["Community DB", "MLP Metadata", "MLP Teks", "1D CNN Audio"],
                "Skor Risiko (%)": [
                    result['detail_scores']['DB'], 
                    result['detail_scores']['Meta'], 
                    result['detail_scores']['Teks'], 
                    result['detail_scores']['Audio']
                ]
            })
            st.bar_chart(df_scores.set_index("Sumber Data"), color="#ff4b4b" if result['risk_score'] > 50 else "#00cc00")
            
            st.markdown("#### 🌍 Community Shield (Database)")
            if result['db_info']['reports'] > 0:
                st.error(f"⚠️ Ditemukan **{result['db_info']['reports']} Laporan** dari komunitas.")
                st.info(f"Kategori Terdaftar: **{result['db_info']['kategori']}** | Verified: {'✅ Ya' if result['db_info']['verified'] else '❌ Belum'}")
            else:
                st.success("Nomor belum pernah dilaporkan di database komunitas.")

        with col_dash_right:
            st.markdown("#### 🚨 Status & Tindakan")
            if result['risk_score'] >= 70:
                st.error("**REKOMENDASI:** BLOKIR NOMOR & JANGAN ANGKAT.")
            elif result['risk_score'] >= 40:
                st.warning("**REKOMENDASI:** WASPADA. Jangan berikan data pribadi.")
            else:
                st.success("**REKOMENDASI:** Panggilan terindikasi AMAN.")

            if pesan:
                st.markdown("**📝 Analisis Pesan Teks:**")
                st.code(pesan, language="text")

        st.markdown("---")
        
        # Edukasi & Honeypot
        if result['risk_score'] >= 40:
            st.subheader("📚 Edukasi Post-Call & Proteksi")
            with st.expander("⚠️ Lihat Edukasi Keamanan (Klik untuk membuka)", expanded=True):
                st.warning(result['edukasi'])
                
            if result['risk_score'] >= 60:
                with st.expander("🍯 Fitur Auto-Honeypot Reply (Balasan Pancingan)", expanded=True):
                    st.markdown("Sistem menyarankan balasan otomatis untuk membuang waktu scammer dan mengumpulkan bukti:")
                    st.chat_message("user").write(result['honeypot'])
                    st.caption("*Balasan ini dapat dikirimkan otomatis oleh sistem untuk melindungi Anda sambil melacak pola scammer.*")
                    
    else:
        # Tampilan awal sebelum dianalisis
        st.info("👆 Silakan isi parameter simulasi di atas dan klik **ANALISIS PANGGILAN SEKARANG** untuk memulai deteksi AI.")

if __name__ == "__main__":
    main()