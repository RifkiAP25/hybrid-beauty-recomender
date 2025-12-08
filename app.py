import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
import requests
import pickle
from io import BytesIO
import google.generativeai as genai
import os

# ======================= LOAD DATA FROM GITHUB ===========================
URL_PRODUCTS = "https://raw.githubusercontent.com/RifkiAP25/hybrid-beauty-recomender/refs/heads/main/model/products.pkl"
URL_SVM      = "https://raw.githubusercontent.com/RifkiAP25/hybrid-beauty-recomender/refs/heads/main/model/svm_model.pkl"
URL_FAISS    = "https://raw.githubusercontent.com/RifkiAP25/hybrid-beauty-recomender/refs/heads/main/model/faiss_index.bin"

@st.cache_resource
def load_models():
    product_df = pickle.load(BytesIO(requests.get(URL_PRODUCTS).content))
    svm_model  = joblib.load(BytesIO(requests.get(URL_SVM).content))

    faiss_bytes = requests.get(URL_FAISS).content
    with open("faiss_index.bin", "wb") as f:
        f.write(faiss_bytes)
    faiss_index = faiss.read_index("faiss_index.bin")

    return product_df, svm_model, faiss_index

product_df, svm_model, faiss_index = load_models()
faiss_embed = np.vstack(product_df["embedding"].values).astype("float32")

# =================== CSS CUSTOM ================================
hide_password_icon = """
<style>
input[type="password"]::-ms-reveal,
input[type="password"]::-ms-clear {
    display: none;
    width: 0;
    height: 0;
}
input[type="password"]::-webkit-textfield-decoration-container {
    display: none;
}
div[data-testid="stDataFrame"] table {
    border-radius: 12px;
}
</style>
"""
st.markdown(hide_password_icon, unsafe_allow_html=True)

# =================== SIDEBAR NAVIGASI ==========================
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard", "💄 Rekomendasi Produk", "ℹ️ Tentang Aplikasi"])

# =================== DASHBOARD ================================
if menu == "🏠 Dashboard":
    st.title("💋 AI Beauty Recommendation Dashboard")

    st.markdown("""
    Selamat datang! 🎉  
    Aplikasi ini menggunakan **Hybrid AI System** untuk merekomendasikan produk kecantikan berdasarkan:

    🔍 *Kemiripan formula & fungsi*  
    💬 *Review dan sentimen konsumen*  
    ⭐ *Prediksi rating menggunakan SVM*  
    🤖 *Penjelasan rekomendasi lewat Chatbot AI (Gemini)*  

    ### 🧠 Cara Kerja:
    1️⃣ Mencari produk **paling mirip** dengan FAISS  
    2️⃣ Analisis sentimen dari ulasan pengguna  
    3️⃣ SVM memprediksi potensi rating tinggi  
    4️⃣ Sistem menggabungkan semuanya menjadi **Hybrid Score**  
    5️⃣ Chatbot menjelaskan alasan rekomendasinya ✨  

    👉 Mulai coba rekomendasi di menu **💄 Rekomendasi Produk**
    """)

# =================== HALAMAN REKOMENDASI ======================
elif menu == "💄 Rekomendasi Produk":
    st.title("✨ Beauty AI Recommender")
    st.caption("Hybrid Semantic + Sentiment + Prediction + Explainable Chatbot")

    product_list = sorted(product_df["item_reviewed"].unique())
    selected = st.selectbox("🔎 Pilih Produk:", product_list)

    if st.button("🎯 Tampilkan Rekomendasi"):
        idx = product_df[product_df["item_reviewed"] == selected].index[0]
        query_vec = faiss_embed[idx].reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vec)

        D, I = faiss_index.search(query_vec, 6)
        candidates = product_df.iloc[I[0][1:]].copy()
        candidates["faiss_sim"] = D[0][1:]
        candidates["prob_svm"] = svm_model.predict_proba(np.vstack(candidates["embedding"]))[:, 1]
        candidates["hybrid_score"] = 0.6 * candidates["faiss_sim"] + 0.4 * candidates["prob_svm"]

        st.session_state["candidates"] = candidates

        st.subheader("💄 Rekomendasi Produk Mirip (Top 5)")
        st.dataframe(candidates[["item_reviewed", "sentiment_score", "hybrid_score"]].head(5))

    st.write("---")
    st.subheader("🤖 Tanya Alasan Rekomendasi")

    default_key = os.getenv("GEMINI_API_KEY")

    if default_key:
        st.success("🔐 API Key terdeteksi dari sistem. Chatbot siap digunakan!")
        api_key = default_key
    else:
        api_key = st.text_input("Masukkan API Key Gemini:", type="password")

        st.markdown(
            "<a href='https://ai.google.dev/gemini-api/docs/api-key' target='_blank'>📌 Cara mendapatkan API Key Gemini</a>",
            unsafe_allow_html=True
        )

    if st.button("📌 Jelaskan Rekomendasi Teratas"):
        if "candidates" not in st.session_state:
            st.warning("⚠ Tampilkan rekomendasi dulu!")
        elif not api_key:
            st.error("⚠ Masukkan API Key dulu ya!")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                top = st.session_state["candidates"].iloc[0]

                prompt = f"""
                Jelaskan secara natural mengapa **{top['item_reviewed']}** cocok
                sebagai alternatif dari **{selected}**. Fokus pada manfaat, pengalaman pengguna, dan kesesuaian fungsi.
                Tidak perlu menyebut angka, skor, atau istilah teknis.
                """

                response = model.generate_content(prompt)
                st.success("✨ Penjelasan Rekomendasi:")
                st.write(response.text)

            except Exception as e:
                st.error("Terjadi error:")
                st.code(str(e))

# =================== ABOUT PAGE ================================
elif menu == "ℹ️ Tentang Aplikasi":
    st.title("👩‍💻 Tentang Aplikasi & Tim")

    st.markdown("""
    ### 🧑‍🎓 Tim Pengembang
    | Nama | Peran |
    |------|-------|
    | **Rifki Abdul** | Data Science, NLP, Model AI |
    | **Anggota 2** | Data Analyst |
    | **Anggota 3** | Backend & Deployment |
    | **Anggota 4** | UI/UX & Documentation |

    ---
    ### 💄 Tujuan Aplikasi
    Membantu pengguna menemukan **alternatif produk kecantikan terbaik** dengan analisis:
    - Kemiripan komposisi & fungsi
    - Review konsumen & sentimen
    - Prediksi rating memakai Machine Learning
    - Penjelasan menggunakan AI Generatif (Gemini)

    🚀 Dibangun menggunakan:
    **Python, Streamlit, FAISS, SentenceTransformer, SVM, Gemini AI**
    """)
