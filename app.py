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

# --- Load data dari GitHub
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


# --- CSS theme & visibility fix
st.markdown("""
<style>

:root {
    --primary: #ff4b9f;           /* pink cerah */
    --text-dark-pink: #b30059;    /* pink gelap */
}

/* background gradient */
.stApp {
    background-image:
        linear-gradient(rgba(255, 192, 203, 0.1), rgba(255, 192, 203, 0.99)),
        url("https://images.unsplash.com/photo-1600428877878-1a0fd85beda8?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}


/* fix max width */
.block-container {
    max-width: 900px;
    margin: auto;
}

/* =========================================
1) GLOBAL TEXT 
========================================= */
html, body, p, li, span, div, td, th {
    color: var(--text-dark-pink) !important;
}

/* markdown container override */
.markdown-text-container * {
    color: var(--text-dark-pink) !important;
}

/* =========================================
2) SIDEBAR TEXT 
========================================= */
aside, .stSidebar, .stSidebar * {
    color: var(--primary) !important;
    font-weight: 600 !important;
}

/* =========================================
3) HEADER TITLES 
========================================= */
h1, h2, h3 {
    color: var(--primary) !important;
    font-weight: 800 !important;
}

/* card styling */
.section-card {
    padding: 20px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* buttons */
div.stButton > button {
    background-color: var(--primary) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
}
div.stButton > button:hover {
    background-color: #ff72b6 !important;
}

/* dataframe style */
div[data-testid="stDataFrame"] table {
    border-radius: 12px !important;
    overflow: hidden;
}

/* Ubah warna label Selectbox & semua teks di dalamnya */
div[data-baseweb="select"] * {
    color: var(--primary) !important;
    font-weight: 600 !important;
}

/* hide password reveal */
input[type="password"]::-ms-reveal,
input[type="password"]::-ms-clear {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# --- Header card component
def header_card(title, subtitle):
    st.markdown(f"""
        <div class="section-card">
            <h1 style="margin-bottom:-8px;">{title}</h1>
            <p style="font-size:17px;color:#555;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)


# --- Sidebar menu
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Dashboard", "Rekomendasi Produk", "Tentang Aplikasi"])


# --- Dashboard Page
if menu == "Dashboard":

    header_card(
        "AI Beauty Recommendation Dashboard",
        "Aplikasi rekomendasi kecantikan berbasis Hybrid AI: Semantic Search, Sentiment Analysis, SVM Prediction, dan Explainable AI."
    )

    col1, col2 = st.columns([1, 1])

    # FIXED image: Unsplash direct image URL
    with col1:
        st.image(
    "https://plus.unsplash.com/premium_vector-1721478541700-44a71b5aea37?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    caption="Beauty Inspiration",
    use_container_width=True
)

    with col2:
        st.markdown("""
        ### Fitur Utama
        - Deteksi kemiripan produk menggunakan FAISS
        - Analisis sentimen ulasan pengguna
        - Prediksi kepuasan memakai SVM
        - Penjelasan rekomendasi oleh Gemini AI

        ### Cara Kerja Sistem
        1. Cari produk paling mirip  
        2. Hitung sentimen dan prediksi rating  
        3. Gabungkan menjadi Hybrid Score  
        4. Jelaskan secara natural lewat AI  
        """)

    st.markdown("<br>", unsafe_allow_html=True)


# --- Rekomendasi Produk Page
elif menu == "Rekomendasi Produk":

    header_card("Beauty AI Recommender",
                "Hybrid Semantic • Sentiment • Prediction • Explainable AI")

    product_list = sorted(product_df["item_reviewed"].unique())
    selected = st.selectbox("Pilih Produk:", product_list)

    if st.button("Tampilkan Rekomendasi"):

        with st.spinner("Menganalisis kemiripan produk..."):
            idx = product_df[product_df["item_reviewed"] == selected].index[0]
            query_vec = faiss_embed[idx].reshape(1, -1).astype("float32")
            faiss.normalize_L2(query_vec)

            D, I = faiss_index.search(query_vec, 6)
            candidates = product_df.iloc[I[0][1:]].copy()
            candidates["faiss_sim"] = D[0][1:]
            candidates["prob_svm"] = svm_model.predict_proba(np.vstack(candidates["embedding"]))[:, 1]
            candidates["hybrid_score"] = 0.6 * candidates["faiss_sim"] + 0.4 * candidates["prob_svm"]

            st.session_state["candidates"] = candidates

        st.success("Rekomendasi siap ditampilkan.")

        st.subheader("Rekomendasi Produk Mirip (Top 5)")
        st.dataframe(candidates[["item_reviewed", "sentiment_score", "hybrid_score"]].head(5))


    st.write("---")
    st.subheader("Tanya Alasan Rekomendasi")

    default_key = os.getenv("GEMINI_API_KEY")

    if default_key:
        st.success("API Key otomatis terdeteksi.")
        api_key = default_key
    else:
        api_key = st.text_input("Masukkan API Key Gemini:", type="password")
        st.markdown(
            "<a href='https://ai.google.dev/gemini-api/docs/api-key' target='_blank'>Cara mendapatkan API Key Gemini</a>",
            unsafe_allow_html=True
        )

    if st.button("Jelaskan Rekomendasi Teratas"):

        if "candidates" not in st.session_state:
            st.warning("Tampilkan rekomendasi terlebih dahulu.")
        elif not api_key:
            st.error("Masukkan API Key terlebih dahulu.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                top = st.session_state["candidates"].iloc[0]

                prompt = f"""
                Jelaskan secara natural mengapa {top['item_reviewed']} cocok
                sebagai alternatif dari {selected}. Fokus pada manfaat, pengalaman pengguna, dan kesesuaian fungsi.
                """

                response = model.generate_content(prompt)

                with st.expander("Penjelasan AI"):
                    st.write(response.text)

            except Exception as e:
                st.error("Terjadi error:")
                st.code(str(e))


# --- About Page
elif menu == "Tentang Aplikasi":

    header_card("Tentang Aplikasi & Tim",
                "Dikembangkan oleh Kelompok 8 - Program Celerates")

    st.markdown("""
    ### Tim Pengembang
    | Nama |
    |------|
    | Rifki Alifiani Putra |
    | Rafiqa Ardelia Rahmawati |
    | Alifa Zuriyatul Haq |
    | M. Farhan Khoirur Ridho |

    ---
    ### Tujuan Aplikasi
    Membantu pengguna menemukan alternatif produk kecantikan terbaik dengan:
    - Analisis komposisi dan fungsi
    - Sentimen ulasan pengguna
    - Prediksi rating menggunakan SVM
    - Penjelasan menggunakan Gemini AI

    Dibangun dengan:
    Python, Streamlit, FAISS, SentenceTransformer, SVM, Gemini AI
    """)


# --- Footer
st.markdown("""
<br><hr>
<div style="text-align:center; padding:15px; margin-top:20px;
            background: rgba(255, 255, 255, 0.5);
            border-radius:12px; color:#ffffff;">
    <b>Beauty AI Recommender</b><br>
    Celerates 2025 · Kelompok 8
    <div style="margin-top:8px;">
        <a href="https://www.kaggle.com/datasets/natashamessier/sephora-skincare-reviews-and-sentiment?utm_source=chatgpt.com" 
        target="_blank" 
        style="margin:0 8px; color:#ff4b9f; text-decoration:none;">
        Kaggle Source
        </a>
        |
        <a href="https://github.com/RifkiAP25/hybrid-beauty-recomender" 
        target="_blank" 
        style="margin:0 8px; color:#ff4b9f; text-decoration:none;">
        GitHub
        </a>
        |
        <a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://celerates.co.id/&ved=2ahUKEwiv66-hobWRAxUlyzgGHTkLBjIQFnoECDkQAQ&usg=AOvVaw2U_aT-Sqqp7fOz4D1456tS" 
        target="_blank" 
        style="margin:0 8px; color:#ff4b9f; text-decoration:none;">
        Celerates
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
