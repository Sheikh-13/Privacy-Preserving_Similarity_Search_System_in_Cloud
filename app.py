import streamlit as st
import numpy as np
import os
import json
import logging
import itertools
import base64
import io
import pandas as pd
from datetime import datetime

from phe import paillier
from scipy.spatial import distance
from sklearn.decomposition import PCA
import plotly.express as px

# =====================
# LOGGING AND MONITORING
# =====================
class PrivacyLogger:
    def __init__(self, log_file='privacy_search.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_search_operation(self, query_vector, results, user_id=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_vector': np.asarray(query_vector).tolist(),
            'results_count': len(results),
            'user_id': user_id or 'anonymous'
        }
        self.logger.info(f"Similarity Search: {json.dumps(log_entry)}")

# =====================
# DATASET VALIDATION
# =====================
def validate_dataset(dataset):
    try:
        arr = np.array(dataset, dtype=str)

        # Identify numeric columns
        numeric_mask = []
        for col in arr.T:
            try:
                col.astype(float)
                numeric_mask.append(True)
            except:
                numeric_mask.append(False)

        numeric_mask = np.array(numeric_mask)
        numeric_data = arr[:, numeric_mask].astype(float)
        numeric_data = np.nan_to_num(numeric_data, nan=0.0)

        return True, arr, numeric_data, numeric_mask
    except Exception as e:
        st.error(f"Dataset Validation Error: {e}")
        return False, None, None, None

# =====================
# CSV LOADING UTILITY
# =====================
def load_csv_with_header(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            text = raw.decode('utf-8')
        except:
            text = raw.decode('latin-1')
    else:
        text = raw

    s = io.StringIO(text)
    # Read the first line as header
    header = s.readline().strip().split(",")
    # Load rest of data
    data = np.genfromtxt(s, delimiter=",", dtype=str)
    if data.ndim == 1:  # Only 1 row
        data = np.array([data])
    return header, data

# =====================
# ENCRYPTION FUNCTIONS
# =====================
def generate_keys():
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

class MultiLayerEncryption:
    def __init__(self, public_key):
        self.public_key = public_key
        self.salt = os.urandom(16)

    def encrypt(self, data):
        numeric_data = [float(x) for x in data]
        return [self.public_key.encrypt(x) for x in numeric_data]

# =====================
# DOWNLOAD UTILITY
# =====================
def create_download_link_text_and_cipher(text_df: pd.DataFrame, cipher_list, filename):
    cipher_cols = {}
    if len(cipher_list) > 0:
        n_cipher = len(cipher_list[0])
        for j in range(n_cipher):
            cipher_cols[f"enc_col_{j}"] = [row[j] for row in cipher_list]

    download_df = pd.concat([text_df.reset_index(drop=True), pd.DataFrame(cipher_cols)], axis=1)
    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)

    b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Encrypted Dataset (text + ciphertexts)</a>'
    return href

# =====================
# VP-TREE IMPLEMENTATION
# =====================
class OptimizedVPTree:
    def __init__(self, encrypted_data, private_key, distance_metric='euclidean'):
        self.private_key = private_key
        self.data = encrypted_data
        self.distance_metric = distance_metric
        self.cache = {}
        self.distance_func = {
            'euclidean': distance.euclidean,
            'manhattan': distance.cityblock,
            'cosine': distance.cosine
        }.get(distance_metric, distance.euclidean)
        self.tree = self.build_tree(self.data)

    def decrypt_row(self, row):
        return [self.private_key.decrypt(cell) for cell in row]

    def cached_distance(self, point1, point2):
        k1 = tuple(map(float, point1))
        k2 = tuple(map(float, point2))
        cache_key = (k1, k2)
        if cache_key not in self.cache:
            self.cache[cache_key] = float(self.distance_func(point1, point2))
        return self.cache[cache_key]

    def build_tree(self, data):
        if data is None or len(data) == 0:
            return None
        vp = self.decrypt_row(data[0])
        st.write("Vantage Point (Decrypted):", vp)
        if len(data) == 1:
            return {'vp': vp, 'mu': None, 'left': None, 'right': None}
        distances = [self.cached_distance(vp, self.decrypt_row(point)) for point in data[1:]]
        mu = float(np.median(distances))
        left = [data[i + 1] for i, d in enumerate(distances) if d <= mu]
        right = [data[i + 1] for i, d in enumerate(distances) if d > mu]
        return {'vp': vp, 'mu': mu, 'left': self.build_tree(left), 'right': self.build_tree(right)}

    def search(self, query, k=3):
        decrypted_query = list(map(float, query))
        distances = [self.cached_distance(decrypted_query, self.decrypt_row(row)) for row in self.data]
        nearest_indices = np.argsort(distances)[:k]
        return [self.decrypt_row(self.data[i]) for i in nearest_indices]

# =====================
# MAIN STREAMLIT APPLICATION
# =====================
def main():
    st.title("Privacy-Preserving Similarity Search")

    dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode", value=False)
    if dark_mode:
        st.markdown("""
            <style>
            .stApp { background-color: #121212; color: white; }
            section[data-testid="stSidebar"] { background-color: #1e1e1e; color: white; }
            </style>
        """, unsafe_allow_html=True)
    # 🎨 Gradient for Header & Sidebar
    st.markdown("""
        <style>
        .st-emotion-cache-10trblm {
            background: linear-gradient(90deg, #6a11cb, #2575fc, #ff6f61);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4);
            background-size: 200% 200%;
            animation: sidebarGradient 6s ease infinite;
        }
        @keyframes sidebarGradient {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        </style>
    """, unsafe_allow_html=True)
    st.write("Upload your dataset with headers. Text columns such as Patient ID and Age will be preserved.")
    privacy_logger = PrivacyLogger()
    st.sidebar.header("Dataset Management")

    public_key, private_key = generate_keys()
    multi_layer_encryption = MultiLayerEncryption(public_key)

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            header, dataset = load_csv_with_header(uploaded_file)
            is_valid, cleaned_dataset, numeric_data, numeric_mask = validate_dataset(dataset)
            if not is_valid:
                st.stop()

            st.write("Raw Dataset:")
            st.dataframe(np.vstack([header, cleaned_dataset]))
             # 🎬 Step Animation
            st.markdown("""
                <div style="display:flex;align-items:center;gap:15px;">
                    <div style="padding:8px 12px;border-radius:8px;background:#6a11cb;color:white;">🔒 Encrypt</div>
                    <div style="animation:blink 1s infinite;">➡️</div>
                    <div style="padding:8px 12px;border-radius:8px;background:#2575fc;color:white;">⚙️ Compute</div>
                    <div style="animation:blink 1s infinite;">➡️</div>
                    <div style="padding:8px 12px;border-radius:8px;background:#ff6f61;color:white;">🔓 Decrypt</div>
                </div>

                <style>
                @keyframes blink { 50% { opacity: 0; } }
                </style>
            """, unsafe_allow_html=True)

            st.info("⚙️ Performing encryption and computation...")


            # Encrypt numeric columns only
            encrypted_dataset = [multi_layer_encryption.encrypt(row[numeric_mask]) for row in cleaned_dataset]
            encrypted_flat = [[str(num.ciphertext()) for num in row] for row in encrypted_dataset]
            st.write("Encrypted Dataset (Ciphertexts):", encrypted_flat)

            vp_tree = OptimizedVPTree(encrypted_dataset, private_key)
            st.write("VP-Tree built successfully.")

            # 🔍 Enhanced Visualization
            if st.sidebar.checkbox("Visualize Dataset Embedding"):
                plot_type = st.sidebar.selectbox("Choose Visualization Type:", ["Scatter Plot", "Histogram"])
                try:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(cleaned_dataset)

                    if plot_type == "Scatter Plot":
                        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], title="Dataset Scatter Plot (PCA Reduced)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Scatter Plot Insights
                        - Each point represents a dataset record after dimension reduction.
                        - Points closer together imply higher similarity.
                        - Helps identify possible clusters and groupings.
                        """)

                    elif plot_type == "Histogram":
                        fig = px.histogram(x=reduced_data[:, 0], title="Dataset Histogram (First PCA Component)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Histogram Insights
                        - Shows distribution of dataset values along the primary PCA component.
                        - Identifies most common value ranges and spread.
                        - Useful to detect variance and skewness.
                        """)

                except Exception as e:
                    st.error(f"Visualization Error: {e}")

            if st.sidebar.button("Download Encrypted Dataset"):
                download_filename = "encrypted_dataset.csv"
                download_link = create_download_link(encrypted_flat, download_filename)
                st.sidebar.markdown(download_link, unsafe_allow_html=True)

            st.sidebar.header("Query")
            query_input = st.sidebar.text_input("Enter Patient ID,Age,Weight,Blood Pressure,Cholestrol,Disease Risk Score (comma-separated):")
            k = st.sidebar.slider("Select number of similar results (k):", 1, max(1, numeric_data.shape[0]), 3)

            if query_input:
                try:
                    query_vector = np.array([float(x) for x in query_input.split(",")])
                    if query_vector.size != numeric_data.shape[1]:
                        st.error(f"Query must have {numeric_data.shape[1]} numeric values.")
                    else:
                        decrypted_results = vp_tree.search(query_vector, k)
                        st.write(f"Top {k} Similar Records:")

                        # Merge text + numeric columns
                        final_output = []
                        for idx, row in enumerate(decrypted_results):
                            merged = []
                            num_idx = 0
                            text_idx = 0
                            for is_num in numeric_mask:
                                if is_num:
                                    merged.append(row[num_idx])
                                    num_idx += 1
                                else:
                                    merged.append(cleaned_dataset[idx][text_idx])
                                    text_idx += 1
                            final_output.append(merged)

                        st.dataframe(np.vstack([header, final_output]))
                        privacy_logger.log_search_operation(query_vector, decrypted_results)
                except ValueError:
                    st.error("Query input contains invalid numeric values.")
        except Exception as e:
            st.error(f"Dataset Processing Error: {e}")
    else:
        st.info("Please upload a dataset to begin.")

if __name__ == "__main__":
    main()
