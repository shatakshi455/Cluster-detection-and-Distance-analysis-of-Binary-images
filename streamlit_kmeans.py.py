import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

st.set_page_config(page_title="K-Means Image Clustering", layout="centered")
st.title("🧠 K-Means Clustering on Binary Images")
st.markdown("Drag and drop a grayscale image. The app will identify white regions and cluster them using K-Means.")

# --- Functions ---
def load_binary_image(uploaded_file, threshold=128):
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)
    _, binary = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    return img_np, binary

def get_foreground_coords(binary_img):
    return np.column_stack(np.where(binary_img == 255))

def compute_wcss(points, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(points)
        wcss.append(model.inertia_)
    return wcss

def find_elbow_point(wcss):
    deltas = [wcss[i - 1] - wcss[i] for i in range(1, len(wcss))]
    for i in range(1, len(deltas) - 1):
        if 5 * deltas[i] < deltas[i - 1]:
            return i + 1
    return len(wcss)

def perform_clustering(points, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(points)
    return model.cluster_centers_

def compute_centroid_distances(centroids):
    return squareform(pdist(centroids))

def plot_elbow(wcss):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(wcss) + 1), wcss, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")
    ax.grid(True)
    return fig

def overlay_centroids(binary_img, centroids):
    fig, ax = plt.subplots()
    ax.imshow(binary_img, cmap='gray')
    if len(centroids) > 0:
        ax.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x', s=60)
    ax.set_title("Centroids Overlay")
    ax.axis('off')
    return fig

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        grayscale_img, binary_img = load_binary_image(uploaded_file)
        points = get_foreground_coords(binary_img)
        
        if len(points) < 2:
            st.error("Image doesn't contain enough white pixels for clustering.")
        else:
            st.subheader("1️⃣ Original and Binary Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(grayscale_img, caption="Grayscale", use_container_width=True, clamp=True)
            with col2:
                st.image(binary_img, caption="Binary", use_container_width=True, clamp=True)

            st.subheader("2️⃣ Elbow Method to Detect Optimal K")
            wcss = compute_wcss(points, max_k=10)
            elbow_fig = plot_elbow(wcss)
            st.pyplot(elbow_fig)

            k = find_elbow_point(wcss)
            st.success(f"Optimal number of clusters: {k}")

            centroids = perform_clustering(points, k)
            dist_matrix = compute_centroid_distances(centroids)

            st.subheader("3️⃣ Final Centroids and Distances")
            for i, c in enumerate(centroids):
                st.write(f"Centroid {i+1}: ({c[0]:.2f}, {c[1]:.2f})")

            st.write("**Pairwise Distance Matrix:**")
            st.dataframe(dist_matrix)

            st.subheader("4️⃣ Centroid Overlay on Image")
            st.pyplot(overlay_centroids(binary_img, centroids))
    except Exception as e:
        st.error(f"Something went wrong: {e}")
