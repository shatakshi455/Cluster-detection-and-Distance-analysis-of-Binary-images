#  K-Means Clustering on Binary Images

This is a simple Streamlit app that performs **K-Means clustering** on the **white (foreground) pixels** of a binary image. It allows users to upload an image, binarizes it, finds the optimal number of clusters using the **elbow method**, and visualizes the centroids on the image.

##  Features

-  Drag & drop image upload (PNG/JPG)
-  Grayscale conversion + binary thresholding
-  K-Means clustering on white pixels
-  Elbow method to estimate optimal number of clusters (WCSS plot)
-  Centroid visualization on image
-  Distance matrix between centroids

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shatakshi455/kmeans-clustering.git
cd kmeans-clustering
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the App
```bash
streamlit run kmeans.py
```
Application opens in your browser.

##  Notes
Best used on clean grayscale or black-and-white sketches, scans, or handwritten inputs.

Works on .png, .jpg, .jpeg

Threshold for binarization is fixed at 128 for simplicity.

