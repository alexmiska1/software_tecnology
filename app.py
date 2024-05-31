import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

# Ρυθμίσεις Streamlit
st.set_page_config(page_title="Data Analysis and Visualization App", layout="wide")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload Data", "2D Visualization", "Classification", "Clustering", "Info"])

# 1. Φόρτωση Δεδομένων
with tab1:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write(data)
        if data.shape[1] < 2:
            st.error("The dataset must have at least 2 columns (features + label).")
        else:
            features = data.iloc[:, :-1]
            labels = data.iloc[:, -1]
            st.write("Features")
            st.write(features)
            st.write("Labels")
            st.write(labels)

# 2. 2D Visualization Tab
with tab2:
    st.header("2D Visualization")
    method = st.selectbox('Select method', ['PCA', 't-SNE'])
    if uploaded_file is not None:
        def plot_2d(data, labels, method):
            if method == 'PCA':
                model = PCA(n_components=2)
            elif method == 't-SNE':
                model = TSNE(n_components=2)
            reduced_data = model.fit_transform(data)
            df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
            df['Label'] = labels
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Component 1', y='Component 2', hue='Label', data=df)
            st.pyplot(plt)
        plot_2d(features, labels, method)
    st.write("Exploratory Data Analysis (EDA)")
    if uploaded_file is not None:
        st.write(sns.pairplot(data, hue=data.columns[-1]))
        st.pyplot()

# 3. Classification Tab
with tab3:
    st.header("Classification Algorithms")
    if uploaded_file is not None:
        k = st.slider("Select k for k-NN", 1, 10, 3)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # k-NN Classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        st.write("k-NN Classification Report")
        st.text(classification_report(y_test, knn_pred))
        
        # RandomForest Classifier
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        st.write("Random Forest Classification Report")
        st.text(classification_report(y_test, rf_pred))

# 4. Clustering Tab
with tab4:
    st.header("Clustering Algorithms")
    if uploaded_file is not None:
        k = st.slider("Select k for k-Means", 1, 10, 3)
        
        # k-Means Clustering
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(features)
        if len(set(clusters)) > 1:
            st.write("k-Means Silhouette Score")
            st.text(silhouette_score(features, clusters))
        else:
            st.write("k-Means did not create enough clusters to compute the silhouette score.")
        
        # DBSCAN Clustering
        dbscan = DBSCAN()
        clusters = dbscan.fit_predict(features)
        if len(set(clusters)) > 1:
            st.write("DBSCAN Silhouette Score")
            st.text(silhouette_score(features, clusters))
        else:
            st.write("DBSCAN did not create enough clusters to compute the silhouette score.")

# 5. Info Tab
with tab5:
    st.header("Application Information")
    st.write("""
    This application is developed for data analysis and visualization.
    - Upload your tabular data in CSV or Excel format.
    - Visualize your data using 2D reduction methods such as PCA or t-SNE.
    - Perform classification using k-NN and Random Forest.
    - Perform clustering using k-Means and DBSCAN.
    Developed by: [Your Team Name]
    """)
