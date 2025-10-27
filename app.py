
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering de Clientes - Taller", layout="wide")

st.title("Taller: Algoritmos de Clúster sobre Datos de Clientes")
st.caption("MINERÍA DE DATOS - UDEC")
st.caption("Fabian Valero - Esteban Fonseca / Docente: XIMENA ACOSTA")

@st.cache_data
def load_default_data():
    # Carga robusta para CSV con ; y codificación Latin-1
    df = pd.read_csv("datos_clientes.csv", encoding="latin-1", sep=";")
    # Limpieza menor de textos
    if "Rango_Ingresos_COP" in df.columns:
        df["Rango_Ingresos_COP"] = df["Rango_Ingresos_COP"].str.replace("mÃ¡s", "más", regex=False)
    return df

def load_any_csv(file):
    try:
        return pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
    except Exception:
        file.seek(0)
        try:
            return pd.read_csv(file, encoding="latin-1", sep=None, engine="python")
        except Exception:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1", sep=";")

with st.sidebar:
    st.header("Datos")
    src = st.radio("Fuente de datos", ["Usar archivo por defecto", "Subir CSV"])
    if src == "Subir CSV":
        up = st.file_uploader("Carga tu CSV", type=["csv"])
        if up is not None:
            df = load_any_csv(up)
        else:
            st.info("Cargaré el archivo por defecto hasta que subas uno.")
            df = load_default_data()
    else:
        df = load_default_data()

    st.divider()
    st.header("Parámetros del modelo")
    alg = st.selectbox("Algoritmo", ["K-Means", "Jerárquico (Agglomerative)"])
    scale = st.checkbox("Estandarizar variables numéricas (recomendado)", value=True)
    auto_k = st.checkbox("Elegir K automáticamente por *silhouette*", value=True)
    k = st.slider("Número de clústeres (K)", 2, 8, 2, disabled=auto_k)
    linkage = st.selectbox("Vinculación (jerárquico)", ["ward", "complete", "average", "single"], disabled=(alg!="Jerárquico (Agglomerative)"))
    st.caption("Distancia euclidiana (por defecto en K-Means y *ward*).")

st.subheader("Vista previa de los datos")
st.dataframe(df.head(10), use_container_width=True)

# Selección de variables
numeric_cols_guess = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols_guess = [c for c in df.columns if c not in numeric_cols_guess]

with st.expander("Selección de variables"):
    num_cols = st.multiselect("Variables numéricas para el clúster", options=numeric_cols_guess, default=[c for c in numeric_cols_guess if c not in ["ID"]])
    cat_cols = st.multiselect("Variables categóricas para análisis descriptivo", options=cat_cols_guess, default=[c for c in ["Genero","Ciudad","Segmento","Rango_Ingresos_COP"] if c in df.columns])

if len(num_cols) < 2:
    st.warning("Elige al menos 2 variables numéricas para ejecutar el clúster.")
    st.stop()

X = df[num_cols].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X) if scale else X

# Selección / validación de K
def choose_k(Xs):
    best_k, best_s = None, -1
    silscores = {}
    for kk in range(2, 7):
        km = KMeans(n_clusters=kk, random_state=42, n_init=20)
        labels = km.fit_predict(Xs)
        s = silhouette_score(Xs, labels)
        silscores[kk] = s
        if s > best_s:
            best_k, best_s = kk, s
    return best_k, best_s, silscores

if alg == "K-Means":
    if auto_k:
        k, best_s, silscores = choose_k(Xs)
        st.info(f"K óptimo por *silhouette*: **{k}** (score={best_s:.3f})")
        st.write(pd.DataFrame.from_dict(silscores, orient="index", columns=["silhouette"]).rename_axis("k"))
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(Xs)
    df_out = df.copy()
    df_out["Cluster"] = labels
    # Centroides en escala original
    centers = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers) if scale else centers
    centroids_df = pd.DataFrame(centers_orig, columns=num_cols)
    centroids_df.index.name = "Cluster"

    st.subheader("Resultados - K-Means")
    st.write("**Centroides (escala original de las variables):**")
    st.dataframe(centroids_df.round(2))

else:
    # Jerárquico
    if linkage == "ward":
        # ward requiere distancia euclidiana y datos estandarizados es recomendable
        if not scale:
            st.warning("Para 'ward' se recomienda marcar 'Estandarizar variables'.")
    if auto_k:
        st.info("Para jerárquico, fija manualmente K con el control deslizante.")
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(Xs)
    df_out = df.copy()
    df_out["Cluster"] = labels

    st.subheader("Resultados - Jerárquico (Agglomerative)")
    st.write("**Clusters asignados:** Muestra de la tabla con etiqueta de clúster.")
    st.dataframe(df_out.head(15))

# Perfilado por clúster
st.subheader("Perfilado de clústeres")
desc = df_out.groupby("Cluster")[num_cols].agg(["mean","median","min","max","count"]).round(2)
st.dataframe(desc, use_container_width=True)

if len(cat_cols) > 0:
    st.write("**Distribuciones por variables categóricas (proporciones):**")
    for col in cat_cols:
        dist = df_out.groupby("Cluster")[col].value_counts(normalize=True).unstack(fill_value=0).round(3)
        st.write(f"Variable: **{col}**")
        st.dataframe(dist)

# Visualizaciones
st.subheader("Visualizaciones")
col1, col2 = st.columns(2)

with col1:
    st.write("**Dispersión 2D (elige ejes)**")
    xvar = st.selectbox("Eje X", num_cols, index=min(1, len(num_cols)-1))
    yvar = st.selectbox("Eje Y", num_cols, index=min(2, len(num_cols)-1))
    fig, ax = plt.subplots()
    for c in np.unique(df_out["Cluster"]):
        m = df_out["Cluster"] == c
        ax.scatter(df_out.loc[m, xvar], df_out.loc[m, yvar], label=f"Cluster {c}")
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.write("**PCA (2 componentes principales)**")
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)
    fig2, ax2 = plt.subplots()
    for c in np.unique(df_out["Cluster"]):
        m = df_out["Cluster"] == c
        ax2.scatter(X2[m,0], X2[m,1], label=f"Cluster {c}")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    st.pyplot(fig2)
    st.caption(f"Varianza explicada: PC1={pca.explained_variance_ratio_[0]:.2f}, PC2={pca.explained_variance_ratio_[1]:.2f}")

# Descarga
st.subheader("Descargas")
st.download_button("Descargar CSV con etiqueta de clúster", data=df_out.to_csv(index=False).encode("utf-8"), file_name="clientes_clusterizados.csv", mime="text/csv")

st.caption("Construido con scikit-learn y Streamlit.")
