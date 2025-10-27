
# Taller de Clustering - UNINCCA

**Cómo ejecutar localmente**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Cómo desplegar en Streamlit Community Cloud**

1. Sube `app.py`, `requirements.txt` y `datos_clientes.csv` a un repositorio público de GitHub.
2. Ve a https://share.streamlit.io , conecta con tu repo y selecciona `app.py` como archivo principal.
3. Asegúrate de incluir `datos_clientes.csv` en la raíz del repo (o usa la carga por interfaz en la app).

La app permite:
- Cargar el CSV por defecto o subir otro CSV.
- Elegir variables numéricas y categóricas.
- Estandarizar variables (opcional).
- Ejecutar **K-Means** (con búsqueda automática de K por *silhouette*) o **Jerárquico**.
- Ver centroides, perfiles y distribuciones por clúster.
- Visualizar clústeres en dispersión 2D y PCA.
- Descargar el CSV con la etiqueta de clúster.
