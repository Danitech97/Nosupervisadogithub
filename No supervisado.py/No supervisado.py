# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Importar los datos
data = pd.read_csv("data.csv")

# Limpieza de los datos
data = data.dropna()

# Preprocesamiento de los datos
data["x"] = data["x"].astype(np.float32)
data["y"] = data["y"].astype(np.float32)

# Entrenamiento del modelo
model = KMeans(n_clusters=3)
model.fit(data[["x", "y"]])

# Evaluación del modelo
labels = model.predict(data[["x", "y"]])