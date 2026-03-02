# PARCIAL-IMAGENES-DIAGNOSTICAS
```mermaid
flowchart TD

A([Inicio]) --> B[Importar librerías<br>cv2, numpy, glob, sklearn]
B --> C[Cargar imágenes<br>cv2.imread + glob]
C --> D[Preprocesamiento<br>Escala de grises<br>Filtrado / Umbral]
D --> E[Segmentación<br>Separar células del fondo]
E --> F[Detección / Conteo<br>Extraer contornos]
F --> G[Calcular F1 Score<br>Comparar con Ground Truth]
G --> H[Mostrar resultados<br>Imágenes + Métricas]
H --> I([Fin])

%% Colores
style A fill:#A2D2FF,stroke:#000,stroke-width:2px
style B fill:#BDE0FE,stroke:#000
style C fill:#CDB4DB,stroke:#000
style D fill:#FFC8DD,stroke:#000
style E fill:#FFAFCC,stroke:#000
style F fill:#FFFFC7,stroke:#000
style G fill:#B9FBC0,stroke:#000
style H fill:#FFD6A5,stroke:#000
style I fill:#A2D2FF,stroke:#000,stroke-width:2px
```


Para empezar se importaron las librerias y funciones necesarias
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import f1_score
from google.colab.patches import cv2_imshow
```

despues descomprimimos las carpetas que contienen las imagenes de los tumores beningnos, malignos y sus respectivas mascaras.
```python
!unzip Malignos.zip
!unzip Benignos.zip
```
Aqui se buscan los archivos .png y se separan entre mascara e imagen. Después imprime la cantidad de imagenes y de mascaras para comprobar que se hayan filtrado adecuadamente. Al final se crea una lista vacia para después guardar los datos de los F1 Scores.
```python
all_files = sorted(glob.glob("*.png"))
image_paths = [f for f in all_files if "_mask" not in f]
mask_paths  = [f for f in all_files if "_mask" in f]
print("Imágenes:", len(image_paths))
print("Máscaras:", len(mask_paths))

scores = []

```
