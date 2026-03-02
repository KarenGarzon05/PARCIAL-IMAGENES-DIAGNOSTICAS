# PARCIAL-IMAGENES-DIAGNOSTICAS
```mermaid
flowchart TD

A([Inicio]) --> B[Importar librerias]
B --> C[Descomprimir Malignos.zip y Benignos.zip]
C --> D[Obtener lista de archivos con glob]
D --> E[Separar imagenes y mascaras]
E --> F[Definir valor del umbral T]
F --> G[Leer imagen y mascara]
G --> H[Convertir imagen a escala de grises]
H --> I[Redimensionar mascara]
I --> J[Aplicar segmentacion binaria]
J --> K[Calcular F1 Score]
K --> L[Calcular promedio y desviacion]
L --> M[Calcular intensidad promedio del tumor]
M --> N[Ingresar nueva imagen]
N --> O[Segmentar nueva imagen]
O --> P[Calcular area tumoral]
P --> Q[Clasificar como Benigna o Maligna]
Q --> R([Fin])

%% Colores
style A fill:#A2D2FF,stroke:#000,stroke-width:2px
style B fill:#BDE0FE,stroke:#000
style C fill:#CDB4DB,stroke:#000
style D fill:#FFC8DD,stroke:#000
style E fill:#FFAFCC,stroke:#000
style F fill:#FFFFC7,stroke:#000
style G fill:#B9FBC0,stroke:#000
style H fill:#FFD6A5,stroke:#000
style I fill:#E7C6FF,stroke:#000
style J fill:#A0C4FF,stroke:#000
style K fill:#CAFFBF,stroke:#000
style L fill:#FFC6FF,stroke:#000
style M fill:#FDFFB6,stroke:#000
style N fill:#9BF6FF,stroke:#000
style O fill:#FFADAD,stroke:#000
style P fill:#FFD6A5,stroke:#000
style Q fill:#BDB2FF,stroke:#000
style R fill:#A2D2FF,stroke:#000,stroke-width:2px
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
Ahora para la segmentación se empieza probando un umbral T = 80, realizando un for para procesar las 60 imagenes, se convierte la imagen a escala de grises, y se redimensiona la mascara para que tenga el mismo tamaño que la imagen con `  mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))`. Se aplica la segmentación con umbral y se convierten con `flatten` de matrices 2D a vectores 1D para aplicar la función `f1_score`.
```python
T=80
for i in range(len(image_paths)):
  img = cv2.imread(image_paths[i])
  mask = cv2.imread(mask_paths[i], 0)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))


  binary = np.zeros_like(gray)
  binary[gray < T] = 1
  mask = mask // 255

  f1 = f1_score(mask.flatten(), binary.flatten())
  scores.append(f1)

print("Promedio F1:", np.mean(scores))
print("Desviación estándar:", np.std(scores))

```
Ahora para la segmentación se empieza probando un umbral T = 80, realizando un for para procesar las 60 imagenes, se convierte la imagen a escala de grises, y se redimensiona la mascara para que tenga el mismo tamaño que la imagen con `mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))`. Se aplica la segmentación con umbral y se convierten con `flatten` de matrices 2D a vectores 1D para aplicar la función `f1_score`.
```python
T=80
for i in range(len(image_paths)):
  img = cv2.imread(image_paths[i])
  mask = cv2.imread(mask_paths[i], 0)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))


  binary = np.zeros_like(gray)
  binary[gray < T] = 1
  mask = mask // 255

  f1 = f1_score(mask.flatten(), binary.flatten())
  scores.append(f1)

print("Promedio F1:", np.mean(scores))
print("Desviación estándar:", np.std(scores))
```
Promedio F1: 0.19928292879913367
Desviación estándar: 0.196648717723429

Para encontrar el mejor procesamiento se prueban distintos valores de T (umbral) hasta encontrar los mejores valores de F1 y desviación estándar.
Se aplican los umbrales "40, 55 y 60". 
```python
T=40

for i in range(len(image_paths)):
  img = cv2.imread(image_paths[i])
  mask = cv2.imread(mask_paths[i], 0)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))


  binary = np.zeros_like(gray)
  binary[gray < T] = 1
  mask = mask // 255

  f1 = f1_score(mask.flatten(), binary.flatten())
  scores.append(f1)

print("Promedio F1:", np.mean(scores))
print("Desviación estándar:", np.std(scores))
```
Promedio F1: 0.19152608869768337
Desviación estándar: 0.20514189571609265

El umbral de 55 dió los mejores resultados, por lo cual se escogió como el código final. A este codigo se le complementó la visualización de cada imagen segmentada con `matplotlib` y su F1 Score respectivo.
```python
T=55

for i in range(len(image_paths)):
  img = cv2.imread(image_paths[i])
  mask = cv2.imread(mask_paths[i], 0)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))


  binary = np.zeros_like(gray)
  binary[gray < T] = 1
  mask = mask // 255

  f1 = f1_score(mask.flatten(), binary.flatten())
  scores.append(f1)

  plt.imshow(binary, cmap='gray')
  plt.title(f'Imagen segmentada {i+1} - F1: {f1:.2f}')
  plt.axis('off')
  plt.show()

print("Promedio F1:", np.mean(scores))
print("Desviación estándar:", np.std(scores))
```
Promedio F1: 0.19959456226063915
Desviación estándar: 0.19649696088820517

```python
T=60

for i in range(len(image_paths)):
  img = cv2.imread(image_paths[i])
  mask = cv2.imread(mask_paths[i], 0)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))


  binary = np.zeros_like(gray)
  binary[gray < T] = 1
  mask = mask // 255

  f1 = f1_score(mask.flatten(), binary.flatten())
  scores.append(f1)

print("Promedio F1:", np.mean(scores))
print("Desviación estándar:", np.std(scores))
```
Promedio F1: 0.1886185514736362
Desviación estándar: 0.2016104178524424

