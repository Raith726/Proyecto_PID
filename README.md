# Hybrid Waste Classifier

Clasificación automática de residuos sólidos en **9 clases**
(`battery, cardboard, glass, metal, organic, paper, plastic, textile, trash`)
mediante un **pipeline híbrido en dos etapas**: un clasificador clásico (SVM sobre
características hechas a mano) resuelve los casos de alta confianza y escala a un
**verificador CNN (MobileNetV2)** solo cuando la confianza es baja.

## Resultados (test desbalanceado, 2 328 imágenes)

| Sistema | Accuracy | Macro-F1 |
|---|---|---|
| SVM-RBF solo | 0.780 | 0.728 |
| **Híbrido SVM + CNN** | **0.891** | **0.854** |
| CNN sola (val) | ~0.929 | — |

> **Sobre la accuracy:** el test refleja la distribución real (desbalanceada; `textile`
> es ~47 %), por lo que la accuracy global está algo inflada por la clase mayoritaria. La
> métrica justa es el **macro-F1**, y aun así el híbrido mejora **+0.126** (0.728 → 0.854),
> con mejora en **todas** las clases — la ganancia es real, no un artefacto del desbalance.
>
> **Trade-off del híbrido:** optimiza **costo de cómputo** (solo el 25.8 % de las imágenes
> pasan por la CNN), no la precisión máxima. Una CNN pura (~0.93 val) es más precisa pero
> más cara de ejecutar.

## Pipeline y estructura

```
src/preprocessing/   1_class_mapping.py     fusión 12→9 + split estratificado 70/15/15 (seed 42)
                     2_data_augmentation.py balanceo de train a 750/clase (flip, rot, brillo, zoom, ruido)
                     3_clahe_normalization.py  CLAHE — SOLO DEMO, no se aplica al dataset (ver notas)
src/segmentation/    edge_detection.py      Sobel + Otsu en canales S y V (HSV)
                     morphological_ops.py   open/close + relleno de contornos adaptativo e iterativo
src/classification/  (stubs vacíos — el código real está en los notebooks)
src/evaluation/      (stubs vacíos)
src/features/        (stubs vacíos)
notebooks/01..07     flujo ejecutable completo (exploración → segmentación → features →
                     separabilidad → entrenamiento/ablation → híbrido → evaluación)
results/features/    features_train.csv (6750), features_val.csv (2328), features_test.csv (2328)
results/models/      svm_clasico.pkl, scaler, cnn_verificador.h5, predicciones, history
results/figures/PPT/ figuras listas para presentación
```

**113 características** por imagen (calculadas sobre el objeto segmentado y la imagen completa):
Gabor (48), textura LBP+GLCM/Haralick (28), color HSV/LAB (19), forma + momentos de Hu (13),
intensidad (5).

## Splits (sobre 15 515 imágenes, 12 clases originales → 9)

| Split | Imágenes | Notas |
|---|---|---|
| train | 6 750 | balanceado a 750/clase (submuestreo + augmentation) |
| val | 2 328 | real, sin balancear |
| test | 2 328 | real, sin balancear (battery 142 … textile 1096) |

## Notas de reproducibilidad

- `data/` y `results/figures/` están en `.gitignore` → el **dataset de imágenes no se versiona**.
  Descargarlo de Kaggle (`mostafaabla/garbage-classification`, 12 clases) en `data/raw/`.
- **CLAHE es código muerto:** `3_clahe_normalization.py` define el filtro pero solo lo muestra
  en un demo; **no** procesa ni sobrescribe `data/processed`. El pipeline no usa CLAHE.
- **TensorFlow:** no hay wheel para Python 3.14. Se usa un venv aparte con **Python 3.11**
  (`.venv-tf`). Como TF ≥ 2.16 trae Keras 3 (sin `ImageDataGenerator`), el NB06 corre con
  `tf-keras` + variable de entorno `TF_USE_LEGACY_KERAS=1` (API Keras 2, sin tocar el notebook).
- Los notebooks 04–07 reescriben `results/models/*.pkl` y figuras al ejecutarse.

## Uso

```powershell
# Pipeline clásico (Python 3.x con opencv, scikit-image, scikit-learn, etc.)
pip install -r requirements.txt
# 1) colocar el dataset en data/raw/, luego:
python src/preprocessing/1_class_mapping.py        # 12→9 + split train/val/test
python src/preprocessing/2_data_augmentation.py    # balancea train a 750/clase
# 2) ejecutar notebooks 01 → 05 (segmentación, features, separabilidad, entrenamiento)

# CNN / híbrido (notebooks 06–07) — requiere venv Python 3.11 con TensorFlow:
uv venv --python 3.11 .venv-tf
uv pip install --python .venv-tf\Scripts\python.exe tensorflow tf-keras scikit-learn pandas joblib matplotlib Pillow scipy
$env:TF_USE_LEGACY_KERAS = "1"   # usar API Keras 2
# ejecutar notebooks 06 → 07 con ese intérprete
```
