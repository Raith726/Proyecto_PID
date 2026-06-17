# Revisión del proyecto "Hybrid Waste Classifier" — contraste con las PPTs del curso

> **Para Claude Desktop:** tú tienes acceso a **todas las presentaciones (PPTs) del
> curso de Procesamiento de Imágenes**. Yo (Claude Code) analicé el repositorio completo
> (clonado de GitHub: `Cisar4321/hybrid-waste-classifier`) y aquí dejo el **inventario
> exhaustivo de cada técnica, método y función** que usa el proyecto. Tu tarea:
>
> 1. **Recorrer las PPTs** y marcar, en la columna *"¿Visto en clase?"* de cada tabla,
>    si el tema fue cubierto (✅ sí / ⚠️ parcial / ❌ no), citando la(s) diapositiva(s)
>    o título de la clase.
> 2. **Listar al final** todo lo que el proyecto usa pero **NO** aparece en las PPTs
>    (riesgo si el profesor exige usar solo lo visto en clase).
> 3. **Validar / ampliar** la sección "Cómo mejorarlo".
>
> Mi columna *"Probablemente fuera de un curso clásico de PI"* es una **hipótesis mía**;
> **manda la PPT**, no mi suposición. Corrígeme donde las diapositivas digan otra cosa.

---

## 0. Resumen del proyecto

Clasifica residuos en 9 clases (`battery, cardboard, glass, metal, organic, paper,
plastic, textile, trash`). Hipótesis central: el color/textura clásicos no separan bien
vidrio/metal/plástico/papel, así que se usa un **pipeline híbrido**: un **SVM** sobre
**113 características hechas a mano** resuelve los casos de alta confianza, y solo escala a
una **CNN MobileNetV2** cuando la confianza del SVM < 0.60. Sobre el **test desbalanceado real
(2 328 img)**: SVM solo = **78.0 %** (macro-F1 0.728); híbrido = **89.1 %** (macro-F1 0.854,
+0.126); CNN sola ≈ **92.9 %** en validación. *(La accuracy global está algo inflada porque
`textile` es ~47 % del test; el macro-F1 es la métrica justa y confirma la mejora en todas las clases.)*

**Datos:** 15 515 imágenes originales (12 clases) → fusión a 9 clases → train balanceado a
750/clase (6 750), val 2 328, test 2 328 (real, sin balancear).

**Estructura real del repo** (lo confirmé tras clonar):
- `src/preprocessing/` → preparación de datos (mapeo de clases, augmentation, CLAHE).
- `src/segmentation/` → bordes (Sobel/Otsu) y morfología (relleno adaptativo).
- `src/classification/`, `src/evaluation/`, `src/features/` → **stubs vacíos (0 bytes)**:
  ese código vive realmente en los notebooks `04`–`07`.
- `notebooks/01`–`07` → flujo ejecutable completo. `data/` (imágenes) está en `.gitignore`.

---

## 1. Preprocesamiento, espacios de color y preparación de datos

| # | Técnica / función | Dónde (archivo) | Para qué | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|---|---|
| 1.1 | Conversión **RGB↔BGR↔HSV↔LAB** (`cv2.cvtColor`) | segmentation, preprocessing, NB03 | base de todo | No | |
| 1.2 | **Resize** 256×256 interpolación cúbica | `3_clahe_normalization.py` | normalizar tamaño | No | |
| 1.3 | **CLAHE** sobre canal L de LAB (`createCLAHE`, clip=2.0, tile 8×8) — ⚠️ **definido pero NO aplicado al dataset (código muerto / solo demo)** | `3_clahe_normalization.py` | normalizar contraste (no usado) | A veces no | |
| 1.4 | **Fusión de clases 12→9** (diccionario de mapeo) | `1_class_mapping.py` | unir clothes+shoes→textile, vidrios→glass, etc. | No (decisión de dataset) | |
| 1.5 | **Split estratificado** train/val/test 70/15/15 (`train_test_split`, seed 42) | `1_class_mapping.py` | partición reproducible | ⚠️ ML, a veces no | |
| 1.6 | **Data augmentation**: flip horizontal, rotación ±15°, brillo ×0.8–1.2, zoom 0.85–1.0, **ruido gaussiano** (σ=8) | `2_data_augmentation.py` | balancear train a 750/clase | A veces no | |
| 1.7 | **Balanceo de clases** (submuestreo si >750, augmentation si <750) | `2_data_augmentation.py` | corregir desbalance ~10× | ⚠️ concepto de ML | |
| 1.8 | **Suavizado Gaussiano** (`GaussianBlur`) | segmentation, NB03 | reducir ruido | No | |
| 1.9 | **Filtro bilateral** condicional según densidad de bordes (`bilateralFilter`) | NB02, NB03 | pasa-bajos que preserva bordes | El bilateral a veces no se ve | |
| 1.10 | Histogramas de color RGB por clase | NB01 | análisis exploratorio | No | |

## 2. Detección de bordes y segmentación  (`src/segmentation/edge_detection.py`)

| # | Técnica / función | Detalle real del código | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|---|
| 2.1 | **Gradiente de Sobel** + magnitud (`cv2.Sobel` ksize=7, `cv2.magnitude`) | Sobel **primero**, normaliza, luego Gaussiano, luego Otsu | No (tema central de bordes) | |
| 2.2 | **Umbralización de Otsu** (`THRESH_OTSU`) | binariza la magnitud de bordes | No | |
| 2.3 | **Densidad de bordes** (% píxeles de borde) | decide si aplicar el filtro bilateral | No (derivado) | |
| 2.4 | Combinación de bordes S∨V (`bitwise_or`) | fusiona evidencia de Saturación y Valor (HSV) | No | |

## 3. Morfología matemática  (`src/segmentation/morphological_ops.py`)

| # | Técnica / función | Detalle real del código | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|---|
| 3.1 | **Apertura** y **Cierre** (`MORPH_OPEN`, `MORPH_CLOSE`) con elementos estructurantes elípticos | limpiar ruido y cerrar contornos | No (tema central) | |
| 3.2 | **Kernels adaptativos** (`kernels_adaptativos`) | tamaño de kernel calculado según el **área del objeto** (componentes conexas) | ⚠️ no estándar, es lógica propia | |
| 3.3 | **Componentes conexas** (`connectedComponentsWithStats`) | medir área del objeto principal | A veces no | |
| 3.4 | **Relleno de contornos** (`findContours RETR_EXTERNAL`, `drawContours FILLED`, filtro `min_area=500`) | máscara sólida del objeto | No | |
| 3.5 | **Relleno iterativo con criterio de éxito** (`relleno_exitoso`, hasta 4 intentos agrandando el cierre) | reintenta si el relleno < 1.5× el área de bordes | ⚠️ heurística propia, no de clase | |
| 3.6 | Enmascarado/recorte del objeto (`bitwise_and`, `copyMakeBorder` con padding) | aislar el objeto | No | |

## 4. Descriptores de TEXTURA  (NB03)

| # | Técnica / función | Detalle | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|---|
| 4.1 | **LBP — Local Binary Patterns** (`local_binary_pattern`, P=8, R=1, "uniform", 10 bins) | textura micro-local | A veces sí (clase de textura) | |
| 4.2 | **GLCM / Haralick** (`graycomatrix` dist=1, 4 ángulos; `graycoprops`: contrast, homogeneity, energy, correlation) | textura estadística de 2º orden | A veces sí | |
| 4.3 | **Filtros de Gabor** (`gabor_kernel`, 3 frecuencias × 4 orientaciones = 12 kernels; media+std) — **48 features, la familia más grande** | textura/orientación multiescala | **A veces NO** (Gabor suele ser avanzado) | |
| 4.4 | **Entropía de Shannon** (`shannon_entropy`) | desorden de intensidad | A veces no | |

## 5. Descriptores de FORMA  (NB03)

| # | Técnica / función | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|
| 5.1 | Área, perímetro (`contourArea`, `arcLength`) | No | |
| 5.2 | **Extent, Solidity** (con `convexHull`) | A veces no | |
| 5.3 | **Aspect ratio, Circularidad** (4πA/P²) | No | |
| 5.4 | **Momentos de Hu** (`HuMoments`, 7 invariantes en escala log) | A veces sí (clase de momentos) | |

## 6. Descriptores de COLOR / INTENSIDAD  (NB03)

| # | Técnica / función | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|
| 6.1 | Estadísticos HSV: media, std, mediana, percentiles 25/75 | No | |
| 6.2 | Estadísticos LAB (a, b) media/std | A veces no | |
| 6.3 | Intensidad: media, std, **skewness, kurtosis** (`scipy.stats`) | Skew/kurtosis a veces no | |

## 7. Reducción de dimensión / análisis del espacio de features  (NB04)

| # | Técnica / función | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|
| 7.1 | **Estandarización** z-score (`StandardScaler`, fit solo en train) | No | |
| 7.2 | **PCA** (2 componentes) | PCA suele verse | |
| 7.3 | **t-SNE** (perplexity=30) | **Probablemente NO** (ML/visualización) | |

## 8. Clasificadores (Machine Learning)  (NB04/05/06)

| # | Modelo / técnica | Resultado | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|---|
| 8.1 | **SVM-RBF** (`SVC`) — modelo clásico final | test 0.780 (macro-F1 0.728) | **Probablemente NO** (es de ML) | |
| 8.2 | RandomForest, kNN, LogisticReg, **HistGradientBoosting**, **MLP** (comparación) | HGB mejor en CV (0.748) | **Probablemente NO** | |
| 8.3 | **GridSearchCV** + **validación cruzada 5-fold** | tuneo C=10, γ='scale' (CV 0.741) | Probablemente NO | |
| 8.4 | **predict_proba** + umbral de confianza θ para enrutar al verificador | gate del híbrido | Probablemente NO | |

## 9. Deep Learning (verificador CNN)  (NB06)

| # | Técnica / función | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|
| 9.1 | **CNN MobileNetV2** + **transfer learning** (ImageNet, base congelada) | **Casi seguro NO** (DL avanzado) | |
| 9.2 | GlobalAveragePooling, Dropout(0.3), Dense softmax, Adam, categorical crossentropy | **Casi seguro NO** | |
| 9.3 | `ImageDataGenerator`, `flow_from_directory` | **Casi seguro NO** | |
| 9.4 | **Pipeline híbrido en cascada** (clásico → DL por confianza): test 0.780 → **0.891** (macro-F1 0.728 → 0.854); CNN resuelve el 25.8 % de baja confianza | **Casi seguro NO** (arquitectura propia) | |

## 10. Evaluación  (NB04–07)

| # | Técnica / función | Probablemente fuera de PI clásico | ¿Visto en clase? |
|---|---|---|---|
| 10.1 | Accuracy, **Precision/Recall/F1** por clase, `classification_report` | A veces sí | |
| 10.2 | **Matriz de confusión** (`confusion_matrix`) | A veces sí | |
| 10.3 | **Ablation study** por familia de features | Probablemente NO (concepto de ML) | |
| 10.4 | Curva accuracy vs umbral θ; tasa de activación del CNN por clase; análisis de fallos | Probablemente NO | |

---

## 11. Mi hipótesis de "lo que NO suele verse en un curso clásico de PI"
*(Desktop: confirma o desmiente con las PPTs)*

- **SVM, RandomForest, HistGradientBoosting, MLP, kNN, GridSearch, cross-validation,
  train_test_split, balanceo de clases** → de un curso de Machine Learning, no de PI.
- **CNN / MobileNetV2 / transfer learning / Keras / ImageDataGenerator** → Deep Learning.
  Lo más probable que el profesor señale como "no visto".
- **t-SNE** → visualización de ML.
- **Filtros de Gabor** → depende; en muchos cursos se mencionan pero no se aplican.
- **CLAHE** → ecualización adaptativa; puede no haberse cubierto (revisar PPT de histogramas).
- **Skewness / kurtosis / entropía de Shannon** como features → estadística avanzada.
- **Espacio LAB** → puede no haberse visto si solo cubrieron RGB/HSV/escala de grises.
- **Kernels morfológicos adaptativos + relleno iterativo** → heurística propia, no estándar.

Lo que **sí** es núcleo de PI y casi seguro está en las PPTs: conversión de espacios de
color, suavizado gaussiano, **Sobel**, **Otsu**, **morfología (open/close, elementos
estructurantes)**, **contornos**, **histogramas**, **LBP/GLCM/Haralick (textura)**,
**momentos de Hu (forma)**.

---

## 12. Cómo mejorarlo y hacerlo "más fino"

### A) Correcciones / verificaciones (problemas reales detectados en el repo)

1. **CLAHE = código muerto (CONFIRMADO).** `3_clahe_normalization.py` define el CLAHE pero su
   `__main__` es solo un *demo*; no procesa ni sobrescribe `data/processed` (se verificó: el
   conteo de archivos y las fechas de modificación no cambian al ejecutarlo). **El pipeline no
   usa CLAHE.** Mejora pendiente: aplicarlo de verdad (añadir un bucle de procesamiento) y medir
   si sube la separabilidad de textura/color.
2. **Split resuelto (RESUELTO).** Se regeneró todo el pipeline sobre el split real
   **estratificado 70/15/15** (`1_class_mapping.py`): train 6 750 balanceado, val 2 328,
   test **2 328 desbalanceado** (battery 142 … textile 1096). Los resultados nuevos (SVM 0.780,
   híbrido 0.891) son sobre ese test real. *(Los viejos 0.693/0.842 eran sobre un test balanceado
   de 7 096 de una corrida anterior, no comparables directamente.)*
3. **Elegir θ en validación, no en test.** Hoy θ=0.60 está fijo y la curva θ se explora sobre
   el set de test (NB07) → sesgo optimista. Moverlo a `val`.
4. **Calibrar las probabilidades del SVM** (`CalibratedClassifierCV`, Platt/isotónica): el
   umbral de confianza solo es fiable si las probabilidades están bien calibradas.
5. **Rellenar los stubs vacíos** (`src/classification`, `src/evaluation`, `src/features`):
   hoy el código vive en notebooks; modularizarlo mejora mantenibilidad y reproducibilidad.

### B) Mejoras DENTRO del temario clásico de PI

6. **Segmentación más robusta**: añadir **GrabCut** o **watershed con marcadores** y comparar
   contra la actual (Sobel+Otsu+morfología), que falla con fondos no uniformes u objetos al borde.
7. **Histogramas de color como features** (binned, 16 bins/canal HSV) y **momentos de color**
   (1º/2º/3º) — hoy solo media/std/percentiles. El ablation mostró "solo color" = 0.54.
8. **LBP multiescala / rotation-invariant** (R=1,2,3) y **GLCM a varias distancias** (d=1,2,4)
   — hoy GLCM solo usa d=1.
9. **Descriptores de Fourier de contorno** o **momentos de Zernike** para forma.
10. **Selección de características**: 113 features con mucha redundancia (Gabor = 48). Aplicar
    **información mutua / RFE / quitar features correlacionadas** o **LDA** antes del SVM.

### C) Mejoras avanzadas (suben el techo, requieren temas extra)

11. **Fine-tuning de la CNN** (descongelar capas superiores con LR bajo): +3–5 pts típicos.
12. **Backbone más fuerte** (EfficientNet-B0/B3) + augmentation en entrenamiento + class weights.
13. **Features profundas + SVM**: usar los *embeddings* de MobileNet (1280-D) como features,
    o **concatenar features hechas a mano + profundas**.
14. **Clasificadores binarios "desempate"** para los pares más confundidos
    (plastic↔glass↔metal, paper↔cardboard).
15. **Test-Time Augmentation** y **ensemble/stacking** (SVM + CNN + HGB) con meta-clasificador.
16. **Opción de abstención** cuando ni SVM ni CNN superan un umbral (útil en reciclaje real).

### Aviso clave para la defensa
El **híbrido (89.1 %) es PEOR que la CNN sola (≈92.9 % val)**. Su única justificación es el
**costo de cómputo** (solo el 25.8 % de las imágenes pasan por la CNN). Conviene declararlo
así explícitamente: es un trade-off precisión↔cómputo, no "lo mejor de ambos mundos".
Además, **no presentar "0.780 → 0.891" como salto de accuracy sin contexto**: la métrica
honesta sobre el test desbalanceado es el macro-F1 (**0.728 → 0.854, +0.126**), que sí
demuestra mejora real en todas las clases.

### Confusiones a atacar
Clases más débiles tras el híbrido (F1 más bajos): **plastic (0.765)** y **metal (0.790)**,
con confusiones residuales en el grupo `plastic↔glass↔metal`. Materiales física y visualmente
ambiguos: aquí rinden más la forma/textura multiescala y el fine-tuning de la CNN.

---

## 13. Pregunta concreta para Desktop

1. Rellena la columna *"¿Visto en clase?"* de las tablas 1–10 citando la PPT/diapositiva.
2. Devuélveme la lista de **"técnicas usadas que NO están en las PPTs"**, ordenadas por riesgo
   (imprescindibles para el proyecto vs. accesorias que podríamos quitar/justificar).
3. De la sección 12, dime **cuáles mejoras caen dentro del temario** y cuáles exigen temas no vistos.
4. Si alguna PPT cubre una técnica mejor o distinta a la usada, propón reemplazarla por la del curso.
