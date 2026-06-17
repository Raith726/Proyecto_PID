# Guion de defensa — Hybrid Waste Classifier

Documento de preparación para la sustentación. Reúne: (1) el cruce con las PPTs,
(2) los puentes conceptuales para justificar cada técnica, (3) el argumento de "otro
curso" con sus límites, (4) las preguntas difíciles con respuestas, y (5) el guion de apertura.

---

## 1. Apertura sugerida (di esto al empezar)

> *"El proyecto combina tres niveles. El **núcleo es procesamiento de imágenes clásico**
> y cubre lo visto en clase: espacios de color, histogramas, suavizado, Sobel, Otsu,
> morfología y contornos. Sobre eso construimos un **clasificador SVM**, en la misma línea
> del HOG+SVM de la Clase 17. Y añadimos un **verificador CNN** como técnica adicional
> justificada por el problema. Nuestra contribución propia es la **arquitectura decisional
> por confianza**: decidir cuándo confiar en el método clásico barato y cuándo escalar al
> costoso."*

Esto pone como protagonista la **arquitectura de decisión** (defendible con lógica del
curso) y deja la CNN como un componente, no como la estrella.

---

## 2. Lo que SÍ está respaldado por las PPTs (la mitad sólida)

| Tema del proyecto | Clase PPT |
|---|---|
| Espacios de color (RGB/HSV/LAB) | Clase 11 |
| Histogramas y ecualización | Clase 04 |
| Suavizado gaussiano | Clases 06 / 09 |
| Sobel (bordes) | Clases 05 / 06 |
| Umbralización de Otsu | Clase 15 |
| Morfología: apertura / cierre / relleno | Clases 13 / 14 |
| Contornos | Clases 13 / 14 |
| Paradigma "extractor de features + clasificador" (HOG+SVM) | Clase 17 |

---

## 3. Las 3 alarmas y cómo responderlas

### Alarma A — Descriptores de textura (LBP, GLCM, Gabor, entropía) = 76/113 features, sin PPT
**Riesgo: medio. Muy defendible como extensión de lo visto.**

Puente conceptual (úsalo):
- **GLCM / Haralick** = un *histograma de segundo orden* (co-ocurrencia de pares de píxeles). → extiende **histogramas (Clase 04)**.
- **Entropía de Shannon** = escalar que resume un histograma. → **Clase 04**.
- **LBP** = umbralizar el vecindario de cada píxel y armar un histograma. → **umbralización (Otsu, Clase 15)** + **histogramas (04)** aplicados localmente.
- **Gabor** = filtro gaussiano modulado por una sinusoide; es **convolución con un kernel** orientado. → misma maquinaria del **gaussiano (06/09)** y **Sobel (05/06)**.

Frase: *"Ninguno es una técnica ajena: son histogramas de segundo orden y filtros de
convolución orientados, que extienden las Clases 04, 06 y 15 al dominio de la textura."*

Justificación empírica (refuérzalo con datos del proyecto):
- El análisis exploratorio (NB01) mostró que el **color no separa** glass/battery/paper/plastic.
- El **ablation** lo confirma: solo color = 0.537, solo textura = 0.569, todo junto = 0.739.
- → La textura **no es decorativa, es necesaria y está medida**.

### Alarma B — CNN / MobileNetV2 / transfer learning (el salto 78 % → 89 % accuracy; macro-F1 0.728 → 0.854), sin ninguna PPT de redes
**Riesgo: alto. No se puede disfrazar de "visto en clase".**

Estrategia:
1. **Declárala como añadido**, no como núcleo. El entregable clásico (PI + features + SVM)
   se sostiene solo en 78 % (macro-F1 0.728); la CNN solo entra en el **25.8 % de casos difíciles**.
2. Puente conceptual honesto: *"Una CNN aprende automáticamente filtros de convolución en
   sus primeras capas —detectores de bordes y texturas equivalentes a Sobel y Gabor
   (Clases 05/06)— en lugar de diseñarlos a mano. Es la misma idea de convolución del curso,
   llevada al aprendizaje automático."*
3. Si algún integrante vio CNN/DL en otro curso → menciónalo (ver §4).

### Alarma C — SVM-RBF, C/γ, validación cruzada (solo aparece HOG+SVM de pasada en Clase 17)
**Riesgo: bajo.**

- *"Es el mismo pipeline HOG+SVM de la Clase 17, reemplazando HOG por nuestro vector de
  textura/color/forma."*
- Explica en 2 minutos: **RBF** = similitud por distancia entre muestras; **C** = equilibrio
  margen vs. errores; **γ** = alcance de influencia de cada ejemplo; **CV** = estimar el
  desempeño de forma honesta sin sobreajustar.

---

## 4. Argumento "lo vimos en otro curso" (dos integrantes cursaron Machine Learning)

Es legítimo y fuerte, **pero solo para el bloque de ML**. Úsalo con precisión:

**SÍ justifica (úsalo con confianza):** SVM, kernel RBF, C/γ, validación cruzada,
GridSearch, train/test split, PCA, t-SNE, ablation, comparación de modelos.

**NO cubre:**
- **Textura (LBP/GLCM/Gabor/entropía)** → es Visión/PI, no ML. Para esto usa el puente de
  histogramas+filtros (§3-A), **no** "lo vimos en ML".
- **CNN/transfer learning** → solo si el curso de ML de tus compañeros **incluyó deep
  learning**. Si no llegó a redes, mantén la CNN como "adicional autoaprendida" + el puente
  de convolución aprendida (§3-B).

**Cómo decirlo (fortaleza, no excusa):**
- ✅ *"El equipo tiene formación complementaria: dos integrantes cursaron Machine Learning,
  lo que nos permitió integrar el clasificador con rigor —kernel RBF, validación cruzada,
  ablation— en vez de usarlo como caja negra."*
- ❌ *"Usamos SVM porque ya lo sabíamos de otro curso."* (suena a esquivar el temario de PI)

**Advertencia:** revisa la **rúbrica**. Si exigía usar *solo* técnicas del curso, "lo vimos
en otro lado" explica que saben usarlo pero **no cumple la restricción** → en ese caso el peso
recae en los puentes conceptuales (§3), no en el otro curso.

---

## 5. Preguntas difíciles (no son de PPT, prepáralas igual)

**P: ¿El híbrido es mejor que la CNN sola?**
R (honesta, owned): *"No. La CNN sola da ~93 % (val) y el híbrido 89 %. El híbrido **no optimiza
precisión, optimiza costo de cómputo**: solo el 25.8 % de las imágenes pagan la inferencia cara
de la CNN; el resto las resuelve el SVM, que es mucho más liviano."*
⚠️ NO digas "lo mejor de ambos mundos"; si te lo refutan, pierdes credibilidad.

**P: ¿No es trampa decir que mejoró de 78 % a 89 %?**
R: *"La accuracy está inflada porque el test es desbalanceado (textile ~47 %). Por eso reportamos
también el **macro-F1, que sube de 0.728 a 0.854 (+0.126)** y mejora en las 9 clases — esa es la
prueba honesta de que el híbrido ayuda de verdad, no solo en la clase mayoritaria."*

**P: ¿Cómo eligieron el umbral θ=0.60?**
R: *"Es un punto de mejora: la curva de θ se exploró sobre el conjunto de test, cuando debió
fijarse en validación para evitar sesgo. Lo reconocemos como trabajo futuro."* (Honestidad > inventar.)

**P: ¿Su conjunto de test está balanceado / cómo lo construyeron?**
R: *"El test es el split estratificado 70/15/15 real (2 328 imágenes), **desbalanceado a propósito**
para reflejar la distribución real de residuos (textile domina). Por eso usamos macro-F1 como
métrica principal."* (Resuelto: ya no hay inconsistencia con los notebooks.)

**P: ¿Por qué SVM y no el HistGradientBoosting, si dio mejor en CV (0.748 vs 0.739)?**
R: *"El híbrido necesita probabilidades de confianza calibradas para decidir cuándo escalar
a la CNN; el SVM con `predict_proba` se integraba mejor en esa arquitectura."*

**P: ¿Por qué no usar solo la CNN si es la más precisa?**
R: *"Por costo. En un escenario real de planta de reciclaje, correr una CNN en cada imagen es
caro; el cascada clásico→CNN procesa la mayoría con un modelo liviano y reserva la CNN para
los casos ambiguos (metal/vidrio/plástico)."*

**P: ¿Cuáles son los errores principales y por qué?**
R: Las clases con menor F1 tras el híbrido son **plastic (0.765) y metal (0.790)**, por
confusiones en el grupo plastic↔glass↔metal. *"Son materiales física y visualmente ambiguos;
por eso reforzamos con textura multiescala y la verificación CNN."*

---

## 6. Tabla resumen de la estrategia

| Técnica | Argumento principal de defensa |
|---|---|
| SVM, CV, GridSearch, PCA, t-SNE, ablation | ✅ "dos integrantes cursamos ML" (directo) |
| LBP, GLCM, Gabor, entropía de Shannon | 🔗 extensión de histogramas + filtros de convolución (Clases 04/06/15) + ablation que prueba que el color no basta |
| CNN / MobileNetV2 / transfer learning | Si ML incluyó DL → "otro curso"; si no → "técnica adicional autoaprendida" + convolución aprendida (Clases 05/06) |
| Arquitectura híbrida por confianza | 🏆 contribución propia; defendible con lógica del curso (decisión costo↔precisión) |
| PI clásico (color, Sobel, Otsu, morfología, contornos) | ✅ respaldo directo de PPT (Clases 04/05/06/09/11/13/14/15) |

---

## 7. Cierre sugerido

> *"En síntesis: la base del proyecto es procesamiento de imágenes clásico visto en clase;
> el clasificador sigue el paradigma de la Clase 17 con herramientas de ML que el equipo
> domina; y la CNN es un verificador adicional. La decisión de diseño que aporta valor es
> **cuándo usar cada uno**, y eso lo medimos con el ablation y el análisis de fallos."*

### Recordatorio de honestidad
Los puntos débiles reales (híbrido < CNN sola, θ elegido en test, accuracy inflada por el
desbalance → reportar macro-F1) **decláralos tú primero** si vienen al caso. Anticiparlos da
más credibilidad que defenderlos a la defensiva.

### Números finales (test desbalanceado real, 2 328 img) — para tener a mano
- SVM solo: accuracy **0.780**, macro-F1 **0.728**
- Híbrido: accuracy **0.891**, macro-F1 **0.854**, weighted-F1 0.893
- CNN sola (val): ~**0.929** · Enrutamiento: SVM 74.2 % / CNN 25.8 %
- Mejor SVM: C=10, γ='scale' (CV 0.741) · HistGradBoost CV 0.748
