"""
Clasificador clasico por reglas (sin machine learning).

Metodo: UMBRALIZACION MULTIDIMENSIONAL. Es la extension directa de la
umbralizacion vista en clase (Clase 15-16, Segmentacion: g(x,y)=1 si f<=T)
a un vector de descriptores. En lugar de un solo umbral sobre la intensidad,
cada clase define un conjunto de bandas [lo, hi] sobre descriptores de:

  - COLOR   (Clase 11 - Imagenes a color): rangos en HSV y CIELAB.
            HSV separa tono/saturacion/brillo; LAB separa los ejes
            oponentes a* (verde-rojo) y b* (azul-amarillo).
  - FORMA   (contornos de la segmentacion + Clase 13-14 Morfologia):
            circularidad, aspect ratio, solidez, extent del objeto.
  - TEXTURA (contraste/homogeneidad GLCM): dispersion local de grises.

Cada regla cumplida suma su peso; la clase con mayor puntaje gana
(desempate por prioridad fija). No hay entrenamiento, ni sklearn, ni datos:
los umbrales son CONSTANTES fijas, derivadas de los cuartiles observados por
clase en el set de entrenamiento y expresadas aqui como reglas legibles.

Las 9 clases: battery, cardboard, glass, metal, organic, paper, plastic,
textile, trash. Todas las features usan la escala de OpenCV
(H in [0,179]; S,V,L,a,b in [0,255]).
"""

INF = float('inf')

# (feature, lo, hi, peso, descripcion legible para la UI)
# Se cumple la regla si  lo <= valor <= hi.
REGLAS = {
    'trash': [
        ('color_v_mean',       185, INF, 3.0, 'V muy alto (>185): objeto/fondo muy claro y brillante'),
        ('int_mean',           165, INF, 2.0, 'intensidad media alta (>165)'),
        ('int_std',           -INF,  50, 1.0, 'poca variacion de intensidad (<50)'),
        ('color_v_std',       -INF,  50, 1.0, 'brillo homogeneo (std V<50)'),
        ('color_s_mean',      -INF,  75, 0.5, 'baja saturacion (<75)'),
    ],
    'battery': [
        ('obj_glcm_contrast',  520, INF, 2.0, 'contraste GLCM muy alto (>520): etiquetas/bordes marcados'),
        ('int_std',             63, INF, 1.5, 'alta variacion de intensidad (>63)'),
        ('color_v_std',         63, INF, 1.5, 'alto contraste de brillo (std V>63)'),
        ('color_s_mean',      -INF,  50, 0.5, 'saturacion baja (<50)'),
    ],
    'organic': [
        ('color_s_mean',       100, INF, 2.2, 'saturacion muy alta (>100): colores vivos'),
        ('lab_b_mean',         146, INF, 1.8, 'b* alto (>146): tonos calidos amarillo/marron'),
        ('obj_glcm_homogeneity', -INF, 0.26, 1.5, 'textura poco homogenea (GLCM<0.26): superficie irregular'),
        ('color_s_std',         58, INF, 1.0, 'saturacion muy variable (std S>58)'),
        ('lab_a_mean',         132, INF, 0.6, 'a* hacia el rojo (>132)'),
    ],
    'cardboard': [
        ('color_h_mean',      -INF,  30, 2.5, 'tono bajo (H<30): marron/naranja del carton'),
        ('color_s_mean',        70, INF, 1.5, 'saturacion alta (>70)'),
        ('lab_b_mean',         138, INF, 1.0, 'b* alto (>138): calido'),
        ('color_v_mean',       130, INF, 0.5, 'brillo medio-alto (>130)'),
    ],
    'paper': [
        ('color_v_mean',       150, INF, 1.3, 'brillo alto (>150): hoja clara'),
        ('int_mean',           140, INF, 1.3, 'intensidad media alta (>140)'),
        ('obj_glcm_contrast',  600, INF, 1.0, 'contraste GLCM alto (>600): texto/arrugas'),
        ('int_std',           -INF,  62, 0.8, 'variacion de intensidad moderada (<62, no tan extrema como pila)'),
        ('color_h_mean',        30,  75, 0.6, 'tono neutro (30-75)'),
        ('color_s_mean',        28,  78, 0.8, 'saturacion media-baja (28-78)'),
    ],
    'plastic': [
        ('color_h_mean',        62, 120, 1.5, 'tono frio (62-120): cian/azul/verde comun en envases'),
        ('color_v_mean',       150, INF, 1.2, 'brillo alto (>150): superficie que refleja'),
        ('obj_glcm_contrast', -INF, 180, 1.0, 'textura lisa (contraste GLCM<180)'),
        ('color_v_std',       -INF,  50, 1.0, 'brillo homogeneo (std V<50)'),
        ('color_s_mean',        25, 105, 0.5, 'saturacion media (25-105)'),
    ],
    'glass': [
        ('color_s_mean',        82, INF, 1.3, 'saturacion alta (>82): vidrio de color'),
        ('forma_aspect_ratio', -INF, 1.0, 1.2, 'objeto igual de alto o mas alto que ancho (aspect<=1.0)'),
        ('obj_glcm_homogeneity', 0.35, 0.6, 1.0, 'textura homogenea (GLCM 0.35-0.6): superficie lisa'),
        ('lab_b_mean',         128, 150, 0.8, 'b* moderado (128-150)'),
        ('color_h_mean',        22,  62, 0.7, 'tono verde/ambar tipico de botellas (22-62)'),
        ('int_std',             45,  72, 0.6, 'variacion de intensidad media (45-72)'),
    ],
    'textile': [
        ('color_v_mean',      -INF, 145, 1.6, 'brillo bajo-medio (<145): tela mate'),
        ('int_mean',          -INF, 132, 1.6, 'intensidad media baja (<132)'),
        ('forma_aspect_ratio', -INF, 1.15, 1.0, 'contorno compacto (aspect<1.15)'),
        ('color_h_mean',        30, 105, 0.8, 'tono amplio no marron (30-105)'),
        ('obj_glcm_homogeneity', 0.18, 0.46, 0.7, 'textura media (GLCM 0.18-0.46): fibras'),
        ('color_s_mean',        35,  98, 0.7, 'saturacion media (35-98)'),
    ],
    'metal': [
        ('color_s_mean',      -INF,  60, 1.4, 'saturacion baja (<60): gris metalico'),
        ('obj_glcm_homogeneity', 0.30, 0.55, 1.0, 'textura homogenea (GLCM 0.30-0.55)'),
        ('forma_solidity',    0.78, INF, 1.0, 'contorno solido/compacto (solidez>0.78)'),
        ('color_v_mean',        95, 165, 0.6, 'brillo medio (95-165)'),
        ('forma_extent',      0.58, INF, 0.6, 'llena bien su caja (extent>0.58)'),
    ],
}

# Orden de prioridad para desempatar cuando dos clases igualan el puntaje.
PRIORIDAD = ['trash', 'battery', 'organic', 'cardboard', 'glass',
             'plastic', 'paper', 'metal', 'textile']

# Orden alfabetico = mismo orden que usa el resto del pipeline (SVM/CNN).
CLASES = sorted(REGLAS)


def _valor(feat, clave):
    """Lee una feature de un dict o de un pandas.Series indistintamente."""
    v = feat.get(clave, 0.0)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = 0.0
    if v != v:  # NaN
        v = 0.0
    return v


def puntuar(feat):
    """Devuelve el dict {clase: puntaje} evaluando todas las reglas."""
    puntajes = {}
    for clase, reglas in REGLAS.items():
        s = 0.0
        for clave, lo, hi, peso, _ in reglas:
            if lo <= _valor(feat, clave) <= hi:
                s += peso
        puntajes[clase] = s
    return puntajes


def razones(feat, clase):
    """Lista las reglas cumplidas por 'clase' (para explicar la decision)."""
    out = []
    for clave, lo, hi, peso, desc in REGLAS[clase]:
        if lo <= _valor(feat, clave) <= hi:
            out.append(desc)
    return out


def probabilidades(puntajes):
    """Normaliza los puntajes a una pseudo-distribucion sobre las 9 clases.

    No son probabilidades calibradas: solo el puntaje relativo de cada regla,
    util para graficar la 'confianza' del clasificador por reglas.
    """
    total = sum(max(0.0, v) for v in puntajes.values())
    if total <= 0:
        return {c: 1.0 / len(CLASES) for c in CLASES}
    return {c: max(0.0, puntajes.get(c, 0.0)) / total for c in CLASES}


def clasificar_reglas(feat):
    """Clasifica un vector de features por reglas.

    Parametros
    ----------
    feat : dict | pandas.Series
        Mapea nombre_de_feature -> valor (mismas claves que el CSV de features).

    Retorna
    -------
    clase : str
    proba : dict {clase: pseudo-probabilidad}
    motivos : list[str]  reglas que gano la clase elegida
    """
    puntajes = puntuar(feat)
    mejor = max(puntajes.values())
    candidatas = [c for c in puntajes if puntajes[c] == mejor]
    clase = next((c for c in PRIORIDAD if c in candidatas), candidatas[0])
    return clase, probabilidades(puntajes), razones(feat, clase)
