"""SafeForm AI — app de evaluacion biomecanica.

Se ejecuta igual en Colab (Seccion 3.3 del notebook de entrenamiento) y como
Hugging Face Space. Un modelo por ejercicio, entrenado en ese notebook.
"""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import keras
import mediapipe as mp
import numpy as np
import requests
import tensorflow as tf

BASE = Path(__file__).parent
SEQ_LEN, NUM_LANDMARKS, FEATURE_SIZE = 64, 33, 133
POSE_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
                  'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task')


# ---------------- Capas custom (identicas a las del notebook) ----------------
# Se resuelven al cargar via CUSTOM_OBJECTS, igual que en el notebook: no se
# registran con keras.saving.register_keras_serializable porque los modelos se
# guardaron sin ese registro y la ruta por custom_objects es la ya verificada.
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1, use_bias=True, name='attention_score')

    def call(self, inputs):
        scores = self.score_dense(inputs)
        weights = keras.ops.softmax(scores, axis=1)
        context = keras.ops.sum(inputs * weights, axis=1)
        return context, keras.ops.squeeze(weights, axis=-1)

    def get_config(self):
        return super().get_config()


class SpatialGraphConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, adjacency, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self._adjacency_init = np.asarray(adjacency, dtype=np.float32)

    def build(self, input_shape):
        self.adjacency = tf.constant(self._adjacency_init, name='adjacency')
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(shape=(in_channels, self.out_channels),
                                       initializer='glorot_uniform', trainable=True,
                                       name='gconv_kernel')
        self.bias = self.add_weight(shape=(self.out_channels,), initializer='zeros',
                                     trainable=True, name='gconv_bias')
        super().build(input_shape)

    def call(self, inputs):
        aggregated = keras.ops.einsum('vw,btwc->btvc', self.adjacency, inputs)
        return keras.ops.einsum('btvc,cd->btvd', aggregated, self.kernel) + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({'out_channels': self.out_channels,
                        'adjacency': self._adjacency_init.tolist()})
        return config

    @classmethod
    def from_config(cls, config):
        adjacency = np.array(config.pop('adjacency'), dtype=np.float32)
        return cls(adjacency=adjacency, **config)


CUSTOM_OBJECTS = {'AttentionPooling': AttentionPooling, 'SpatialGraphConv': SpatialGraphConv}


# ---------------- Extraccion de landmarks (identica al notebook) ----------------
def _ensure_pose_model():
    destino = BASE / 'pose_landmarker_lite.task'
    if not destino.exists() or destino.stat().st_size < 1_000_000:
        with requests.get(POSE_MODEL_URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with open(destino, 'wb') as salida:
                for bloque in response.iter_content(4 * 1024 * 1024):
                    if bloque:
                        salida.write(bloque)
    return destino


_pose_detector = None


def get_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        opciones = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_ensure_pose_model())),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.45, min_pose_presence_confidence=0.45,
            output_segmentation_masks=False)
        _pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(opciones)
    return _pose_detector


def resize_for_pose(frame, max_side=640):
    height, width = frame.shape[:2]
    escala = min(1.0, max_side / max(height, width))
    if escala < 1.0:
        frame = cv2.resize(frame, (round(width * escala), round(height * escala)),
                            interpolation=cv2.INTER_AREA)
    return frame


def pose_result_to_features(result):
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return np.full(FEATURE_SIZE, np.nan, dtype=np.float32)
    image_landmarks = result.pose_landmarks[0]
    world_landmarks = result.pose_world_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in world_landmarks], dtype=np.float32)
    visibility = np.array([getattr(p, 'visibility', 0.0) or 0.0 for p in image_landmarks],
                           dtype=np.float32)
    hip_center = (coords[23] + coords[24]) / 2.0
    shoulder_center = (coords[11] + coords[12]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    shoulder_width = float(np.linalg.norm(coords[11] - coords[12]))
    escala = max(torso_length, shoulder_width, 1e-3)
    coords = (coords - hip_center) / escala
    return np.concatenate([coords.reshape(-1), visibility, np.array([1.0], dtype=np.float32)])


def pose_result_to_features_aligned(result, active_landmarks):
    """Igual que la anterior, pero restringida a los landmarks con los que se
    entreno el modelo. Los datasets de captura (Vicon, Kinect) solo aportan un
    subconjunto de los 33 de MediaPipe: si en inferencia se le pasan los 33 con
    valores reales, ~40% del tensor satura al normalizar y la prediccion pierde
    todo significado. Ver Seccion 3.0.b del notebook."""
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return np.full(FEATURE_SIZE, np.nan, dtype=np.float32)
    world_landmarks = result.pose_world_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in world_landmarks], dtype=np.float32)

    mascara = np.zeros(NUM_LANDMARKS, dtype=bool)
    mascara[list(active_landmarks)] = True
    coords[~mascara] = 0.0

    hip_center = (coords[23] + coords[24]) / 2.0
    shoulder_center = (coords[11] + coords[12]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    shoulder_width = float(np.linalg.norm(coords[11] - coords[12]))
    escala = max(torso_length, shoulder_width, 1e-3)
    coords = (coords - hip_center) / escala
    return np.concatenate([coords.reshape(-1), mascara.astype(np.float32),
                            np.array([1.0], dtype=np.float32)])


# Landmarks que CADA ejercicio necesita ver de verdad para que su diagnostico
# signifique algo. No es lo mismo que active_landmarks (los que vio el modelo en
# entrenamiento): esto es un control de ENTRADA. MediaPipe siempre devuelve los 33
# —si solo ve el torso, EXTRAPOLA las piernas con baja confianza— asi que sin esta
# comprobacion el sistema mide valgo de rodilla sobre coordenadas inventadas y
# entrega un falso positivo con toda naturalidad. Verificado en produccion.
LANDMARKS_REQUERIDOS = {
    'squat': ([23, 24, 25, 26, 27, 28], 'las caderas, rodillas y tobillos'),
    'inline_lunge': ([23, 24, 25, 26, 27, 28], 'las caderas, rodillas y tobillos'),
    'shoulder_abduction': ([11, 12, 13, 14, 15, 16, 23, 24],
                            'los hombros, codos, munecas y caderas'),
    'elbow_flexion': ([11, 12, 13, 14, 15, 16, 23, 24],
                       'los hombros, codos, munecas y caderas'),
}
VISIBILIDAD_MINIMA = 0.5

# Recall minimo, medido en la validacion por sujeto, para que la app se atreva a
# NOMBRAR un subtipo de error. Bajo este valor dice que hay una desviacion pero
# no cual: en sentadilla, 'valgo_de_rodilla' tiene precision 0.26 -- cuando dice
# "valgo", 3 de cada 4 veces no lo es. Nombrarlo igual seria presentar como
# diagnostico algo que el sistema acierta menos de la mitad de las veces, y el
# usuario no tiene forma de saber cual de las dos veces le toco.
RECALL_MINIMO_PARA_TIPIFICAR = 0.40

# Que articulaciones resaltar segun el criterio cinematico que se salio de rango.
# Antes se indexaba por la clase que devolvia la red; ahora por la medida que
# efectivamente falló, que es lo que el usuario necesita mirar.
FOCOS_POR_CRITERIO = {
    'profundidad': [23, 24, 25, 26, 27, 28],
    'paralelismo_tronco_tibia': [11, 12, 23, 24, 25, 26, 27, 28],
    'valgo_de_rodilla': [25, 26, 27, 28],
    'simetria': [25, 26, 27, 28],
    'rango_incompleto': [11, 12, 13, 14, 15, 16],
    'compensacion_de_tronco': [11, 12, 23, 24],
}


def visibilidad_de(result):
    """Confianza por landmark que reporta MediaPipe, SIN tocar. No entra al
    modelo —el tensor de entrenamiento usa visibilidad binaria— pero es lo unico
    que permite saber si una articulacion se vio o se dedujo."""
    if not result.pose_landmarks:
        return np.zeros(NUM_LANDMARKS, dtype=np.float32)
    return np.array([getattr(p, 'visibility', 0.0) or 0.0 for p in result.pose_landmarks[0]],
                     dtype=np.float32)


def segmento_visible(visibilidad_media, exercise):
    """Comprueba que las articulaciones que este ejercicio necesita se vieran
    realmente. Devuelve (es_evaluable, descripcion, visibilidad_obtenida)."""
    requeridos, descripcion = LANDMARKS_REQUERIDOS.get(exercise, (None, ''))
    if not requeridos:
        return True, descripcion, 1.0
    obtenida = float(np.mean(visibilidad_media[requeridos]))
    return obtenida >= VISIBILIDAD_MINIMA, descripcion, obtenida


def training_domain_report(sequence, train_mean, train_std):
    """Fraccion del tensor que satura tras normalizar: si es alta, la entrada
    esta fuera del dominio aprendido y la prediccion no es confiable."""
    normalizada = (sequence - train_mean) / train_std
    saturadas = float(np.mean(np.abs(normalizada) >= 8.0))
    return {'fraccion_saturada': saturadas, 'confiable': saturadas < 0.15}


def interpolate_missing(sequence):
    detected = np.nan_to_num(sequence[:, -1], nan=0.0)
    columnas = []
    for indice in range(sequence.shape[1] - 1):
        columna = sequence[:, indice]
        validos = ~np.isnan(columna)
        if validos.all():
            columnas.append(columna)
        elif validos.any():
            columnas.append(np.interp(np.arange(len(columna)), np.flatnonzero(validos),
                                       columna[validos]))
        else:
            columnas.append(np.zeros_like(columna))
    coordenadas = np.stack(columnas, axis=1).astype(np.float32)
    salida = np.concatenate([coordenadas, detected[:, None].astype(np.float32)], axis=1)
    return salida, float(detected.mean())


# ---------- Motor cinematico (Seccion 3.0.d) ----------
# INYECTADO desde el notebook: misma cadena CODIGO_CINEMATICA.
# Profundidad de sentadilla por FLEXIÓN DE RODILLA, según Schoenfeld (2010):
#   parcial ~40°  |  media 70-100°  |  profunda >100°
# Ojo con la convención: la flexión es 180° menos el ángulo interno
# cadera-rodilla-tobillo. Mezclarlas invierte los umbrales.
FLEXION_RODILLA_MEDIA = 70.0
FLEXION_RODILLA_PROFUNDA = 100.0
# La sentadilla PROFUNDA es una categoría válida de la literatura, no un error.
# Solo se marca la profundidad INSUFICIENTE.
TOLERANCIA_PROFUNDIDAD = 10.0

# Paralelismo tronco-tibia (Kritz et al., 2009): con el centro de masa sobre el
# medio pie, el vector del torso y el de la tibia quedan aproximadamente
# paralelos. Es un criterio RELATIVO, así que se ajusta solo a la profundidad y
# a la anatomía de cada persona — que es justo lo que un umbral absoluto de
# inclinación no puede hacer. Las tolerancias son convención de este trabajo.
# Kritz et al. lo enuncian de forma cualitativa ("aproximadamente paralelos"),
# no como un numero. 20°/30° es tolerancia de este trabajo, elegida amplia a
# proposito: la proporcion femur/tibia cambia mucho entre personas y una banda
# estrecha castigaria la anatomia en vez de la tecnica.
PARALELISMO_EN_RANGO = 20.0
PARALELISMO_LIMITE = 30.0

# Valgo: desplazamiento medial de la rodilla respecto de la recta cadera-tobillo,
# normalizado por el ancho de caderas. Adimensional, así que no depende del
# tamaño de la persona. Tolerancias: convención de este trabajo.
VALGO_EN_RANGO = 0.10
VALGO_LIMITE = 0.20

# Asimetría entre piernas y compensación de tronco: convención de este trabajo.
ASIMETRIA_EN_RANGO = 10.0
ASIMETRIA_LIMITE = 20.0
COMPENSACION_TRONCO_EN_RANGO = 15.0
COMPENSACION_TRONCO_LIMITE = 25.0

# Rango articular de hombro y codo: convención de este trabajo, tomada de los
# valores de amplitud habituales del gesto completo.
ABDUCCION_COMPLETA = 150.0
ABDUCCION_LIMITE = 120.0
FLEXION_CODO_COMPLETA = 120.0
FLEXION_CODO_LIMITE = 95.0

# Confianza mínima del azimut para creerle al plano estimado.
CONFIANZA_VISTA_MINIMA = 0.25


def _angulo_articular(a, vertice, c):
    """Ángulo (grados) en `vertice` entre los segmentos vertice->a y vertice->c."""
    va, vc = np.asarray(a) - np.asarray(vertice), np.asarray(c) - np.asarray(vertice)
    coseno = np.sum(va * vc, axis=-1) / (
        np.linalg.norm(va, axis=-1) * np.linalg.norm(vc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0)))


def _angulo_con_vertical(vectores, desde_abajo=False):
    """Ángulo (grados) respecto de la vertical. En la convención canónica el eje
    vertical es Y y crece hacia ABAJO.

    desde_abajo=False: desviación respecto de la LÍNEA vertical, 0-90°. Correcto
        para el tronco y la tibia, que no se invierten.
    desde_abajo=True: ángulo respecto de la DIRECCIÓN hacia abajo, 0-180°
        (0 = colgando, 90 = horizontal, 180 = sobre la cabeza). Obligatorio para
        el brazo: acotado a 90°, una abducción de 170° mediría 10° y sería
        indistinguible del brazo colgando.
    """
    vectores = np.asarray(vectores)
    componente_vertical = vectores[..., 1] if desde_abajo else np.abs(vectores[..., 1])
    fuera_de_eje = np.linalg.norm(vectores[..., [0, 2]], axis=-1)
    return np.degrees(np.arctan2(fuera_de_eje, componente_vertical + 1e-9))


# Longitud minima de un segmento para creerle, en unidades de torso. Por debajo
# de esto el esqueleto esta colapsado y cualquier angulo que se calcule es ruido.
SEGMENTO_MINIMO = 0.05


def postura_utilizable(coords, ejercicio):
    """Comprueba que el esqueleto tenga geometria real antes de medir angulos.

    Sin esto, un tensor degenerado —todo en cero, o un landmark que MediaPipe no
    vio y quedo en el origen— produce angulos perfectamente calculables y sin
    ningun sentido, y el sistema declararia "correcto" una postura que no existe.
    """
    medio = coords[len(coords) // 2]
    if ejercicio in EJERCICIOS_DE_PIERNA:
        segmentos = ((23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (11, 23))
    else:
        segmentos = ((11, 12), (12, 14), (14, 16), (11, 23), (12, 24))
    largos = [float(np.linalg.norm(medio[a] - medio[b])) for a, b in segmentos]
    if min(largos) < SEGMENTO_MINIMO:
        return False, 'el esqueleto detectado no tiene geometria valida (segmentos colapsados)'
    if not np.isfinite(coords).all():
        return False, 'hay coordenadas no finitas en la secuencia'
    return True, ''


def plano_de_la_vista(info_vista):
    """Qué plano permite medir este encuadre.

    Trabajamos con coordenadas 3D, así que en teoría cualquier ángulo se puede
    calcular desde cualquier ángulo de cámara. En la práctica no: la coordenada
    de profundidad de MediaPipe es una ESTIMACIÓN monocular con mucho más error
    que las dos de imagen. Entonces lo que importa es qué medida cae en el plano
    de la imagen —donde el dato es bueno— y cuál cae en la profundidad.

      vista LATERAL  -> el plano sagital está en la imagen: profundidad de
                        sentadilla, inclinación de tronco y tibia son confiables.
      vista FRONTAL  -> el plano frontal está en la imagen: la alineación de
                        rodillas (valgo) y la simetría son confiables.

    El azimut que la Sección 1.4.5 quitó al canonicalizar es justamente el
    ángulo de la cámara, así que ya lo tenemos calculado.
    """
    if not info_vista or not info_vista.get('aplicado'):
        return {'plano': 'indeterminado', 'motivo': 'no se pudo estimar el ángulo de cámara'}
    if float(info_vista.get('confianza_azimut', 0.0)) < CONFIANZA_VISTA_MINIMA:
        return {'plano': 'indeterminado',
                'motivo': 'la orientación del cuerpo no se estimó con confianza suficiente'}
    # |cos(azimut)| alto => la línea de caderas estaba a lo ancho de la imagen.
    coseno = abs(float(np.cos(np.radians(info_vista.get('azimut_grados', 0.0)))))
    if coseno > 0.70:
        return {'plano': 'frontal', 'motivo': ''}
    if coseno < 0.35:
        return {'plano': 'sagital', 'motivo': ''}
    return {'plano': 'oblicuo', 'motivo': 'la cámara está en diagonal'}


def _indice_mas_profundo(coords):
    """Instante de máxima flexión de rodilla: el punto más bajo del movimiento."""
    flexion = 180.0 - (_angulo_articular(coords[:, 23], coords[:, 25], coords[:, 27]) +
                       _angulo_articular(coords[:, 24], coords[:, 26], coords[:, 28])) / 2.0
    return int(np.argmax(flexion)), flexion


def medidas_de_sentadilla(coords):
    """Medidas cinemáticas en el instante más profundo. `coords` es (T,33,3) ya
    canonicalizado: +X a la izquierda de la persona, +Y abajo, -Z adelante.

    Los ángulos son invariantes a la escala, así que da igual que la secuencia
    venga normalizada por la longitud de torso de cada fotograma.
    """
    indice, flexion_media = _indice_mas_profundo(coords)
    c = coords[indice]

    flexion_izq = 180.0 - float(_angulo_articular(c[23], c[25], c[27]))
    flexion_der = 180.0 - float(_angulo_articular(c[24], c[26], c[28]))

    centro_cadera = (c[23] + c[24]) / 2.0
    centro_hombro = (c[11] + c[12]) / 2.0
    tronco = float(_angulo_con_vertical(centro_hombro - centro_cadera))
    tibia = float(np.mean([_angulo_con_vertical(c[25] - c[27]),
                           _angulo_con_vertical(c[26] - c[28])]))

    # Valgo: cuánto se mete la rodilla hacia la línea media respecto de la recta
    # cadera-tobillo, medido en el eje lateral (X) y normalizado por el ancho de
    # caderas para que no dependa del tamaño de la persona.
    ancho_caderas = float(abs(c[23][0] - c[24][0])) or 1e-6
    valgo = []
    for cadera, rodilla, tobillo, signo in ((23, 25, 27, 1.0), (24, 26, 28, -1.0)):
        proporcion = float(np.clip(
            (c[rodilla][1] - c[cadera][1]) / ((c[tobillo][1] - c[cadera][1]) or 1e-6), 0.0, 1.0))
        x_esperado = c[cadera][0] + proporcion * (c[tobillo][0] - c[cadera][0])
        # Positivo = la rodilla se fue hacia adentro (hacia la línea media).
        valgo.append(signo * (x_esperado - c[rodilla][0]) / ancho_caderas)

    return {
        'indice_clave': indice,
        'flexion_rodilla': (flexion_izq + flexion_der) / 2.0,
        'flexion_rodilla_izq': flexion_izq,
        'flexion_rodilla_der': flexion_der,
        'asimetria_rodillas': abs(flexion_izq - flexion_der),
        'inclinacion_tronco': tronco,
        'inclinacion_tibia': tibia,
        'desalineacion_tronco_tibia': abs(tronco - tibia),
        'valgo': max(valgo),
        'recorrido_flexion': float(np.ptp(flexion_media)),
    }


def medidas_de_brazo(coords, indice=None):
    """Abducción de hombro, flexión de codo y compensación de tronco."""
    hombro, codo, muneca = 12, 14, 16          # lado derecho tras el espejado
    # La abducción de hombro es el ángulo del HÚMERO (hombro->codo), no de la
    # línea hombro->muñeca. Usar la muñeca mezcla dos articulaciones: con el codo
    # flexionado 10°, una abducción real de 165° se mediría como 156°, y el
    # error crece con la flexión de codo. Es la diferencia entre medir el gesto
    # y medir una suma de gestos.
    abduccion = _angulo_con_vertical(coords[:, codo] - coords[:, hombro], desde_abajo=True)
    flexion_codo = 180.0 - _angulo_articular(coords[:, hombro], coords[:, codo], coords[:, muneca])
    if indice is None:
        indice = int(np.argmax(abduccion))
    centro_cadera = (coords[:, 23] + coords[:, 24]) / 2.0
    centro_hombro = (coords[:, 11] + coords[:, 12]) / 2.0
    tronco = _angulo_con_vertical(centro_hombro - centro_cadera)
    return {
        'indice_clave': int(indice),
        'abduccion_max': float(np.max(abduccion)),
        'flexion_codo_max': float(np.max(flexion_codo)),
        'extension_codo_min': float(np.min(flexion_codo)),
        'inclinacion_tronco_max': float(np.max(tronco)),
        'recorrido_abduccion': float(np.ptp(abduccion)),
        'recorrido_codo': float(np.ptp(flexion_codo)),
    }


def _veredicto(valor, en_rango, limite, mayor_es_peor=True):
    """Clasifica un valor en tres bandas en vez de un corte binario."""
    if mayor_es_peor:
        if valor <= en_rango:
            return 'en_rango'
        return 'limite' if valor <= limite else 'fuera_de_rango'
    if valor >= en_rango:
        return 'en_rango'
    return 'limite' if valor >= limite else 'fuera_de_rango'


def criterios_de_sentadilla(medidas, plano):
    """Lista de hallazgos, cada uno con su valor medido, su referencia y si el
    encuadre permitía evaluarlo."""
    sagital = plano in ('sagital', 'oblicuo')
    frontal = plano in ('frontal', 'oblicuo')
    hallazgos = []

    # 1. Profundidad. La sentadilla profunda NO es un error: solo se marca la
    #    que no alcanza el rango medio (paralelo).
    flexion = medidas['flexion_rodilla']
    if flexion >= FLEXION_RODILLA_PROFUNDA:
        categoria = 'profunda'
    elif flexion >= FLEXION_RODILLA_MEDIA:
        categoria = 'media (paralelo)'
    else:
        categoria = 'parcial'
    hallazgos.append({
        'clave': 'profundidad',
        'nombre': 'Profundidad de la sentadilla',
        'valor': round(flexion, 1), 'unidad': '° de flexión de rodilla',
        'referencia': f'>= {FLEXION_RODILLA_MEDIA:.0f}° para alcanzar el rango medio; '
                      f'> {FLEXION_RODILLA_PROFUNDA:.0f}° es sentadilla profunda',
        'veredicto': ('no_evaluable' if not sagital else
                      _veredicto(flexion, FLEXION_RODILLA_MEDIA,
                                 FLEXION_RODILLA_MEDIA - TOLERANCIA_PROFUNDIDAD,
                                 mayor_es_peor=False)),
        'lectura': f'Clasificación: sentadilla {categoria}.',
        'plano': 'sagital', 'fuente': 'schoenfeld2010',
    })

    # 2. Paralelismo tronco-tibia. Criterio RELATIVO: se ajusta solo a la
    #    profundidad y a la anatomía, que es lo que un umbral absoluto de
    #    inclinación no puede hacer.
    desalineacion = medidas['desalineacion_tronco_tibia']
    hallazgos.append({
        'clave': 'paralelismo_tronco_tibia',
        'nombre': 'Alineación del tronco con la tibia',
        'valor': round(desalineacion, 1), 'unidad': '° de diferencia',
        'referencia': f'<= {PARALELISMO_EN_RANGO:.0f}° (tronco y tibia aproximadamente paralelos)',
        'veredicto': ('no_evaluable' if not sagital else
                      _veredicto(desalineacion, PARALELISMO_EN_RANGO, PARALELISMO_LIMITE)),
        'lectura': (f"Tronco {medidas['inclinacion_tronco']:.0f}° y tibia "
                    f"{medidas['inclinacion_tibia']:.0f}° respecto de la vertical."),
        'plano': 'sagital', 'fuente': 'kritz2009',
    })

    # 3. Valgo. Plano frontal: solo se evalúa si la cámara lo permite.
    hallazgos.append({
        'clave': 'valgo_de_rodilla',
        'nombre': 'Alineación de las rodillas',
        'valor': round(medidas['valgo'], 3), 'unidad': 'del ancho de caderas',
        'referencia': f'<= {VALGO_EN_RANGO:.2f} de desplazamiento medial',
        'veredicto': ('no_evaluable' if not frontal else
                      _veredicto(medidas['valgo'], VALGO_EN_RANGO, VALGO_LIMITE)),
        'lectura': ('' if frontal else
                    'El valgo es una medida del plano frontal: hace falta grabar de frente.'),
        'plano': 'frontal', 'fuente': 'hewett2005',
    })

    # 4. Simetría entre piernas.
    hallazgos.append({
        'clave': 'simetria',
        'nombre': 'Simetría entre piernas',
        'valor': round(medidas['asimetria_rodillas'], 1), 'unidad': '° de diferencia',
        'referencia': f'<= {ASIMETRIA_EN_RANGO:.0f}°',
        'veredicto': ('no_evaluable' if not frontal else
                      _veredicto(medidas['asimetria_rodillas'], ASIMETRIA_EN_RANGO,
                                 ASIMETRIA_LIMITE)),
        'lectura': (f"Izquierda {medidas['flexion_rodilla_izq']:.0f}°, "
                    f"derecha {medidas['flexion_rodilla_der']:.0f}°."),
        'plano': 'frontal', 'fuente': None,
    })
    return hallazgos


def criterios_de_brazo(medidas, plano, ejercicio):
    hallazgos = []
    if ejercicio == 'shoulder_abduction':
        valor = medidas['abduccion_max']
        hallazgos.append({
            'clave': 'rango_incompleto',
            'nombre': 'Amplitud de la abducción',
            'valor': round(valor, 1), 'unidad': '° desde el brazo colgando',
            'referencia': f'>= {ABDUCCION_COMPLETA:.0f}° para el recorrido completo',
            'veredicto': _veredicto(valor, ABDUCCION_COMPLETA, ABDUCCION_LIMITE,
                                     mayor_es_peor=False),
            'lectura': '', 'plano': 'frontal', 'fuente': None,
        })
    else:
        valor = medidas['flexion_codo_max']
        hallazgos.append({
            'clave': 'rango_incompleto',
            'nombre': 'Amplitud de la flexión de codo',
            'valor': round(valor, 1), 'unidad': '° de flexión',
            'referencia': f'>= {FLEXION_CODO_COMPLETA:.0f}° para el recorrido completo',
            'veredicto': _veredicto(valor, FLEXION_CODO_COMPLETA, FLEXION_CODO_LIMITE,
                                     mayor_es_peor=False),
            'lectura': '', 'plano': 'sagital', 'fuente': None,
        })
    compensacion = medidas['inclinacion_tronco_max']
    hallazgos.append({
        'clave': 'compensacion_de_tronco',
        'nombre': 'Estabilidad del tronco',
        'valor': round(compensacion, 1), 'unidad': '° respecto de la vertical',
        'referencia': f'<= {COMPENSACION_TRONCO_EN_RANGO:.0f}°',
        'veredicto': _veredicto(compensacion, COMPENSACION_TRONCO_EN_RANGO,
                                 COMPENSACION_TRONCO_LIMITE),
        'lectura': 'Balancear el tronco para ayudar al brazo desplaza el trabajo '
                   'fuera del músculo objetivo.',
        'plano': 'sagital', 'fuente': None,
    })
    return hallazgos


EJERCICIOS_DE_PIERNA = ('squat', 'inline_lunge')


def evaluar_cinematica(secuencia, ejercicio, info_vista=None):
    """Tipificación por ángulos medidos. Devuelve el plano detectado, las
    medidas crudas y los hallazgos ordenados por gravedad."""
    coords = _coords_de(secuencia)
    vista = plano_de_la_vista(info_vista or {})
    utilizable, motivo = postura_utilizable(coords, ejercicio)
    if not utilizable:
        return {'plano': 'indeterminado', 'motivo_plano': motivo, 'medidas': {},
                'hallazgos': [], 'principal': None, 'n_fuera_de_rango': 0, 'n_limite': 0,
                'evaluables': [], 'no_evaluables': [], 'postura_utilizable': False}
    if ejercicio in EJERCICIOS_DE_PIERNA:
        medidas = medidas_de_sentadilla(coords)
        hallazgos = criterios_de_sentadilla(medidas, vista['plano'])
    else:
        medidas = medidas_de_brazo(coords)
        hallazgos = criterios_de_brazo(medidas, vista['plano'], ejercicio)

    orden = {'fuera_de_rango': 0, 'limite': 1, 'en_rango': 2, 'no_evaluable': 3}
    hallazgos.sort(key=lambda h: (orden[h['veredicto']], -abs(h['valor'])))
    fuera = [h for h in hallazgos if h['veredicto'] == 'fuera_de_rango']
    limite = [h for h in hallazgos if h['veredicto'] == 'limite']
    return {
        'plano': vista['plano'], 'motivo_plano': vista['motivo'],
        'medidas': medidas, 'hallazgos': hallazgos,
        'principal': (fuera or limite or [None])[0],
        'n_fuera_de_rango': len(fuera), 'n_limite': len(limite),
        'evaluables': [h['clave'] for h in hallazgos if h['veredicto'] != 'no_evaluable'],
        'no_evaluables': [h['clave'] for h in hallazgos if h['veredicto'] == 'no_evaluable'],
        'postura_utilizable': True,
    }



# ---------- Canonicalizacion de punto de vista (Seccion 1.4.5) ----------
# INYECTADO desde el notebook: es la misma cadena CODIGO_CANONICALIZACION
# con la que se construyo cada tensor de entrenamiento.
CANONICALIZAR_VISTA = True
CORREGIR_INCLINACION_CAMARA = True
MAX_CORRECCION_INCLINACION_GRADOS = 20.0
LANDMARKS_SUELO = [27, 28, 29, 30, 31, 32]
VISIBILIDAD_SUELO_MINIMA = 0.5
PLANITUD_MINIMA = 0.15
RESIDUO_MAXIMO_PLANO = 0.35

def _coords_de(secuencia: np.ndarray) -> np.ndarray:
    """(T,133) -> (T,33,3) con las coordenadas."""
    return np.asarray(secuencia, dtype=np.float32)[:, :NUM_LANDMARKS * 3].reshape(
        -1, NUM_LANDMARKS, 3)


def _visibilidad_de_secuencia(secuencia: np.ndarray) -> np.ndarray:
    """(T,133) -> (T,33) con el canal de visibilidad."""
    return np.asarray(secuencia, dtype=np.float32)[:, NUM_LANDMARKS * 3:NUM_LANDMARKS * 4]


def _con_coords(secuencia: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Devuelve una copia de la secuencia con otras coordenadas. Los landmarks
    con visibilidad 0 se vuelven a poner en cero EXACTO: son huecos, no datos, y
    dejarles un 1e-9 de error numérico rompería active_landmarks_from_training
    y el control de saturación de dominio de la app."""
    salida = np.asarray(secuencia, dtype=np.float32).copy()
    visibilidad = _visibilidad_de_secuencia(salida)
    coords = np.asarray(coords, dtype=np.float32).copy()
    coords[visibilidad <= 0.0] = 0.0
    salida[:, :NUM_LANDMARKS * 3] = coords.reshape(len(coords), -1)
    return salida


def aplicar_rotacion(coords: np.ndarray, rotacion: np.ndarray) -> np.ndarray:
    """Rota (T,33,3) por una matriz 3x3 (vectores fila: c' = R c)."""
    return np.asarray(coords, dtype=np.float32) @ np.asarray(rotacion, dtype=np.float32).T


def matriz_rotacion_vertical(theta: float) -> np.ndarray:
    """Rotación de -theta alrededor del eje vertical (Y), que lleva un vector
    horizontal de ángulo theta sobre +X. Y queda intacto, así que la inclinación
    respecto a la vertical se conserva."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float32)


def rotacion_entre_vectores(origen: np.ndarray, destino: np.ndarray) -> np.ndarray:
    """Rotación mínima (Rodrigues) que lleva `origen` sobre `destino`. Mínima
    importa: cualquier otra añadiría un giro extra alrededor del eje común, y
    ese giro sí cambiaría el tensor sin ninguna justificación anatómica."""
    a = np.asarray(origen, dtype=np.float64)
    b = np.asarray(destino, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    b = b / max(float(np.linalg.norm(b)), 1e-9)
    v = np.cross(a, b)
    seno = float(np.linalg.norm(v))
    coseno = float(np.dot(a, b))
    if seno < 1e-8:
        if coseno > 0:
            return np.eye(3, dtype=np.float32)
        # Antiparalelos: media vuelta alrededor de cualquier eje perpendicular.
        auxiliar = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        eje = np.cross(a, auxiliar)
        eje = eje / max(float(np.linalg.norm(eje)), 1e-9)
        V = np.array([[0.0, -eje[2], eje[1]], [eje[2], 0.0, -eje[0]], [-eje[1], eje[0], 0.0]])
        return (np.eye(3) + 2.0 * V @ V).astype(np.float32)
    V = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return (np.eye(3) + V + V @ V * ((1.0 - coseno) / (seno ** 2))).astype(np.float32)


def estimar_normal_del_suelo(coords: np.ndarray, visibilidad: np.ndarray):
    """Ajusta un plano a los landmarks del pie (que están sobre el suelo) y
    devuelve (normal_hacia_abajo, diagnóstico). La normal es la vertical REAL:
    comparada con el eje Y del sensor, da la inclinación de la cámara.
    Devuelve (None, motivo) si la geometría no permite un ajuste confiable."""
    visibilidad_media = np.asarray(visibilidad, dtype=np.float32).mean(axis=0)
    presentes = [i for i in LANDMARKS_SUELO if visibilidad_media[i] >= VISIBILIDAD_SUELO_MINIMA]
    if len(presentes) < 3:
        return None, {'motivo': f'solo {len(presentes)} landmarks de pie visibles (hacen falta 3)'}

    nube = np.asarray(coords, dtype=np.float64)[:, presentes, :].reshape(-1, 3)
    nube = nube[np.isfinite(nube).all(axis=1)]
    if len(nube) < 6:
        return None, {'motivo': 'muy pocos puntos de pie utilizables'}

    centrada = nube - nube.mean(axis=0)
    try:
        _u, sigmas, vt = np.linalg.svd(centrada, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, {'motivo': 'el ajuste del plano no convergió'}
    if sigmas[0] <= 1e-9:
        return None, {'motivo': 'los puntos de pie colapsan en uno solo'}

    planitud = float(sigmas[1] / sigmas[0])
    residuo = float(sigmas[2] / max(sigmas[1], 1e-9))
    if planitud < PLANITUD_MINIMA:
        return None, {'motivo': f'los pies son casi colineales (planitud {planitud:.2f})',
                      'planitud': planitud}
    if residuo > RESIDUO_MAXIMO_PLANO:
        return None, {'motivo': f'los pies no forman un plano (residuo {residuo:.2f})',
                      'planitud': planitud, 'residuo': residuo}

    normal = vt[2]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
    # Orientada hacia ABAJO, igual que +Y en la convención MediaPipe.
    if normal[1] < 0:
        normal = -normal
    inclinacion = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    if inclinacion > MAX_CORRECCION_INCLINACION_GRADOS:
        return None, {'motivo': f'inclinación estimada {inclinacion:.0f}° sobre el tope de '
                                f'{MAX_CORRECCION_INCLINACION_GRADOS:.0f}°: ajuste poco creíble',
                      'inclinacion_grados': inclinacion}
    return normal.astype(np.float32), {'planitud': planitud, 'residuo': residuo,
                                        'inclinacion_grados': inclinacion}


def estimar_azimut(coords: np.ndarray, visibilidad: np.ndarray):
    """Ángulo (radianes) de la línea de caderas en el plano horizontal, y su
    confianza. Se usa la cadera y no el hombro porque el tronco rota durante el
    gesto y la pelvis no: la pelvis define la orientación de la PERSONA, el
    hombro la del tronco en ese instante."""
    coords = np.asarray(coords, dtype=np.float64)
    visibilidad = np.asarray(visibilidad, dtype=np.float32)

    for izquierdo, derecho in ((23, 24), (11, 12)):
        utiles = (visibilidad[:, izquierdo] > 0) & (visibilidad[:, derecho] > 0)
        if not utiles.any():
            continue
        linea = (coords[utiles, izquierdo, :] - coords[utiles, derecho, :]).mean(axis=0)
        horizontal = float(np.hypot(linea[0], linea[2]))
        norma = float(np.linalg.norm(linea))
        if horizontal < 1e-6 or norma < 1e-6:
            continue
        # La confianza cae si el segmento apunta sobre todo hacia arriba/abajo
        # (cámara muy picada) o si su largo es anómalo respecto al torso, que es
        # la escala con la que ya se normalizó la secuencia.
        confianza = float(horizontal / norma) * float(min(1.0, norma / 0.25))
        return float(np.arctan2(linea[2], linea[0])), confianza
    return 0.0, 0.0


def canonicalizar_vista(secuencia: np.ndarray, corregir_inclinacion: Optional[bool] = None,
                         activo: Optional[bool] = None):
    """Lleva una secuencia (T,133) al marco corporal. ESTA función es el único
    lugar donde se define el punto de vista canónico, y la app ejecuta una copia
    idéntica: si divergieran, el modelo vería en producción un sistema de
    coordenadas distinto del de entrenamiento."""
    activo = CANONICALIZAR_VISTA if activo is None else activo
    corregir_inclinacion = (CORREGIR_INCLINACION_CAMARA if corregir_inclinacion is None
                            else corregir_inclinacion)
    secuencia = np.asarray(secuencia, dtype=np.float32)
    info = {'aplicado': False, 'nivelado': False, 'azimut_grados': 0.0,
            'confianza_azimut': 0.0, 'inclinacion_grados': 0.0, 'motivo_sin_nivelar': ''}
    if not activo:
        return secuencia, info

    coords = _coords_de(secuencia)
    visibilidad = _visibilidad_de_secuencia(secuencia)

    if corregir_inclinacion:
        normal, diagnostico = estimar_normal_del_suelo(coords, visibilidad)
        if normal is None:
            info['motivo_sin_nivelar'] = diagnostico.get('motivo', '')
        else:
            coords = aplicar_rotacion(coords, rotacion_entre_vectores(
                normal, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
            info['nivelado'] = True
            info['inclinacion_grados'] = float(diagnostico.get('inclinacion_grados', 0.0))
            info['planitud_suelo'] = float(diagnostico.get('planitud', float('nan')))

    theta, confianza = estimar_azimut(coords, visibilidad)
    coords = aplicar_rotacion(coords, matriz_rotacion_vertical(theta))
    info['aplicado'] = True
    info['azimut_grados'] = float(np.degrees(theta))
    info['confianza_azimut'] = float(confianza)
    return _con_coords(secuencia, coords), info


def canonicalizar_lote(X: np.ndarray, **kwargs) -> np.ndarray:
    """Aplica canonicalizar_vista a un conjunto (N,T,133)."""
    return np.stack([canonicalizar_vista(m, **kwargs)[0] for m in X]).astype(np.float32)

# El bloque de arriba NO se escribe a mano: la celda lo inyecta con
# inspect.getsource() desde las funciones YA EJECUTADAS de la Seccion 1.4.5, asi
# que es literalmente el mismo codigo con el que se construyo cada tensor de
# entrenamiento. Copiarlo a mano seria la forma mas facil de que la app y el
# notebook acaben en sistemas de coordenadas distintos sin que nadie lo note.


def extract_video_sequence(video_path, active_landmarks=None, sequence_size=SEQ_LEN):
    captura = cv2.VideoCapture(str(video_path))
    if not captura.isOpened():
        raise ValueError('No se pudo abrir el video.')
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        contados = 0
        while True:
            ok, _ = captura.read()
            if not ok:
                break
            contados += 1
        captura.release()
        captura = cv2.VideoCapture(str(video_path))
        total = contados
    if total <= 0:
        captura.release()
        raise ValueError('El video no contiene fotogramas decodificables.')

    indices = np.linspace(0, total - 1, sequence_size).round().astype(int)
    secuencia = np.full((sequence_size, FEATURE_SIZE), np.nan, dtype=np.float32)
    visibilidades = np.zeros((sequence_size, NUM_LANDMARKS), dtype=np.float32)
    detector = get_pose_detector()
    for posicion, indice in enumerate(indices):
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
        ok, frame = captura.read()
        if not ok:
            continue
        frame = resize_for_pose(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        visibilidades[posicion] = visibilidad_de(resultado)
        secuencia[posicion] = (pose_result_to_features(resultado) if active_landmarks is None
                                else pose_result_to_features_aligned(resultado, active_landmarks))
    captura.release()
    secuencia, cobertura = interpolate_missing(secuencia)
    # Canonicalizacion de punto de vista: MISMA funcion y mismo momento que en
    # el notebook (final de normalize_sequence_like_camera_app). Es lo que hace
    # que un video grabado en diagonal produzca el mismo tensor que uno de
    # frente, y lo que evita que el angulo de camara se lea como error tecnico.
    secuencia, info_vista = canonicalizar_vista(secuencia)
    return secuencia, cobertura, visibilidades.mean(axis=0), info_vista


# ---------------- Visualizacion: esqueleto y zona del error ----------------
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 31),
    (24, 26), (26, 28), (28, 32),
]
COLOR_NORMAL, COLOR_FOCO, COLOR_OK = (170, 170, 170), (60, 60, 235), (90, 200, 90)


def _landmarks_a_pixeles(result, width, height):
    if not result.pose_landmarks:
        return None
    return np.array([[lm.x * width, lm.y * height] for lm in result.pose_landmarks[0]],
                     dtype=np.float32)


def indice_instante_clave(poses, exercise):
    """Frame extremo del movimiento, por ejercicio. En pixeles, y crece hacia ABAJO."""
    if exercise in ('squat', 'inline_lunge'):
        return int(np.argmax((poses[:, 23, 1] + poses[:, 24, 1]) / 2.0))
    if exercise == 'shoulder_abduction':
        return int(np.argmin(np.minimum(poses[:, 15, 1], poses[:, 16, 1])))
    if exercise == 'elbow_flexion':
        izquierda = np.linalg.norm(poses[:, 15] - poses[:, 11], axis=1)
        derecha = np.linalg.norm(poses[:, 16] - poses[:, 12], axis=1)
        return int(np.argmin(np.minimum(izquierda, derecha)))
    referencia = np.median(poses[:max(1, len(poses) // 10)], axis=0)
    return int(np.argmax(np.linalg.norm((poses - referencia).reshape(len(poses), -1), axis=1)))


def dibujar_esqueleto(frame, puntos, focos, es_correcto=False):
    lienzo = frame.copy()
    foco_set = set(focos)
    color_resalte = COLOR_OK if es_correcto else COLOR_FOCO
    for a, b in POSE_CONNECTIONS:
        if a >= len(puntos) or b >= len(puntos):
            continue
        en_foco = a in foco_set and b in foco_set
        cv2.line(lienzo, tuple(puntos[a].astype(int)), tuple(puntos[b].astype(int)),
                  color_resalte if en_foco else COLOR_NORMAL, 6 if en_foco else 3, cv2.LINE_AA)
    for indice in {i for conexion in POSE_CONNECTIONS for i in conexion}:
        if indice >= len(puntos):
            continue
        en_foco = indice in foco_set
        cv2.circle(lienzo, tuple(puntos[indice].astype(int)), 8 if en_foco else 5,
                    color_resalte if en_foco else COLOR_NORMAL, -1, cv2.LINE_AA)
    return lienzo


def _escribir_banner(frame, titulo, subtitulo, es_correcto):
    alto, ancho = frame.shape[:2]
    escala = max(0.5, min(1.1, ancho / 900))
    alto_banner = int(58 * escala) + (int(30 * escala) if subtitulo else 0)
    superposicion = frame.copy()
    cv2.rectangle(superposicion, (0, 0), (ancho, alto_banner), (25, 25, 25), -1)
    frame = cv2.addWeighted(superposicion, 0.72, frame, 0.28, 0)
    cv2.putText(frame, titulo, (int(16 * escala), int(36 * escala)), cv2.FONT_HERSHEY_SIMPLEX,
                 0.85 * escala, COLOR_OK if es_correcto else COLOR_FOCO, 2, cv2.LINE_AA)
    if subtitulo:
        cv2.putText(frame, subtitulo, (int(16 * escala), int(66 * escala)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55 * escala, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def _a_h264(entrada):
    salida = entrada.with_name(entrada.stem + '_h264.mp4')
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(entrada),
                         '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', str(salida)],
                        check=True, capture_output=True)
        return salida
    except (subprocess.CalledProcessError, FileNotFoundError):
        return entrada


def render_pose_overlay(video_path, exercise, class_id, zona, max_frames=150,
                         lado_maximo=720, titulo_forzado=None, focos_forzados=None):
    """Video con el esqueleto dibujado y la region del error resaltada, mas la
    imagen del instante mas critico del movimiento.

    MEMORIA: la version anterior guardaba cada fotograma anotado en una lista
    para poder elegir el instante clave al final. Con 300 fotogramas de 1080p eso
    son ~1,8 GB de RAM y el proceso muere en cualquier servidor gratuito. Aqui
    solo se conservan los PUNTOS del esqueleto (33 pares de coordenadas, unos
    pocos KB) y, una vez elegido el instante, se vuelve a leer ESE fotograma.
    Tambien se reduce la resolucion: para ver un esqueleto dibujado, 720 px de
    lado mayor sobran, y baja el consumo a una fraccion.
    """
    salida_dir = Path(tempfile.mkdtemp(prefix='safeform_'))
    captura = cv2.VideoCapture(str(video_path))
    if not captura.isOpened():
        raise ValueError('No se pudo abrir el video.')
    ancho_original = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_original = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    escala = min(1.0, lado_maximo / max(ancho_original, alto_original, 1))
    ancho = max(2, int(round(ancho_original * escala)) // 2 * 2)
    alto = max(2, int(round(alto_original * escala)) // 2 * 2)

    if titulo_forzado:
        # Con tipificacion cinematica se resalta la articulacion que se salio de
        # rango; sin ella no se resalta nada, porque senalar una articulacion
        # concreta afirmaria algo que ninguna medida sostiene.
        focos = list(focos_forzados or [])
        es_correcto, titulo = False, titulo_forzado
        subtitulo = zona[:78]
    else:
        focos = CLINICAL['focus_landmarks'][exercise][str(class_id)]
        nombre_clase = CLINICAL['taxonomy'][exercise][str(class_id)]
        es_correcto = nombre_clase == 'correcto'
        titulo = ('Ejecucion correcta' if es_correcto
                  else f"Detectado: {nombre_clase.replace('_', ' ')}")
        subtitulo = ('' if es_correcto else zona)[:78]

    indices = (np.arange(total) if 0 < total <= max_frames
               else np.linspace(0, max(total - 1, 0), max_frames).round().astype(int))
    fps_salida = fps if 0 < total <= max_frames else max(1.0, fps * len(indices) / max(total, 1))

    ruta_video = salida_dir / 'analisis_esqueleto.mp4'
    escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*'mp4v'),
                                fps_salida, (ancho, alto))
    detector = get_pose_detector()

    def preparar(bruto):
        return cv2.resize(bruto, (ancho, alto)) if escala < 1.0 else bruto

    def anotar(frame, puntos):
        return _escribir_banner(dibujar_esqueleto(frame, puntos, focos, es_correcto),
                                 titulo, subtitulo, es_correcto)

    puntos_por_frame, indices_con_pose = [], []
    for indice in indices:
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
        ok, bruto = captura.read()
        if not ok:
            continue
        frame = preparar(bruto)
        resultado_pose = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        puntos = _landmarks_a_pixeles(resultado_pose, ancho, alto)
        if puntos is None:
            escritor.write(_escribir_banner(frame, titulo, subtitulo, es_correcto))
            continue
        escritor.write(anotar(frame, puntos))
        puntos_por_frame.append(puntos)
        indices_con_pose.append(int(indice))
    escritor.release()

    # Instante clave: se elige con los puntos (baratos) y se vuelve a leer solo
    # ese fotograma del video, en vez de haberlos guardado todos.
    ruta_imagen = salida_dir / 'instante_clave.jpg'
    clave = None
    if puntos_por_frame:
        posicion = indice_instante_clave(np.stack(puntos_por_frame), exercise)
        captura.set(cv2.CAP_PROP_POS_FRAMES, indices_con_pose[posicion])
        ok, bruto = captura.read()
        if ok:
            clave = anotar(preparar(bruto), puntos_por_frame[posicion])
    captura.release()
    if clave is None:
        clave = np.zeros((alto, ancho, 3), dtype=np.uint8)
    cv2.imwrite(str(ruta_imagen), clave)
    return str(_a_h264(ruta_video)), str(ruta_imagen)


# ---------------- Modelos y motor clinico ----------------
CLINICAL = json.loads((BASE / 'clinical_kb.json').read_text(encoding='utf-8'))
_bundles = {}


class CascadaInferencia:
    """Version de solo-inferencia del modelo en cascada del notebook. Compone
    P(correcto)=1-P(error) y P(subtipo)=P(error)*P(subtipo|error), de modo que
    el resto de la app trabaja con un vector de probabilidades normal."""

    def __init__(self, detector, tipificador, num_classes, umbral=0.5):
        self.detector, self.tipificador, self.num_classes = detector, tipificador, num_classes
        self.umbral = float(umbral)

    def predict(self, X, verbose=0):
        p_error = np.asarray(self.detector.predict(X, verbose=0))[:, 1]
        p_sub = np.asarray(self.tipificador.predict(X, verbose=0))
        salida = np.zeros((len(X), self.num_classes), dtype=np.float32)
        salida[:, 0] = 1.0 - p_error
        salida[:, 1:] = p_sub * p_error[:, None]
        return salida

    def decidir(self, X):
        # El umbral calibrado, no el argmax: con datos desbalanceados el corte en
        # 0.5 deja pasar la mayoria de las ejecuciones incorrectas.
        p = self.predict(X)
        p_error = 1.0 - p[:, 0]
        subtipo = 1 + np.argmax(p[:, 1:], axis=1)
        return np.where(p_error >= self.umbral, subtipo, 0).astype(int)


def raiz_de_modelos():
    """Carpeta que contiene un subdirectorio por ejercicio.

    Se acepta tanto `modelos/<ejercicio>/` como `<ejercicio>/` en la raiz. Al
    subir el paquete a GitHub por la web es muy facil que las subcarpetas
    queden un nivel mas arriba de lo previsto, y eso tumbaba la app con un
    FileNotFoundError que no decia nada util. Verificado en despliegue real.
    """
    candidata = BASE / 'modelos'
    return candidata if candidata.is_dir() else BASE


def load_bundle(exercise):
    if exercise not in _bundles:
        carpeta = raiz_de_modelos() / exercise
        mapa = json.loads((carpeta / 'class_map_multiclase.json').read_text(encoding='utf-8'))
        if mapa.get('en_cascada'):
            ruta_umbral = carpeta / 'umbral_deteccion.json'
            umbral = (json.loads(ruta_umbral.read_text(encoding='utf-8'))['umbral']
                       if ruta_umbral.exists() else 0.5)
            modelo = CascadaInferencia(
                tf.keras.models.load_model(carpeta / 'modelo_deteccion.keras',
                                            custom_objects=CUSTOM_OBJECTS),
                tf.keras.models.load_model(carpeta / 'modelo_tipificacion.keras',
                                            custom_objects=CUSTOM_OBJECTS),
                len(mapa['classes']), umbral)
        else:
            modelo = tf.keras.models.load_model(
                carpeta / 'safeform_biomecanico_multiclase.keras', custom_objects=CUSTOM_OBJECTS)
        normalizacion = np.load(carpeta / 'preprocesamiento_normalizacion_multiclase.npz')
        _bundles[exercise] = {'model': modelo, 'mean': normalizacion['mean'],
                               'std': normalizacion['std'], 'class_map': mapa,
                               'active_landmarks': mapa['active_landmarks']}
    return _bundles[exercise]


def ejercicios_disponibles():
    """Un ejercicio es una carpeta que trae su mapa de clases. Filtrar por ese
    archivo -y no por "es un directorio"- impide que .streamlit, __pycache__ o
    cualquier otra carpeta del repositorio se cuele como si fuera un modelo."""
    return sorted(p.name for p in raiz_de_modelos().iterdir()
                  if p.is_dir() and (p / 'class_map_multiclase.json').exists())


def etiqueta_es(exercise):
    return CLINICAL['labels'].get(exercise, exercise)


def evaluar(video_path, etiqueta_ejercicio):
    """Devuelve el diagnostico como DATOS, no como texto ya maquetado.

    Separar el analisis de su presentacion es lo que permite que Gradio y
    Streamlit se vean distintos sin duplicar una linea de logica: las dos
    interfaces llaman aqui y cada una decide como mostrarlo. analizar() de mas
    abajo es solo la version que arma markdown a partir de este diccionario.
    """
    if not video_path:
        return {'estado': 'sin_video'}

    # Acepta tanto la etiqueta legible ("Sentadilla") como la clave interna
    # ("squat"). La resolucion vive aqui, en el nucleo, y no en la interfaz.
    exercise = {etiqueta_es(e): e for e in ejercicios_disponibles()}.get(
        etiqueta_ejercicio, etiqueta_ejercicio)
    bundle = load_bundle(exercise)
    try:
        # Restringido a los landmarks con los que se entreno este modelo.
        secuencia, cobertura, visibilidad, info_vista = extract_video_sequence(
            video_path, bundle['active_landmarks'])
    except ValueError as error:
        return {'estado': 'ilegible', 'mensaje': str(error)}

    # Control de encuadre ANTES de predecir. La cobertura global no basta:
    # MediaPipe detecta una pose valida viendo solo el torso y deduce el resto,
    # asi que un video sin piernas en cuadro llega aqui con 90% de cobertura y
    # el modelo acaba midiendo las rodillas sobre coordenadas extrapoladas.
    evaluable, que_falta, visto = segmento_visible(visibilidad, exercise)
    if not evaluable:
        return {'estado': 'fuera_de_encuadre', 'cobertura': cobertura,
                'visibilidad': visto,
                'mensaje': (f'Para evaluar {etiqueta_es(exercise).lower()} hacen falta '
                            f'{que_falta}, y en este video no se ven con claridad '
                            f'(visibilidad media {visto * 100:.0f}%). Vuelve a grabar con el '
                            'cuerpo completo en cuadro.')}

    if cobertura < 0.5:
        return {'estado': 'sin_cobertura', 'cobertura': cobertura,
                'mensaje': ('No se detecto el cuerpo en suficientes fotogramas '
                            f'(cobertura {cobertura * 100:.0f}%). Grabate de cuerpo entero, '
                            'con buena luz y con la camara fija.')}

    dominio = training_domain_report(secuencia, bundle['mean'], bundle['std'])
    normalizada = np.clip((secuencia - bundle['mean']) / bundle['std'], -8.0, 8.0)
    entrada = normalizada[None, ...].astype(np.float32)
    probabilidades = bundle['model'].predict(entrada, verbose=0)[0]
    modelo = bundle['model']
    clase = (int(modelo.decidir(entrada)[0]) if hasattr(modelo, 'decidir')
              else int(np.argmax(probabilidades)))

    nombres = {int(k): v for k, v in CLINICAL['taxonomy'][exercise].items()}
    metricas_modelo = bundle['class_map'].get('metricas_validacion') or {}
    detecta_desviacion = clase != 0

    # TIPIFICACION POR ANGULOS MEDIDOS, no por la clase que devuelve la red.
    #
    # Las dos tareas del sistema tienen evidencia MUY distinta. Detectar si hay
    # desviacion se valido contra la etiqueta real del dataset. Decir CUAL se
    # validaba contra etiquetas que derivamos nosotros por reglas: circular, y
    # una de esas reglas estaba mal (llamaba "inclinacion lumbar excesiva" a un
    # angulo de tronco, que en una sentadilla profunda es lo correcto).
    #
    # Aqui la red sigue decidiendo si hay desviacion y el motor cinematico dice
    # cual, con angulos medidos y umbrales citados. Un numero verificable pesa
    # mas que una clase aprendida de 180 muestras con etiquetas inventadas.
    cinematica = evaluar_cinematica(secuencia, exercise, info_vista)
    principal = cinematica['principal']
    criterio_fuera = principal is not None and principal['veredicto'] == 'fuera_de_rango'

    # CUANDO LA MEDIDA Y EL MODELO SE CONTRADICEN, MANDA LA MEDIDA.
    #
    # No es una preferencia estetica. La red aprendio de UI-PRMD, donde las
    # repeticiones etiquetadas incorrectas resultaron ser las mas PROFUNDAS
    # (d de Cohen -0,46). O sea aprendio, fielmente, un sesgo del dataset:
    # "profundo = malo". Y la sentadilla profunda es una categoria valida de la
    # literatura (Schoenfeld, 2010), no un error.
    #
    # Entonces, si el motor pudo evaluar varios criterios con este encuadre y
    # TODOS quedaron holgadamente dentro de rango, la ejecucion se reporta como
    # correcta aunque la red marque desviacion, y el desacuerdo se deja anotado
    # en el detalle tecnico. Un angulo medido con umbral citado es evidencia mas
    # fuerte que una clase aprendida de 180 muestras con etiquetas derivadas.
    # Un criterio "en el limite" no es un error: es una observacion. Exigir que
    # todo este holgado para declarar correcta la ejecucion reintroduciria por la
    # puerta de atras el exceso de alarmas que estamos corrigiendo. Los criterios
    # al limite se nombran en la recomendacion, sin cambiar el veredicto.
    cinematica_concluyente = (len(cinematica['evaluables']) >= 2
                               and cinematica['n_fuera_de_rango'] == 0)
    sin_tipificar = detecta_desviacion and not criterio_fuera and not cinematica_concluyente

    avisos = []
    if not bundle['class_map'].get('entrenado_con_datos_reales', True):
        avisos.append(('Modelo de demostracion',
                       'Se entreno con datos sinteticos, no con capturas reales. Sirve para '
                       'probar el flujo de la app, pero su diagnostico no es valido.'))
    if not dominio['confiable']:
        # La red siempre da alta probabilidad a ALGUNA clase, aunque la entrada
        # este fuera de su dominio. Advertirlo es mas util que ocultarlo.
        avisos.append(('Prediccion poco confiable',
                       f"El {dominio['fraccion_saturada'] * 100:.0f}% de las caracteristicas de "
                       'este video cae fuera del rango visto en entrenamiento, pese al '
                       'porcentaje de confianza. Suele deberse a un encuadre o angulo de camara '
                       'muy distinto al de los datos de entrenamiento.'))

    no_evaluables = [h for h in cinematica['hallazgos'] if h['veredicto'] == 'no_evaluable']
    if no_evaluables:
        avisos.append((
            'Criterios que este encuadre no permite medir',
            'Con la camara en vista ' + cinematica['plano'] + ' no se pueden evaluar: '
            + ', '.join(h['nombre'].lower() for h in no_evaluables)
            + '. La profundidad y la inclinacion se miden de perfil; la alineacion de rodillas, '
              'de frente. Ninguna camara da los dos planos a la vez.'))

    if criterio_fuera:
        guia = CLINICAL['guia_cinematica'][principal['clave']]
        etiqueta_clase = principal['nombre'].lower()
        zona = guia['zona_biomecanica']
        correccion = guia['instruccion_correctiva']
        fundamento = guia['fundamento_medico']
        referencia = CLINICAL['citations'].get(guia['citation_key'] or '', '')
    elif sin_tipificar:
        avisos.append((
            'Desviacion sin tipificar',
            'El modelo detecto que la ejecucion se aparta del patron aprendido, pero ningun '
            'criterio cinematico evaluable con este encuadre se sale de su rango de referencia. '
            'Puede ser algo que los criterios actuales no cubren, o un limite del propio modelo. '
            'Preferimos decirlo asi antes que darte un nombre que los angulos no sostienen.'))
        etiqueta_clase = 'desviacion tecnica sin tipificar'
        zona = 'No determinada'
        correccion = ('Revisa la ejecucion con el video anotado. Si el patron se repite, '
                      'consultalo con un kinesiologo o un entrenador.')
        fundamento = ('La deteccion esta validada contra la etiqueta real del dataset; la '
                      'tipificacion se resuelve con criterios cinematicos de umbral citado. '
                      'Cuando la primera marca algo que los segundos no explican, el sistema lo '
                      'reporta en vez de elegir una etiqueta al azar.')
        referencia = ''
    else:
        if detecta_desviacion:
            avisos.append((
                'El modelo y las medidas no coinciden',
                'La red marco esta ejecucion como desviada, pero los '
                f"{len(cinematica['evaluables'])} criterios medibles con este encuadre quedaron "
                'dentro de su rango de referencia. Se reporta lo que dicen las medidas. '
                'La red se entreno con un conjunto donde las repeticiones incorrectas eran '
                'tambien las mas profundas, asi que tiende a penalizar la profundidad; el '
                'detalle tecnico muestra los dos resultados.'))
        etiqueta_clase = 'correcto'
        limites = [h for h in cinematica['hallazgos'] if h['veredicto'] == 'limite']
        zona = '—'
        correccion = ('Mantén el patrón: todos los criterios evaluables quedaron dentro de su '
                      'rango de referencia.'
                      + ('' if not limites else
                         ' En el límite: ' + ', '.join(h['nombre'].lower() for h in limites) + '.'))
        fundamento = ('Los criterios se contrastan contra rangos de referencia publicados y se '
                      'reportan con el valor medido, para que puedas verificarlos.')
        referencia = CLINICAL['citations'].get('schoenfeld2010', '')

    # Lectura visual: esqueleto sobre el video con la zona del error resaltada.
    try:
        if criterio_fuera:
            titulo_overlay = 'Detectado: ' + principal['nombre']
            focos_overlay = FOCOS_POR_CRITERIO.get(principal['clave'], [])
        elif sin_tipificar:
            titulo_overlay, focos_overlay = 'Desviacion detectada, sin tipificar', []
        else:
            titulo_overlay, focos_overlay = None, None
        video_anotado, imagen_clave = render_pose_overlay(
            video_path, exercise, 0 if titulo_overlay else clase, zona,
            titulo_forzado=titulo_overlay, focos_forzados=focos_overlay)
    except Exception as error:   # la evaluacion ya es valida: el overlay es un extra
        print(f'[aviso] no se pudo generar el overlay: {error}')
        video_anotado, imagen_clave = None, None

    return {
        'estado': 'evaluado',
        'ejercicio': exercise,
        'ejercicio_label': etiqueta_es(exercise),
        'clase': clase,
        'clase_nombre': etiqueta_clase,
        # La ejecucion es correcta solo si NINGUN criterio medido se sale de
        # rango y la red tampoco marca desviacion.
        'es_correcto': not criterio_fuera and not sin_tipificar,
        'sin_tipificar': bool(sin_tipificar),
        'cinematica': cinematica,
        'deteccion_red': {'marca_desviacion': bool(detecta_desviacion),
                          'clase_red': nombres[clase].replace('_', ' '),
                          'p_error': float(1.0 - probabilidades[0])},
        'confianza': float(probabilidades[clase]),
        'cobertura': float(cobertura),
        'zona': zona,
        'correccion': correccion,
        'fundamento': fundamento,
        'referencia': referencia,
        'reparto': {nombres[i].replace('_', ' '): float(pr)
                    for i, pr in enumerate(probabilidades)},
        'avisos': avisos,
        'metricas': metricas_modelo or None,
        'vista': info_vista,
        'imagen': imagen_clave,
        'video': video_anotado,
    }


DESCARGO = ('Este sistema apoya la tecnica observable y no constituye un diagnostico medico; '
            'no reemplaza la evaluacion de un profesional de salud o actividad fisica.')


def analizar(video_path, etiqueta_ejercicio):
    """Version en markdown del diagnostico, para la interfaz Gradio."""
    r = evaluar(video_path, etiqueta_ejercicio)
    if r['estado'] == 'sin_video':
        return '### Sube o graba un video para empezar.', {}, None, None
    if r['estado'] == 'ilegible':
        return f"### No se pudo procesar el video\n\n{r['mensaje']}", {}, None, None
    if r['estado'] in ('sin_cobertura', 'fuera_de_encuadre'):
        return f"### No se pudo evaluar\n\n{r['mensaje']}", {}, None, None

    if r['es_correcto']:
        cuerpo = (f"## Ejecucion correcta — {r['ejercicio_label']}\n\n"
                  f"**Confianza:** {r['confianza'] * 100:.1f}%  \n"
                  f"**Cobertura de postura:** {r['cobertura'] * 100:.0f}%\n\n"
                  f"{r['correccion']}\n\n"
                  f"**Por que importa:** {r['fundamento']}\n")
    else:
        encabezado = ('Desviacion tecnica detectada, sin tipificar' if r.get('sin_tipificar')
                      else f"Se detecto: {r['clase_nombre']}")
        cuerpo = (f"## {encabezado}\n\n"
                  f"**Ejercicio:** {r['ejercicio_label']}  \n"
                  f"**Confianza:** {r['confianza'] * 100:.1f}%  \n"
                  f"**Cobertura de postura:** {r['cobertura'] * 100:.0f}%\n\n"
                  f"**Zona biomecanica:** {r['zona']}\n\n"
                  f"**Correccion:** {r['correccion']}\n\n"
                  f"**Por que importa:** {r['fundamento']}\n")
    if r['referencia']:
        cuerpo += f"\n**Referencia:** {r['referencia']}\n"
    for titulo_aviso, detalle in r['avisos']:
        cuerpo += f"\n> **{titulo_aviso}:** {detalle}\n"
    m = r['metricas']
    if m:
        # Un diagnostico sin su margen de error invita a creerle mas de lo que vale.
        cuerpo += (f"\n---\n**Rendimiento validado de este modelo** (validacion por sujeto, "
                   f"{m['muestras']} muestras de {m['sujetos']} personas): detecta el "
                   f"{m['deteccion_sensibilidad'] * 100:.0f}% de las ejecuciones incorrectas; "
                   f"accuracy balanceada {m.get('accuracy_balanceada', float('nan')):.2f}; "
                   f"F1-macro de tipificacion {m['tipificacion_f1_macro']:.2f}.\n")
    cuerpo += f"\n---\n*{DESCARGO}*"
    return cuerpo, r['reparto'], r['imagen'], r['video']


