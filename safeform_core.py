"""SafeForm AI — app de evaluacion biomecanica.

Se ejecuta igual en Colab (Seccion 3.3 del notebook de entrenamiento) y como
Hugging Face Space. Un modelo por ejercicio, entrenado en ese notebook.
"""
import json
import subprocess
import tempfile
from pathlib import Path

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
    return secuencia, cobertura, visibilidades.mean(axis=0)


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
                         lado_maximo=720):
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

    focos = CLINICAL['focus_landmarks'][exercise][str(class_id)]
    nombre_clase = CLINICAL['taxonomy'][exercise][str(class_id)]
    es_correcto = nombre_clase == 'correcto'
    titulo = 'Ejecucion correcta' if es_correcto else f"Detectado: {nombre_clase.replace('_', ' ')}"
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
    """Acepta tanto modelos/<ejercicio>/ como <ejercicio>/ en la raiz."""
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
        secuencia, cobertura, visibilidad = extract_video_sequence(
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
    guia = CLINICAL['knowledge_base'][exercise][str(clase)]

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

    # Lectura visual: esqueleto sobre el video con la zona del error resaltada.
    try:
        video_anotado, imagen_clave = render_pose_overlay(
            video_path, exercise, clase, guia['zona_biomecanica'])
    except Exception as error:   # la evaluacion ya es valida: el overlay es un extra
        print(f'[aviso] no se pudo generar el overlay: {error}')
        video_anotado, imagen_clave = None, None

    return {
        'estado': 'evaluado',
        'ejercicio': exercise,
        'ejercicio_label': etiqueta_es(exercise),
        'clase': clase,
        'clase_nombre': nombres[clase].replace('_', ' '),
        'es_correcto': nombres[clase] == 'correcto',
        'confianza': float(probabilidades[clase]),
        'cobertura': float(cobertura),
        'zona': guia['zona_biomecanica'],
        'correccion': guia['instruccion_correctiva'],
        'fundamento': guia['fundamento_medico'],
        'referencia': CLINICAL['citations'].get(guia['citation_key'], ''),
        'reparto': {nombres[i].replace('_', ' '): float(pr)
                    for i, pr in enumerate(probabilidades)},
        'avisos': avisos,
        'metricas': bundle['class_map'].get('metricas_validacion'),
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
        cuerpo = (f"## Se detecto: {r['clase_nombre']}\n\n"
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


