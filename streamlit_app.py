"""SafeForm AI — interfaz Streamlit (Streamlit Community Cloud).

La logica de inferencia vive completa en safeform_core: aqui solo hay
presentacion. Las dos interfaces llaman a la misma funcion evaluar(), asi que
no pueden dar resultados distintos.
"""
import tempfile
from pathlib import Path

import streamlit as st

from safeform_core import DESCARGO, ejercicios_disponibles, etiqueta_es, evaluar

st.set_page_config(page_title='SafeForm AI', page_icon='\U0001FA7A', layout='wide')

# Paleta clinica. Los colores de ESTADO (correcto / error) son independientes del
# acento de la app: el acento identifica, el semaforo comunica el resultado. Si
# compartieran color, un resultado correcto y uno incorrecto se leerian igual de
# lejos, que es justo lo que no puede pasar aqui.
st.markdown("""
<style>
  .bloque-estado {
    border-left: 4px solid var(--borde);
    background: var(--fondo);
    padding: 14px 18px; border-radius: 4px; margin-bottom: 18px;
  }
  .bloque-estado .etiqueta {
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: #5B6A6E; margin-bottom: 4px;
  }
  .bloque-estado .titulo { font-size: 22px; font-weight: 600; line-height: 1.25; }
  .ok   { --borde: #2C6E49; --fondo: #E9F2EC; }
  .malo { --borde: #8F5714; --fondo: #F6EEE2; }
  .campo { margin-bottom: 16px; }
  .campo .clave {
    font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: #5B6A6E; margin-bottom: 3px;
  }
  .campo .valor { font-size: 15px; line-height: 1.55; }
  .pie { font-size: 12px; color: #5B6A6E; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def catalogo():
    return {etiqueta_es(e): e for e in ejercicios_disponibles()}


CATALOGO = catalogo()

st.title('SafeForm AI')
st.caption('Evaluación biomecánica de ejercicios a partir de video. Identifica el error '
           'específico, la corrección y su fundamento clínico.')

if not CATALOGO:
    st.error('No hay modelos disponibles en la carpeta modelos/. Revisa el despliegue.')
    st.stop()

izquierda, derecha = st.columns([1, 1.35], gap='large')

with izquierda:
    etiqueta = st.selectbox('Ejercicio', list(CATALOGO.keys()))
    archivo = st.file_uploader('Video del ejercicio', type=['mp4', 'mov', 'avi', 'mkv'])
    evaluar_ahora = st.button('Evaluar', type='primary', use_container_width=True,
                              disabled=archivo is None)
    st.caption('**Para mejores resultados:** cuerpo entero visible, cámara fija, buena '
               'iluminación, una sola persona en cuadro y una repetición completa.')
    if archivo is not None:
        st.video(archivo)

with derecha:
    if not evaluar_ahora:
        st.info('Selecciona el ejercicio, sube un video y presiona **Evaluar**.')
        st.stop()

    sufijo = Path(archivo.name).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as temporal:
        temporal.write(archivo.getbuffer())
        ruta_video = temporal.name

    with st.spinner('Analizando la ejecución...'):
        r = evaluar(ruta_video, etiqueta)

    if r['estado'] != 'evaluado':
        titulos = {'ilegible': 'No se pudo procesar el video',
                   'sin_cobertura': 'No se pudo evaluar',
                   'fuera_de_encuadre': 'El cuerpo no está completo en cuadro',
                   'sin_video': 'Sube un video para empezar'}
        st.warning(f"**{titulos.get(r['estado'], 'Sin resultado')}**\n\n{r.get('mensaje', '')}")
        st.stop()

    correcto = r['es_correcto']
    if correcto:
        titulo_estado = 'Ejecución correcta'
    elif r.get('sin_tipificar'):
        # Hay una desviación, pero el subtipo no alcanza el rendimiento mínimo
        # para nombrarse. Decirlo así es más útil que dar un nombre poco fiable.
        titulo_estado = 'Desviación técnica detectada'
    else:
        titulo_estado = 'Detectado: ' + r['clase_nombre']
    st.markdown(
        '<div class="bloque-estado ' + ('ok' if correcto else 'malo') + '">'
        '<div class="etiqueta">' + r['ejercicio_label'] + '</div>'
        '<div class="titulo">' + titulo_estado + '</div></div>',
        unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric('Confianza', f"{r['confianza'] * 100:.1f} %")
    m2.metric('Cobertura de postura', f"{r['cobertura'] * 100:.0f} %",
              help='Porcentaje de fotogramas en que se detectó el cuerpo.')

    for titulo_aviso, detalle in r['avisos']:
        st.warning(f'**{titulo_aviso}.** {detalle}')

    if r['imagen']:
        st.image(r['imagen'], use_container_width=True,
                 caption='Instante más crítico del movimiento — en rojo, la zona evaluada')

    pestanas = st.tabs(['Diagnóstico', 'Video analizado', 'Detalle técnico'])

    with pestanas[0]:
        campos = [] if correcto else [('Zona biomecánica', r['zona'])]
        campos.append(('Recomendación' if correcto else 'Corrección', r['correccion']))
        campos.append(('Por qué importa', r['fundamento']))
        for clave, valor in campos:
            st.markdown('<div class="campo"><div class="clave">' + clave +
                        '</div><div class="valor">' + valor + '</div></div>',
                        unsafe_allow_html=True)
        if r['referencia']:
            st.markdown('<div class="pie"><b>Referencia:</b> ' + r['referencia'] + '</div>',
                        unsafe_allow_html=True)

    with pestanas[1]:
        if r['video']:
            st.video(r['video'])
        else:
            st.caption('No se pudo generar el video anotado.')

    with pestanas[2]:
        st.markdown('**Probabilidad por clase**')
        for nombre, valor in sorted(r['reparto'].items(), key=lambda x: -x[1]):
            st.progress(min(max(float(valor), 0.0), 1.0), text=f'{nombre} — {valor:.1%}')
        met = r['metricas']
        if met:
            # Un diagnostico sin su margen de error invita a creerle mas de lo que vale.
            st.markdown('**Rendimiento validado de este modelo**')
            st.caption(f"Validación por sujeto sobre {met['muestras']} muestras de "
                       f"{met['sujetos']} personas.")
            d1, d2, d3 = st.columns(3)
            d1.metric('Detección de errores', f"{met['deteccion_sensibilidad'] * 100:.0f} %",
                      help='De las ejecuciones realmente incorrectas, cuántas detecta.')
            d2.metric('Accuracy balanceada',
                      f"{met.get('accuracy_balanceada', float('nan')):.2f}",
                      help='Promedio de sensibilidad y especificidad. Su línea base es 0,50.')
            d3.metric('F1-macro tipificación', f"{met['tipificacion_f1_macro']:.2f}",
                      help='Qué tan bien distingue entre los subtipos de error.')

    st.divider()
    st.markdown('<div class="pie"><i>' + DESCARGO + '</i></div>', unsafe_allow_html=True)
