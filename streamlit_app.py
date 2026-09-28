"""SafeForm AI — interfaz Streamlit (Streamlit Community Cloud).

La logica de inferencia vive completa en safeform_core: aqui solo hay
presentacion. Las dos interfaces llaman a la misma funcion evaluar(), asi que
no pueden dar resultados distintos.

DISENO. La pantalla de resultado sigue el patron de un informe de laboratorio:
cada criterio es una fila con su valor medido, la zona de referencia sombreada
y una aguja donde cayo la medicion. Es el mismo problema —un valor contra un
rango normal— y se entiende sin leer los numeros.

El orden es veredicto, evidencia, medidas, correccion. El detalle tecnico va
plegado al final: quien abre la app quiere saber si su ejecucion estuvo bien,
no arbitrar un desacuerdo entre dos subsistemas.
"""
import tempfile
from pathlib import Path

import streamlit as st

from safeform_core import DESCARGO, ejercicios_disponibles, etiqueta_es, evaluar

st.set_page_config(page_title='SafeForm AI', page_icon='\U0001FA7A', layout='wide')

# Paleta clinica. Los colores de ESTADO (en rango / limite / fuera de rango) son
# independientes del acento de la app: el acento identifica, el semaforo
# comunica el resultado. Si compartieran color, un resultado bueno y uno malo se
# leerian igual de lejos, que es justo lo que no puede pasar aqui.
st.markdown("""
<style>
  :root{
    --ink:#12181A; --muted:#56666A; --faint:#8A999D;
    --rule:#DBE3E3; --rule-soft:#EAEFEF; --surface-2:#E9EEEE;
    --pass:#2F6B4F; --pass-bg:#E3EFE7; --pass-track:#C4DECF;
    --warn:#8A5A12; --warn-bg:#F5ECDC; --warn-track:#E4CFA8;
    --fail:#94353B; --fail-bg:#F4E3E3; --fail-track:#E2BEBF;
    --zona:#C7DCD0; --zona-borde:#A9C9BA;
  }
  .bloque-estado{
    border-left:5px solid var(--borde); background:var(--fondo);
    padding:18px 20px 20px; border-radius:6px; margin-bottom:18px;
  }
  .bloque-estado .etiqueta{
    font-size:11px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--muted); margin-bottom:6px;
  }
  .bloque-estado .titulo{ font-size:26px; font-weight:600; line-height:1.15; color:var(--tono); }
  .bloque-estado .resumen{ font-size:15px; margin-top:8px; color:var(--ink); max-width:58ch; }
  .ok   { --borde:var(--pass); --fondo:var(--pass-bg); --tono:var(--pass); }
  .malo { --borde:var(--fail); --fondo:var(--fail-bg); --tono:var(--fail); }
  .duda { --borde:var(--warn); --fondo:var(--warn-bg); --tono:var(--warn); }

  /* Criterios: patron de informe de laboratorio */
  .criterio{ padding:14px 0; border-bottom:1px solid var(--rule-soft); }
  .criterio:last-child{ border-bottom:0; }
  .criterio .fila{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
  .criterio .nombre{ font-weight:600; font-size:14.5px; }
  .criterio .medida{
    margin-left:auto; font-family:ui-monospace,Consolas,monospace; font-size:15px;
    font-weight:600; font-variant-numeric:tabular-nums; color:var(--tono); white-space:nowrap;
  }
  .criterio .chip{
    font-size:10px; letter-spacing:.07em; text-transform:uppercase;
    padding:3px 7px; border-radius:3px; background:var(--chip); color:var(--tono);
  }
  .escala{ position:relative; height:20px; margin-top:8px; }
  .via{ position:absolute; inset:6px 0 auto; height:7px; border-radius:4px; background:var(--surface-2); }
  .banda{ position:absolute; top:6px; height:7px; border-radius:4px;
         background:var(--zona); box-shadow:inset 0 0 0 1px var(--zona-borde); }
  .aguja{ position:absolute; top:0; width:4px; height:19px; border-radius:2px;
          background:var(--tono); transform:translateX(-2px);
          box-shadow:0 0 0 2px #fff; }
  .topes{ display:flex; justify-content:space-between; gap:10px;
          font-size:10.5px; color:var(--faint); }
  .topes .ref{ color:var(--muted); }
  .criterio .pista{ font-size:12.5px; color:var(--muted); margin-top:5px; }
  .en_rango      { --tono:var(--pass); --chip:var(--pass-bg); }
  .limite        { --tono:var(--warn); --chip:var(--warn-bg); }
  .fuera_de_rango{ --tono:var(--fail); --chip:var(--fail-bg); }
  .no_evaluable  { --tono:var(--faint); --chip:var(--surface-2); }
  .no_evaluable .medida{ font-size:12.5px; font-style:italic; font-weight:400; }
  .no_evaluable .escala{ opacity:.45; }

  .campo{ margin-bottom:14px; }
  .campo .clave{
    font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--muted); margin-bottom:4px;
  }
  .campo .valor{ font-size:14.5px; line-height:1.55; }
  .pie{ font-size:12px; color:var(--muted); line-height:1.5; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def catalogo():
    return {etiqueta_es(e): e for e in ejercicios_disponibles()}


CATALOGO = catalogo()

ETIQUETA_ESTADO = {'en_rango': 'En rango', 'limite': 'En el límite',
                   'fuera_de_rango': 'Fuera de rango', 'no_evaluable': 'Sin medir'}


def posicion(valor, escala):
    """Porcentaje dentro de la escala, acotado para que la aguja no se salga."""
    minimo, maximo = escala
    if maximo <= minimo:
        return 0.0
    return max(0.0, min(100.0, (float(valor) - minimo) / (maximo - minimo) * 100.0))


def _numero(valor, unidad, recortar=False):
    """Los grados van con un decimal; el valgo, que vive entre 0 y 0,35, con dos."""
    texto = f'{valor:.1f}' if unidad.startswith('\u00b0') else f'{valor:.2f}'
    if recortar and float(valor) == int(valor):
        texto = str(int(valor))
    return texto.replace('.', ',')


def formatear(valor, unidad):
    """El grado va pegado al numero (112,1°), el resto separado (0,10 del ancho...)."""
    junto = unidad.startswith('\u00b0')
    return f"{_numero(valor, unidad)}{'' if junto else ' '}{unidad}"


def tope(valor, unidad):
    """Extremo de la escala: solo el numero. La unidad ya esta en el valor medido."""
    return _numero(valor, unidad, recortar=True) + ('\u00b0' if unidad.startswith('\u00b0') else '')


def dibujar_criterio(h):
    """Una fila del informe: nombre, valor, barra con la zona de referencia."""
    estado = h['veredicto']
    escala = h.get('escala') or [0.0, 1.0]
    banda = h.get('banda') or [0.0, 0.0]
    desde, hasta = posicion(banda[0], escala), posicion(banda[1], escala)
    medida = ('medición descartada' if h.get('motivo_no_evaluable') == 'implausible'
              else 'requiere otro encuadre' if estado == 'no_evaluable'
              else formatear(h['valor'], h['unidad']))
    if estado == 'no_evaluable' and h.get('motivo_no_evaluable') == 'implausible':
        # No es el encuadre: es que el numero salio fuera de lo fisicamente
        # posible. Mandarlo a grabar del otro lado seria mandarlo a repetir.
        pista = ('La medición salió fuera de todo rango físico posible y se descartó. '
                 'Suele pasar cuando parte del cuerpo queda fuera de cuadro.')
    elif estado == 'no_evaluable':
        encuadre = {'frontal': 'de frente', 'sagital': 'de perfil'}.get(h.get('plano'))
        pista = ('Se mide en el plano ' + h['plano'] + ': hace falta grabar '
                 + encuadre + '.') if encuadre else 'Este encuadre no permite medirlo.'
    else:
        pista = h.get('lectura') or ''
    aguja = ('' if estado == 'no_evaluable'
             else f'<div class="aguja" style="left:{posicion(h["valor"], escala):.1f}%"></div>')
    st.markdown(
        f'<div class="criterio {estado}">'
        f'<div class="fila"><span class="nombre">{h["nombre"]}</span>'
        f'<span class="chip">{ETIQUETA_ESTADO[estado]}</span>'
        f'<span class="medida">{medida}</span></div>'
        f'<div class="escala"><div class="via"></div>'
        f'<div class="banda" style="left:{desde:.1f}%;width:{max(hasta - desde, 0):.1f}%"></div>'
        f'{aguja}</div>'
        f'<div class="topes"><span>{tope(escala[0], h["unidad"])}</span>'
        f'<span class="ref">{h["referencia"]}</span>'
        f'<span>{tope(escala[1], h["unidad"])}</span></div>'
        + (f'<div class="pista">{pista}</div>' if pista else '')
        + '</div>', unsafe_allow_html=True)


st.title('SafeForm AI')
st.caption('Evaluación biomecánica de ejercicios a partir de video. Identifica el error '
           'específico, la corrección y su fundamento clínico.')

if not CATALOGO:
    st.error('No hay modelos disponibles en la carpeta modelos/. Revisa el despliegue.')
    st.stop()

izquierda, derecha = st.columns([1, 1.45], gap='large')

with izquierda:
    etiqueta = st.selectbox('Ejercicio', list(CATALOGO.keys()))
    archivo = st.file_uploader('Video del ejercicio', type=['mp4', 'mov', 'avi', 'mkv'])
    evaluar_ahora = st.button('Evaluar', type='primary', use_container_width=True,
                              disabled=archivo is None)
    st.caption('**Para mejores resultados:** cuerpo entero visible, cámara fija, buena '
               'iluminación, una sola persona en cuadro y una repetición completa. '
               'De perfil se evalúa la profundidad; de frente, la alineación de rodillas.')
    # El video propio solo se muestra ANTES de evaluar: despues compite con la
    # foto del instante analizado, que es la que aporta informacion.
    if archivo is not None and not evaluar_ahora:
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

    cine = r.get('cinematica') or {}
    hallazgos = cine.get('hallazgos') or []
    correcto = r['es_correcto']

    # 1. VEREDICTO ----------------------------------------------------------
    incierto = bool(r.get('sin_tipificar'))
    if correcto:
        titulo_estado = 'Ejecución correcta'
        medibles = len(cine.get('evaluables') or [])
        resumen = ('No se detectaron desviaciones.' if not medibles else
                   'El único criterio que este encuadre permite medir quedó dentro de su rango '
                   'de referencia.' if medibles == 1 else
                   f'Los {medibles} criterios que este encuadre permite medir quedaron dentro '
                   'de su rango de referencia.')
    elif r.get('sin_tipificar'):
        # No es lo mismo "esta mal" que "no lo puedo comprobar". El sistema solo
        # llega aqui cuando el criterio que define el ejercicio quedo sin medir,
        # asi que decirlo como un diagnostico seria mentir por omision.
        titulo_estado = 'No se pudo verificar'
        resumen = (r['correccion'] if r.get('correccion') else
                   'Falta la medida que define este ejercicio.')
    else:
        titulo_estado = r['clase_nombre'].capitalize()
        principal = cine.get('principal') or {}
        resumen = (f"Se midió {formatear(principal['valor'], principal['unidad'])}. "
                   f"Referencia: {principal['referencia']}."
                   if principal else r['correccion'])
    st.markdown(
        '<div class="bloque-estado ' + ('ok' if correcto else 'duda' if incierto else 'malo') + '">'
        '<div class="etiqueta">' + r['ejercicio_label']
        + (f" · vista {cine['plano']}" if cine.get('plano') not in (None, 'indeterminado') else '')
        + '</div><div class="titulo">' + titulo_estado + '</div>'
        '<div class="resumen">' + resumen + '</div></div>', unsafe_allow_html=True)

    for titulo_aviso, detalle in r['avisos']:
        st.warning(f'**{titulo_aviso}.** {detalle}')

    # 2. EVIDENCIA: el fotograma donde se midio -----------------------------
    if r.get('imagen'):
        foto, texto = st.columns([1, 1.5], gap='medium')
        foto.image(r['imagen'], use_container_width=True)
        texto.markdown(
            '<div class="pie">Este es el <b>punto más bajo</b> de la repetición: el fotograma '
            'donde se tomaron las medidas de abajo.'
            + ('' if correcto or incierto else
               ' En rojo, las articulaciones del criterio que se salió de rango.')
            + f"<br><br>Cuerpo detectado en el {r['cobertura'] * 100:.0f}% de los fotogramas."
            + '</div>', unsafe_allow_html=True)
        st.write('')

    # 3. MEDIDAS ------------------------------------------------------------
    if hallazgos:
        st.markdown('##### Lo que se midió')
        for h in hallazgos:
            dibujar_criterio(h)
        if r.get('nota_encuadre'):
            st.caption(r['nota_encuadre'])
        st.write('')

    # 4. CORRECCION ---------------------------------------------------------
    st.markdown('##### ' + ('Recomendación' if correcto else
                            'Qué hacer' if incierto else 'Qué corregir'))
    campos = [] if correcto or incierto else [('Zona', r['zona'])]
    campos.append(('Mantén' if correcto else 'Cómo', r['correccion']))
    campos.append(('Por qué importa', r['fundamento']))
    for clave, valor in campos:
        st.markdown('<div class="campo"><div class="clave">' + clave +
                    '</div><div class="valor">' + valor + '</div></div>',
                    unsafe_allow_html=True)
    if r['referencia']:
        st.markdown('<div class="pie"><b>Referencia:</b> ' + r['referencia'] + '</div>',
                    unsafe_allow_html=True)

    # 5. DETALLE TECNICO ----------------------------------------------------
    with st.expander('Detalle técnico'):
        if cine.get('plano'):
            st.markdown(f"**Plano evaluado:** {cine['plano']}."
                        + (f" {cine['motivo_plano']}" if cine.get('motivo_plano') else ''))
        red = r.get('deteccion_red') or {}
        if red:
            st.markdown('**Qué dijo el modelo de Deep Learning**')
            st.caption(
                ('Marcó desviación' if red['marca_desviacion'] else 'No marcó desviación')
                + f" (probabilidad de error {red['p_error']:.0%}"
                + (f", subtipo más probable: {red['clase_red']}" if red['marca_desviacion'] else '')
                + '). El modelo decide si hay desviación; los criterios de arriba dicen cuál.')
        for titulo_aviso, detalle in r.get('avisos_tecnicos', []):
            st.markdown(f'**{titulo_aviso}**')
            st.caption(detalle)
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
                      help='Rendimiento del tipificador de la red, que ya no se usa para nombrar.')
        st.markdown('**Probabilidad por clase del modelo**')
        for nombre, valor in sorted(r['reparto'].items(), key=lambda x: -x[1]):
            st.progress(min(max(float(valor), 0.0), 1.0), text=f'{nombre} — {valor:.1%}')
        if r.get('video'):
            st.markdown('**Video analizado**')
            st.video(r['video'])

    st.divider()
    st.markdown('<div class="pie"><i>' + DESCARGO + '</i></div>', unsafe_allow_html=True)
