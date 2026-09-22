"""Interfaz Gradio (Hugging Face Space y Colab). La logica vive en safeform_core."""
import gradio as gr

from safeform_core import *          # noqa: F401,F403
from safeform_core import analizar, ejercicios_disponibles, etiqueta_es

EJERCICIOS = ejercicios_disponibles()
ETIQUETA_A_EJERCICIO = {etiqueta_es(e): e for e in EJERCICIOS}

with gr.Blocks(title='SafeForm AI') as demo:
    gr.Markdown('# SafeForm AI — evaluacion biomecanica de ejercicios\n'
                'Sube un video (o grabalo con la camara), elige el ejercicio y obten el error '
                'especifico, la correccion y su fundamento clinico.')
    with gr.Row():
        with gr.Column():
            entrada_video = gr.Video(label='Video del ejercicio', sources=['upload', 'webcam'])
            selector = gr.Dropdown(choices=list(ETIQUETA_A_EJERCICIO.keys()),
                                    value=list(ETIQUETA_A_EJERCICIO.keys())[0],
                                    label='Ejercicio')
            boton = gr.Button('Evaluar', variant='primary')
            gr.Markdown('**Para mejores resultados:** cuerpo entero visible, camara fija, '
                        'buena iluminacion, una sola persona en cuadro y una repeticion completa.')
        with gr.Column():
            salida_imagen = gr.Image(label='Instante mas critico (en rojo, la zona evaluada)',
                                      type='filepath')
            salida_texto = gr.Markdown('### Sube o graba un video para empezar.')
            salida_video = gr.Video(label='Analisis completo con el esqueleto detectado')
            salida_probabilidades = gr.Label(label='Probabilidad por clase', num_top_classes=4)

    boton.click(analizar, inputs=[entrada_video, selector],
                 outputs=[salida_texto, salida_probabilidades, salida_imagen, salida_video])

if __name__ == '__main__':
    demo.launch()
