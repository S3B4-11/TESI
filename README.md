---
title: SafeForm AI
emoji: "🏋"
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# SafeForm AI — Evaluacion biomecanica de ejercicios

Sube un video de tu ejecucion y el sistema identifica el error biomecanico especifico, la correccion a aplicar y el fundamento clinico.

Ejercicios disponibles: Flexión de codo (curl de bíceps), Zancada / estocada, Abducción de hombro (elevación lateral), Sentadilla.

Arquitectura del modelo: LSTM (seleccionada por F1-macro en validacion Leave-One-Subject-Out). Entrenado con UI-PRMD (Vicon) e IntelliRehabDS (Kinect v2).

Este sistema apoya la tecnica observable y no constituye un diagnostico medico.
