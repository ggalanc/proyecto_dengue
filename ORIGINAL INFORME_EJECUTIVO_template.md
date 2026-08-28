# Informe Ejecutivo — Sistema de Predicción de Dengue con Monitoreo de Drift

> **ESTA ES UNA PLANTILLA, NO EL INFORME FINAL.**
> El enunciado exige que el informe ejecutivo lo escriba el equipo (Gerardo y Mabel) con sus
> propias palabras, para una audiencia de negocio (no técnica). Este archivo NO debe
> entregarse tal cual — la IA solo puede dar soporte de "vibe coding" (código, no la
> redacción final). Abajo dejamos: (1) la estructura sugerida con las preguntas que cada
> sección debe responder, y (2) los números y hallazgos reales del proyecto para que ustedes
> los interpreten y redacten con sus propias palabras. Borren estas notas en itálica y los
> corchetes `[...]` antes de entregar. Máximo 3 páginas.

---

## 1. Problemática y por qué importa (≈ media página)

*Preguntas a responder, en lenguaje de negocio:*
- ¿Qué decide un ministerio de salud / servicio de vigilancia epidemiológica con una
  predicción semanal de casos de dengue? (ej. asignar personal, fumigación, camas de hospital)
- ¿Qué pasa si el modelo se "queda dormido" — sigue funcionando pero silenciosamente deja de
  predecir bien porque el clima o la dinámica de la enfermedad cambiaron? ¿Por qué un modelo
  que nadie monitorea es un riesgo operativo, no solo un problema técnico?
- Referencia: esto ya lo desarrollaron en la Fase 0 de su notebook — pueden resumir esa
  justificación aquí, no repetirla entera.

## 2. Qué construimos (≈ media página)

*Resumir sin jerga técnica excesiva. Datos reales para citar:*
- Predicción semanal de casos de dengue para dos ciudades: San Juan (Puerto Rico) e Iquitos
  (Perú), usando el dataset público DengAI (Kaggle `arashnic/epidemy`).
- Un pipeline completo y local (sin depender de servicios en la nube): entrenamiento →
  servicio que recibe datos semanales y predice → monitoreo automático de si el modelo
  "se está desactualizando" → un plan de acción concreto (no solo teórico) para cuándo y
  cómo reentrenarlo.
- El modelo usa como principal señal predictiva el clima de las semanas **anteriores**
  (humedad, temperatura, precipitación, hasta 12 semanas atrás) — es una decisión de diseño
  importante: el dengue tiene un retraso biológico entre el clima favorable para el mosquito
  y el aumento de casos, y el modelo está construido para respetar ese retraso sin "hacer
  trampa" mirando información que no existiría en el momento real de la predicción.

## 3. Resultados (≈ 1 página)

*Interpreten estos números — no los peguen sin explicar qué significan para alguien que no
sabe qué es un MAE o un PSI.*

**Desempeño del modelo (error de predicción, MAE = promedio de cuántos casos por semana
se equivoca el modelo):**
- MAE de validación (entrenamiento, `TimeSeriesSplit`): **≈24.1 casos/semana** (ver
  `models/metadata.json`).
- En la producción simulada (datos que el modelo nunca vio durante el entrenamiento), el
  error varió por ciudad y ventana temporal — ver `reports/mae_por_ventana.csv`:
  - San Juan: MAE entre ~10 y ~27 casos/semana según la ventana (70%-84% del promedio real
    de esa ventana).
  - Iquitos: MAE entre ~3 y ~15 casos/semana (70%-102% del promedio real de esa ventana).
  - *[Equipo: interpreten por qué Iquitos tiene errores relativos más variables — ciudad con
    menos datos históricos (416 semanas de referencia vs. 748 de SJ) y series más cortas.]*

**Monitoreo de drift (¿el mundo real se parece a los datos con los que se entrenó el
modelo?):**
- Usamos dos métricas estándar de la industria: PSI (Population Stability Index) y el test
  de Kolmogorov-Smirnov, aplicadas a 4 ventanas temporales de producción simulada por ciudad.
- Hallazgo más importante para explicar en el informe: **en las 8 combinaciones
  ciudad×ventana analizadas, el sistema detectó drift en al menos una variable climática en
  todas ellas, pero el desempeño del modelo NO se degradó de forma proporcional** (ver
  `reports/alertas_por_ventana.csv` — las 8 ventanas quedaron en nivel "REVISAR", ninguna
  llegó a "REENTRENAR" con el umbral usado). *[Equipo: esto es un hallazgo real, no un
  defecto que hay que esconder. Explíquenlo como una lección: el drift estadístico (los
  datos se ven distintos) no siempre implica drift de negocio (el modelo predice peor). Es
  la razón por la que diseñamos el reentrenamiento como semi-automático — ver sección 4.]*
- También documentamos que el PSI es sensible al tamaño de la ventana: con muestras chicas
  (26 semanas en Iquitos) el PSI puede inflarse por ruido de muestreo, no solo por drift
  real. Por eso ajustamos el número de bins usado en el cálculo (ver notebook de Fase 4,
  Paso 3, para la comparación empírica que hicimos).

## 4. Plan de acción ante drift (≈ media página)

*Resumir en lenguaje simple lo que ya está en `PLAN_DE_ACCION.md` — no technical detail,
solo el "qué hacemos cuando pasa X":*
- Tres niveles de alerta: **OK** (todo normal), **REVISAR** (una señal aislada, se
  monitorea más de cerca sin reentrenar todavía), **REENTRENAR** (drift y caída de
  desempeño coinciden — ahí sí se actualiza el modelo).
- El reentrenamiento es semi-automático a propósito: el equipo lo dispara manualmente
  (`reentrenar.py`) después de confirmar la alerta, y el sistema **no promueve** un modelo
  nuevo a producción si empeora el error más de 10% — queda guardado para revisión en vez de
  romper el servicio. *[Equipo: expliquen por qué "un humano en el loop" es la decisión
  correcta para este proyecto — pueden usar el hallazgo de la sección 3 (drift detectado sin
  caída de desempeño) como la justificación central.]*
- Existe un mecanismo de reversión (`rollback`) probado, por si un modelo promovido resulta
  peor de lo esperado una vez en producción.

## 5. Limitaciones y próximos pasos (≈ media página)

*Ver la sección "Limitaciones conocidas" de `README.md` y "Fuera de alcance" de
`PLAN_DE_ACCION.md` — tradúzcanlas a impacto de negocio, no a jerga técnica. Por ejemplo:*
- No hay notificaciones automáticas (push/email) cuando se dispara una alerta — hoy alguien
  del equipo tiene que abrir el dashboard.
- Los umbrales de alerta (cuánto drift o cuánto error es "demasiado") están justificados con
  el análisis que hicimos, pero no fueron calibrados con un experimento formal — es una
  mejora futura razonable antes de un uso en producción real.
- *[Equipo: agreguen cualquier otra limitación que quieran destacar para la defensa oral.]*

---

*Fuentes de los números de este documento: `models/metadata.json`,
`reports/mae_por_ventana.csv`, `reports/alertas_por_ventana.csv`,
`reports/drift_detalle.csv`, y los notebooks de las Fases 1, 2 y 4.*
