# Plan de Acción ante Drift — DengAI MLOps

Este documento es el entregable de la **Fase 5** (RAA4): qué hace el sistema, en concreto,
cuando `evaluar_alertas()` (Fase 4, notebook de drift) detecta drift o degradación de
desempeño. El gatillo está **implementado**, no solo descrito — ver `service/reentrenar.py`.

## Los tres niveles de alerta

`evaluar_alertas()` cruza dos señales independientes (drift de variables vía PSI, y
desempeño vía MAE) y devuelve una de tres acciones:

| Acción | Condición | Qué significa |
|---|---|---|
| **OK** | Ni el drift ni el desempeño superan sus umbrales | El modelo sigue siendo confiable en esta ventana |
| **REVISAR** | Drift **o** desempeño (no ambos) superan el umbral | Señal aislada — puede ser ruido de muestreo (ventanas de 26-47 semanas, ver Fase 4 Paso 3) o el inicio real de un cambio |
| **REENTRENAR** | Drift **y** desempeño superan el umbral simultáneamente | Ambas señales coinciden — evidencia más fuerte de que el modelo dejó de ajustarse a la realidad actual |

## Qué hace el equipo en cada caso

### OK
Ninguna acción. El sistema sigue monitoreando la siguiente ventana con normalidad.

### REVISAR
1. El dashboard (`dashboard/dashboard.py`) marca la ventana en amarillo — el equipo la revisa
   manualmente al abrir el dashboard, no hay una notificación push automática (fuera del
   alcance de este práctico, ver sección "Fuera de alcance").
2. **No se reentrena automáticamente.** Decisión deliberada: en la Fase 4 comprobamos que el
   PSI es sensible al tamaño de ventana (ventanas chicas → falsos positivos más probables).
   Reentrenar cada vez que aparece una señal aislada arriesga perseguir ruido en vez de una
   señal real, y cada reentrenamiento tiene un costo (cómputo, riesgo de overfitting a una
   ventana chica, pérdida de trazabilidad de versiones).
3. Se aumenta la frecuencia de revisión: en vez de esperar a la ventana completa siguiente,
   el equipo revisa el dashboard con la próxima porción de datos disponible, para confirmar
   si la señal persiste o fue puntual.

### REENTRENAR (gatillo semi-automático)
1. El equipo confirma la alerta en el dashboard e identifica la ciudad y la(s) ventana(s)
   involucradas (columnas `city` / `ventana` de `reports/alertas_por_ventana.csv`).
2. Ejecuta manualmente:
   ```bash
   cd service
   python reentrenar.py --city <sj|iq> --ventanas <V1 V2 ...>
   ```
3. El script (`service/reentrenar.py`):
   - Incorpora esa(s) ventana(s) —que ya tienen etiqueta verdadera, porque vienen del propio
     dataset histórico reservado en la Fase 2— al set de referencia.
   - Reentrena el pipeline completo con la misma arquitectura y validación temporal de la
     Fase 2 (`src/modelo.py`, sin fuga: `TimeSeriesSplit`).
   - **Guardrail automático**: solo promueve el modelo nuevo a producción si su MAE de
     validación no empeora más de un 10% respecto al modelo vigente. Si empeora más que eso,
     lo deja en `models/candidatos/` para revisión manual y el modelo en producción NO cambia
     — evita que un reentrenamiento mal calibrado degrade el servicio automáticamente.
   - Si promueve, hace un **backup versionado** del modelo anterior en `models/archivo/`
     antes de sobrescribir `models/modelo_dengue.joblib`, y registra el evento en
     `models/historial_versiones.json`.
4. El equipo verifica el nuevo MAE reportado por el script y, si todo luce bien, deja el
   modelo nuevo en producción. El dashboard usará automáticamente el modelo/metadata más
   reciente la próxima vez que se regenere `reports/` (correr el notebook de la Fase 4 de
   nuevo con el modelo actualizado).

### Rollback
Si, después de promover un modelo nuevo, el monitoreo de las siguientes ventanas muestra que
el desempeño empeoró en vez de mejorar, el equipo ejecuta:
```bash
cd service
python reentrenar.py --rollback
```
Esto restaura la última versión respaldada en `models/archivo/` como el modelo en producción,
y deja registro en `models/historial_versiones.json`.

## Por qué semi-automático y no 100% automático

El enunciado no exige automatizar el reentrenamiento completo, y decidimos explícitamente
**no** hacerlo por dos razones, ambas evidenciadas en la Fase 4:

1. El umbral de PSI puede dispararse por tamaño de ventana chico, no solo por drift real
   (Fase 4, Paso 3) — un reentrenamiento 100% automático perseguiría ese ruido.
2. Reentrenar con datos de una sola ventana de outbreak severo (ej. si coincidiera con un
   evento como el de SJ 1994) podría sesgar el modelo hacia ese evento puntual sin que nadie
   lo revise antes de que quede en producción.

Un humano en el loop antes de promover un modelo nuevo es, para el tamaño y criticidad de
este proyecto, una defensa más razonable que la automatización completa.

## Fuera de alcance (documentado, no implementado)

- Notificaciones push/email/Slack cuando se dispara una alerta (el dashboard es "pull", el
  equipo lo revisa activamente).
- Reentrenamiento con datos de AMBAS ciudades combinadas automáticamente (el script opera
  por ciudad, consistente con la decisión de la Fase 1/2 de tratar `city` como variable
  estructural).
- CI/CD o infraestructura cloud (explícitamente fuera de alcance del enunciado).
