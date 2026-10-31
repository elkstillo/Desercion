# Predicción de deserción estudiantil — Documentación

## 1. Funcionamiento general

La aplicación (Streamlit) estima el riesgo de deserción de un estudiante de educación superior a partir de 13 variables, usando una **SVM con kernel RBF** (`C=5`, `class_weight="balanced"`) guardada en `modelo-clas_Desercion.pkl`.

**Flujo**

1. Se carga una sola vez el artefacto `.pkl`: `(modelo, labelencoder, variables, min_max_scaler)`.
2. El usuario completa el formulario lateral y presiona **Predecir**.
3. Se arma un `DataFrame` de una fila y se prepara igual que en entrenamiento:
   `get_dummies` → `reindex` a las columnas del modelo (faltantes = 0) → `MinMaxScaler.transform` (solo `transform`, nunca `fit`).
4. El modelo devuelve la clase (`SI` = deserta / `NO`) y el **margen** (`decision_function`). A partir del margen se calcula un **índice de riesgo 0–100** (sigmoide del margen; 50 = frontera de decisión).
5. Se muestran el veredicto, el margen, la tabla de datos ingresados y dos análisis complementarios (sección 5).

**Variables de entrada**

| Tipo | Variables |
|---|---|
| Numéricas (escaladas) | `TERM_ID`, `GPA_TERM`, `CREDITS_ATTEMPTED_TERM`, `CREDITS_FAILED_TERM`, `GPA_CUM`, `CREDITS_APPROVED_CUM_ALL`, `CREDITS_FAILED_CUM` |
| Categóricas (dummies) | `COD_ESCUELA`, `ACADEMIC_STATUS_TERM`, `GENERO`, `RECIBIO_APOYO_FINANCIERO`, `RECIBIO_APOYO_ACADEMICO`, `RECIBIO_OTROS_APOYOS` |

## 2. Estructura del proyecto

```
├── app_Desercion.py          # Interfaz (Streamlit): estilos, formulario, gráficos
├── desercion_core.py         # Lógica: carga, preparación, predicción, sensibilidad
├── modelo-clas_Desercion.pkl # Artefacto del modelo (el suyo, sin cambios)
├── requirements.txt
└── .streamlit/config.toml    # Tema oscuro base
```

Ejecución: `pip install -r requirements.txt` y luego `streamlit run app_Desercion.py`.
`scikit-learn` está fijado en **1.6.1** porque es la versión con la que se guardó el `.pkl`.

## 3. Cambios y mejoras frente al código original

### Correcciones de fondo
| Original | Ahora |
|---|---|
| **Todos** los sliders tenían rango 14–52 y valor 20 (copiados de otra app). Un `GPA_TERM` de 20–52 o `TERM_ID` = 20 queda muy fuera del rango de entrenamiento. | Los rangos salen del propio `MinMaxScaler` (`data_min_` / `data_max_`): GPA 0–5, créditos inscritos 2–36, etc. `TERM_ID` usa un `number_input` limitado a 202520–202610. |
| La predicción se mostraba como `0/1` sin traducir; el `labelencoder` cargado no se usaba. | Se traduce con `labelencoder.inverse_transform` → `SI` / `NO`, con veredicto legible. |
| Sin información de confianza. | Se muestra el margen SVM y un índice de riesgo 0–100, aclarando que **no** es una probabilidad calibrada (el SVC se entrenó con `probability=False`). |

### Rendimiento
- `@st.cache_resource` carga el `.pkl` **una vez por proceso** (antes se leía el archivo en cada interacción).
- Formulario con `st.form`: el modelo se evalúa solo al pulsar *Predecir*, no en cada movimiento de un slider.
- Todo el análisis de sensibilidad es **vectorizado**: las ~35 variantes del gráfico de factores se evalúan en **una sola llamada** a `decision_function`.

### Legibilidad y buenas prácticas
- Separación **lógica / interfaz**: `desercion_core.py` no importa Streamlit; se puede probar, reutilizar en notebooks o usar para predicción por lotes.
- Listas de columnas definidas una sola vez como constantes (`NUM_COLS`, `CAT_COLS`, `CAT_OPTIONS`, `LABELS`) en lugar de repetirlas 4 veces.
- Se eliminó código residual de Colab (`matplotlib`/`numpy` sin uso, `print`, `.head()` sueltos, enlaces de Colab).
- Tipado (`type hints`), `dataclass` inmutables (`Artifacts`, `Prediction`) y *docstrings*.
- Validación de que el modelo existe, con mensaje de error claro.
- Se verificó que el nuevo pipeline reproduce **exactamente** el margen del flujo original para el mismo perfil (−0.80418283).

## 4. Diseño de interfaz (UI/UX)

- **Paleta**: azules oscuros mate (`#0a1424` fondo, `#101e36` superficies, `#1f3358` bordes, `#5b86c6` acento); rojo y verde apagados solo para el estado de riesgo.
- **Fondo**: degradado mate + patrón SVG académico (birrete, libro, lápiz, átomo) embebido en CSS y fijado con **opacidad 0.07**, detrás del contenido y sin capturar clics. No depende de archivos externos.
- **Sombras suaves** en paneles, tarjetas y botón; bordes de 1 px de bajo contraste.
- **Interacción fluida**: la tarjeta de resultado entra con animación; el medidor de riesgo se "llena" al calcularse; las métricas se elevan ligeramente al pasar el cursor; los gráficos hacen transiciones suaves.
- **Jerarquía** (patrón de su imagen de referencia): barra lateral con datos → resultado a la izquierda → análisis a la derecha. Entradas agrupadas por tema (periodo, desempeño, historial, contexto, apoyos).
- **Minimalismo**: se ocultan menú y pie de página de Streamlit; sin iconos ni colores decorativos.

## 5. Visualización adicional (Plotly, interactiva)

Panel con dos pestañas, ambas calculadas con el modelo y el perfil ingresado:

1. **Factores críticos** (gráfico tipo tornado): para cada variable se mueve *solo ella* a sus extremos (o a cada categoría) y se grafica el rango del índice de riesgo. Barras largas = variables que más pueden cambiar el resultado de este estudiante. Líneas de referencia: el perfil actual y la frontera del SVM (50).
2. **Sensibilidad**: curva del índice de riesgo al variar una variable numérica elegida en todo su rango, con el perfil actual marcado y la zona de riesgo sombreada.

`TERM_ID` se excluye de estos análisis porque el periodo no es un factor sobre el que se pueda intervenir.

## 6. Notas y limitaciones

- **Interpretación**: el análisis de sensibilidad es "una variable a la vez"; no captura interacciones entre variables. Describe el comportamiento del modelo, no causalidad.
- **Recall 93 %**: el modelo prioriza detectar deserción (`class_weight="balanced"`), por lo que puede generar falsas alarmas. Se informa en la interfaz.
- **Seguridad**: `pickle` ejecuta código al cargarse; use solo archivos `.pkl` de origen confiable.
- **Streamlit**: escrito para la API de la versión ≥ 1.50 (`width="stretch"`), con alternativa automática (`use_container_width`) para versiones anteriores. Los selectores CSS dependen de la estructura interna de Streamlit; si una actualización cambia algún `data-testid`, el ajuste es puntual en `inject_css()`.
