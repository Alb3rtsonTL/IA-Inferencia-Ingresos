# PROYECTO - Inferencia de Ingresos

En este notebook implementamo el pipeline completo: carga, limpieza, EDA, ingeniería, entrenamiento de modelos de regresión, evaluación, guardado de modelos y funciones de inferencia.

---
#### Flujo de Trabajo

- **Fase 1:** Carga y concatenación de datos  
- **Fase 2:** Limpieza, estandarización y preprocesamiento inicial  
- **Fase 3:** Análisis Exploratorio de Datos (EDA) con estadísticas  
- **Fase 4:** Preparación del dataset para entrenamiento  
- **Fase 5:** Entrenamiento de los *10 modelos de regresión*  
- **Fase 6:** Selección del mejor modelo según desempeño (**R²**)  
- **Fase 7:** Evaluación y análisis de errores  
- **Fase 8:** Funciones de predicción para datos nuevos  
---
#### 📁 Carpetas
`./data/:` Es la carpeta donde se encuentran los archivos CSV con los datos de nómina.

`./models/:` Carpeta donde se guardan los modelos entrenados en formato joblib.