# Generar un notebook .ipynb actualizado (versión A: limpia, sin outputs).
# El notebook resuelve los problemas que encontraste:
# - Compatibilidad OneHotEncoder (sparse_output vs sparse)
# - Detección automática de separador CSV (; or ,)
# - Mapeo y homogeneización de encabezados distintos a un esquema estándar
# - Cálculo de sueldo_bruto / sueldo_neto cuando falten
# - Añade columna 'hospital' a partir del nombre del archivo (acrónimo)
# - Pipeline de preprocesamiento robusto y entrenamientos de los 10 modelos requeridos
#
# El archivo se guardará en /mnt/data/PROYECTO_Inferencia_de_Ingresos_v2.ipynb

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path
nb = new_notebook()
cells = []

# Markdown intro
cells.append(new_markdown_cell("# PROYECTO - Inferencia de Ingresos (v2)\n\n"
"""**Versión A — Notebook limpio (sin outputs)**\n\nEste notebook contiene un pipeline robusto para:\n\n- Cargar y concatenar nóminas CSV con separadores variables.\n- Homogeneizar columnas y mapear distintos encabezados a un esquema estándar.\n- Limpiar valores numéricos (comas, puntos, símbolos) y calcular sueldo_bruto / sueldo_neto cuando sea necesario.\n- Añadir columna `hospital` a partir del acrónimo en el nombre del archivo.\n- Preprocesamiento compatible con distintas versiones de scikit-learn (OneHotEncoder).\n- Entrenamiento de 10 modelos de regresión requeridos por la rúbrica.\n- Guardado del mejor modelo y función para predecir desde un CSV nuevo.\n\n**Instrucciones:** coloca tus CSV en la carpeta `./data/` y ejecuta el notebook.\n"""))

# Code: imports and settings
cells.append(new_code_cell("""# Imports y configuración
import os, glob, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# sklearn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# modelos
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

print('sklearn version:', sklearn.__version__)"""))

# Code: helpers for reading CSVs with unknown separator and cleaning numbers
cells.append(new_code_cell("""# Helpers: detectar separador, limpiar números y mapear columnas

def detect_separator(path, nrows=5):
    \"\"\"Detecta si el CSV usa ';' o ',' mirando las primeras líneas.\"\"\"
    with open(path, 'r', encoding='latin1') as f:
        sample = ''.join([next(f) for _ in range(min(20, nrows))])
    # contar comas y puntos y punto y coma en primera linea
    if sample.count(';') > sample.count(','):
        return ';'
    return ','

def clean_numeric_series(s):
    \"\"\"Limpia una serie de texto que representa números, manejando comas y puntos.\"\"\"
    if s.dtype == object:
        # Reemplazar espacios y símbolos comunes
        s_clean = s.astype(str).str.strip().str.replace('\\s+', '', regex=True).str.replace('\\$', '', regex=True)
        # Reemplazar coma como separador de miles => eliminar
        # Si hay coma y punto, inferir: '16,500.00' -> remove comma
        s_clean = s_clean.str.replace('(?<=\\d),(?=\\d{3}\\b)', '', regex=True)
        # Reemplazar coma decimal por punto si no hay punto decimal
        s_clean = s_clean.str.replace(',', '.', regex=False)
        # Finalmente convertir a numérico
        return pd.to_numeric(s_clean.replace(['', 'nan', 'None', 'NaN'], np.nan), errors='coerce')
    else:
        return pd.to_numeric(s, errors='coerce')

# Schema estándar objetivo
STANDARD_COLUMNS = [
    'hospital', 'nombre_completo', 'nombre', 'apellido',
    'departamento', 'cargo', 'tipo_empleado',
    'sueldo_base', 'completivo', 'otros_ingresos',
    'sueldo_bruto', 'isr', 'seguro_medico', 'seguro_vejez', 'otros_descuentos', 'sueldo_neto',
    'mes', 'año', 'ano', 'anio', 'fecha'
]

# Mapas comunes (lowercase keys)
COLUMN_MAP = {
    # nombres variantes hacia nombre_completo
    'nombre': 'nombre',
    'nombres': 'nombre',
    'nombre_completo': 'nombre_completo',
    'nombres_y_apellidos': 'nombre_completo',
    'apellido': 'apellido',
    'apellidos': 'apellido',
    # sueldos
    'sueldo_base': 'sueldo_base',
    'sueldo base': 'sueldo_base',
    'sueldo_bruto': 'sueldo_bruto',
    'sueldo bruto': 'sueldo_bruto',
    'total_de_sueldo': 'sueldo_bruto',
    'total_ingresos': 'sueldo_bruto',
    'total_ingreso': 'sueldo_bruto',
    'total_de_ingresos': 'sueldo_bruto',
    'sueldo_neto': 'sueldo_neto',
    'sueldo neto': 'sueldo_neto',
    'otros_ingresos': 'otros_ingresos',
    'completivo_a_sueldo': 'completivo',
    'completivo': 'completivo',
    'incentivos': 'otros_ingresos',
    'isr': 'isr',
    'mes': 'mes',
    'año': 'año',
    'aÃ±o': 'año',
    'ano': 'año',
    'año ': 'año',
    'genero': 'genero',
    'sexo': 'genero',
    'departamento': 'departamento',
    'departamento ': 'departamento',
    'cargo que desempeña': 'cargo',
    'funciÃ³n': 'cargo',
    'funcion': 'cargo',
    'posicion': 'cargo',
    'estatus': 'tipo_empleado',
    'tipo_de_empleado': 'tipo_empleado'
}

def standardize_columns(df):
    df = df.copy()
    # lowercase and strip
    df.columns = [c.strip().lower() for c in df.columns]
    new_cols = {}
    for c in df.columns:
        key = c
        if key in COLUMN_MAP:
            new_cols[c] = COLUMN_MAP[key]
        else:
            # try removing accents and punctuation for matching
            key2 = re.sub(r'[^\w]', '', key)
            if key2 in COLUMN_MAP:
                new_cols[c] = COLUMN_MAP[key2]
            else:
                new_cols[c] = key  # keep as is
    df.rename(columns=new_cols, inplace=True)
    return df
"""))

# Code: function to load and normalize a single file, add hospital from filename
cells.append(new_code_cell("""# Función para cargar un CSV y mapear a esquema estándar
ACRONYM_TO_HOSPITAL = {
    'hdpb': 'Hospital Docente Padre Billini',
    'hdssd': 'Hospital Docente San Salvador del Distrito',
    'hduddc': 'Hospital Docente Universitario Dr. Dario Contreras',
    'hgdvc': 'Hospital General Docente Villa Consuelo'
}

def infer_hospital_from_filename(fname):
    base = os.path.basename(fname).lower()
    # buscar acrónimo exacto en el nombre del archivo
    for acr, full in ACRONYM_TO_HOSPITAL.items():
        if acr in base:
            return acr, full
    # fallback: devolver filename prefix
    prefix = re.sub(r'[^a-z0-9]', '', os.path.splitext(os.path.basename(fname))[0].lower())
    return prefix, prefix

def load_and_standardize(path):
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, encoding='latin1', engine='python')
    # Añadir columna de fuente
    df['_source_file'] = os.path.basename(path)
    # normalizar encabezados y mapear
    df = standardize_columns(df)
    # añadir hospital
    acr, full = infer_hospital_from_filename(path)
    df['hospital'] = acr
    df['hospital_full'] = full
    # juntar nombre + apellido si hay columnas separadas
    if 'nombre' in df.columns and 'apellido' in df.columns and 'nombre_completo' not in df.columns:
        df['nombre_completo'] = (df['nombre'].astype(str).str.strip() + ' ' + df['apellido'].astype(str).str.strip()).str.strip()
    # limpiar números comunes
    for col in ['sueldo_bruto', 'sueldo_neto', 'sueldo_base', 'completivo', 'otros_ingresos', 'isr']:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])
    # si existe 'total_de_sueldo' mapeado ya a sueldo_bruto por standardize_columns
    # intentar calcular sueldo_bruto si no existe
    if 'sueldo_bruto' not in df.columns:
        # sumar sueldo_base + completivo + otros_ingresos si existen
        parts = []
        if 'sueldo_base' in df.columns:
            parts.append(df['sueldo_base'].fillna(0))
        if 'completivo' in df.columns:
            parts.append(df['completivo'].fillna(0))
        if 'otros_ingresos' in df.columns:
            parts.append(df['otros_ingresos'].fillna(0))
        if parts:
            df['sueldo_bruto'] = sum(parts)
    # si sueldo_neto no existe pero sí descuentos e impuestos, intentar calcular
    if 'sueldo_neto' not in df.columns and 'sueldo_bruto' in df.columns:
        # intentar restar isr y otros descuentos si existen
        net = df['sueldo_bruto'].copy()
        if 'isr' in df.columns:
            net = net - df['isr'].fillna(0)
        # otros descuentos posibles
        for col in ['seguro_medico', 'seguro_vejez', 'otros_descuentos']:
            if col in df.columns:
                net = net - df[col].fillna(0)
        df['sueldo_neto'] = net
    # agregar columna 'mes' y 'año' si existen variantes como 'mes ' con espacios
    for c in list(df.columns):
        if c.strip() in ['mes', 'año', 'ano', 'anio']:
            df.rename(columns={c: 'mes' if 'mes' in c else 'año'}, inplace=True)
    return df
"""))

# Code: load all files and concat
cells.append(new_code_cell("""# Cargar todos los CSV y concatenar
data_folder = Path('./data')
csv_files = sorted(glob.glob(str(data_folder / '*.csv')))
loaded = []
for f in csv_files:
    try:
        df_temp = load_and_standardize(f)
        loaded.append(df_temp)
    except Exception as e:
        print('Error cargando', f, e)

if not loaded:
    raise RuntimeError('No se cargaron archivos. Por favor coloque CSV en ./data/')

full = pd.concat(loaded, ignore_index=True)
# mostrar columnas resultantes
print('Shape concatenado:', full.shape)
print('Columnas:', list(full.columns))"""))

# Code: basic cleaning and ensuring minimal schema
cells.append(new_code_cell("""# Limpieza adicional: normalizar nombres, meses y años, y columnas numéricas
def normalize_text_columns(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace('\\s+', ' ', regex=True)
    return df

text_cols = ['nombre_completo','departamento','cargo','tipo_empleado','hospital_full','hospital']
full = normalize_text_columns(full, text_cols)

# normalizar meses a capitalized Spanish names si están en texto
if 'mes' in full.columns:
    full['mes'] = full['mes'].astype(str).str.strip().str.capitalize()

# normalizar año
if 'año' in full.columns:
    full['año'] = pd.to_numeric(full['año'], errors='coerce')

# Asegurar columnas numéricas están limpias
for col in ['sueldo_base','completivo','otros_ingresos','sueldo_bruto','sueldo_neto','isr']:
    if col in full.columns:
        full[col] = clean_numeric_series(full[col])

# Agregar columna 'anio' para consistencia
if 'año' in full.columns:
    full['anio'] = full['año']
elif 'ano' in full.columns:
    full['anio'] = pd.to_numeric(full['ano'], errors='coerce')

# Reordenar columnas a un esquema legible
cols_order = ['hospital','hospital_full','_source_file','nombre_completo','nombre','apellido','genero','departamento','cargo','tipo_empleado','sueldo_base','completivo','otros_ingresos','sueldo_bruto','isr','sueldo_neto','mes','anio']
existing = [c for c in cols_order if c in full.columns]
full = full[existing + [c for c in full.columns if c not in existing]]

print('Columnas finales visibles:', existing)"""))

# Code: ensure dataset size and select target
cells.append(new_code_cell("""# Verificar requisitos mínimos del proyecto
print('Filas totales (antes de filtrar nulos):', len(full))
# Elegir target: preferir sueldo_neto, si demasiados nulos usar sueldo_bruto
target = 'sueldo_neto'
if full[target].isnull().mean() > 0.5:
    if 'sueldo_bruto' in full.columns:
        target = 'sueldo_bruto'
print('Etiqueta objetivo seleccionada:', target)

# Asegurar al menos 6 características candidatas
candidate_features = []
# numéricas
for c in ['sueldo_base','completivo','otros_ingresos','sueldo_bruto','anio']:
    if c in full.columns:
        candidate_features.append(c)
# categóricas
for c in ['cargo','departamento','hospital','genero','tipo_empleado','mes']:
    if c in full.columns:
        candidate_features.append(c)
print('Features candidatas (muestra):', candidate_features[:12])"""))

# Code: build preprocessing with compatibility for OneHotEncoder
cells.append(new_code_cell("""# Construir pipeline de preprocesamiento compatible con varias versiones de sklearn
# Detectar si OneHotEncoder usa sparse_output o sparse
ohe_kwargs = {}
if 'sparse_output' in OneHotEncoder.__init__.__code__.co_varnames:
    ohe_kwargs['sparse_output'] = False
else:
    ohe_kwargs['sparse'] = False

numeric_cols = [c for c in ['anio','sueldo_base','completivo','otros_ingresos','sueldo_bruto'] if c in full.columns]
categorical_cols = [c for c in ['cargo','departamento','hospital','genero','tipo_empleado','mes'] if c in full.columns]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', **ohe_kwargs))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], remainder='drop')

print('Numeric cols:', numeric_cols)
print('Categorical cols:', categorical_cols)"""))

# Code: modeling training (train-test split, models, evaluation) - note: will run when executed
cells.append(new_code_cell("""# Entrenamiento de modelos (se ejecuta cuando el notebook se corra)
# Preparar X,y eliminando filas donde target sea nulo
df_model = full.dropna(subset=[target]).copy()
X = df_model[numeric_cols + categorical_cols]
y = df_model[target].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'BayesianRidge': BayesianRidge(),
    'Lasso': Lasso(max_iter=10000),
    'KNN': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'SVR': SVR(),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42)
}

results = {}
import time
for name, model in models.items():
    print('\\nEntrenando', name)
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t1 = time.time()
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = (mse ** 0.5)
    r2 = r2_score(y_test, preds)
    results[name] = {'model': pipe, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'train_time_s': t1-t0}
    print(f'{name} listo. MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f} (train_time {t1-t0:.1f}s)')

# Resumen
res_df = pd.DataFrame([
    {'model':k, **{metric: v for metric,v in results[k].items() if metric!='model'}}
    for k in results
]).set_index('model').sort_values('rmse')

print('\\nResultados:')
print(res_df)"""))

# Code: save best model and prediction function
cells.append(new_code_cell("""# Guardar mejor modelo y función de predicción
best_name = res_df.index[0]
best_model = results[best_name]['model']

models_dir = Path('./models')
models_dir.mkdir(exist_ok=True)
joblib.dump(best_model, models_dir / f'best_model_{best_name}.joblib')

def predict_from_csv(csv_path, model_path=None):
    if model_path is None:
        model = best_model
    else:
        model = joblib.load(model_path)
    df_in = load_and_standardize(csv_path)
    # asegurar columnas
    for c in numeric_cols:
        if c not in df_in.columns:
            df_in[c] = 0
    for c in categorical_cols:
        if c not in df_in.columns:
            df_in[c] = 'desconocido'
    X_new = df_in[numeric_cols + categorical_cols]
    preds = model.predict(X_new)
    df_in['predicted_' + target] = preds
    return df_in

print('Mejor modelo guardado:', best_name)"""))

# Save notebook
out_path = './PROYECTO_Inferencia_de_Ingresos_v2.ipynb'
nb['cells'] = cells
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

out_path

