# PROYECTO - Inferencia de Ingresos

Este notebook implementa el pipeline completo requerido por la rúbrica:
- Carga y concatenación de nóminas públicas (CSV)
- Limpieza y preprocesamiento
- EDA (estadísticas y visualizaciones)
- Entrenamiento de 10 modelos de regresión exigidos
- Evaluación y comparación de métricas
- Función para predecir desde un nuevo archivo CSV

**Instrucciones:** Los archivosCSV estan en la `./data/csv/`. 

Las nomínas son de los siguientes hospitales:
| Acrónimo   | Hospital                                           |URL
| ---------- | -------------------------------------------------- |-----
| **HDPB**   | Hospital Docente Padre Billini                     |https://datos.gob.do/dataset/nomina-de-empleados-hdpb-2018
| **HDSSD**  | Hospital Docente San Salvador del Distrito         |https://datos.gob.do/dataset/hospital-docente-semma
| **HDUDDC** | Hospital Docente Universitario Dr. Darío Contreras |https://datos.gob.do/dataset/h-d-c
| **HGDVC**  | Hospital General Docente de Villa Consuelo         |https://datos.gob.do/dataset/nomina_de_empleados

**Estos son los formatos de las Cabeceras de los CSV y su primer registro**:

HDPB-Nomina-2024.csv
"NOMBRE","APELLIDO","DEPARTAMENTO","CARGO QUE DESEMPEÑA","SUELDO BASE","COMPLETIVO A SUELDO","TOTAL DE SUELDO","TIPO DE EMPLEADO","MES","AÑO"
"YADENYS DEL CARMEN"," TORIBIO","DIRECCION GENERAL","ASISTENTE EJECUTIVA","20000","0","20000","CONTRATADO INTERNO","ENERO","2019"

HDSSD-Nomina-2025.csv
"Nombre","Genero","Departamento","Posicion","Estatus","Sueldo Bruto","Otros Ingresos","Total Ingresos","ISR","Seguro Medico","Seguro Vejez","Otros Descuentos","Sueldo Neto","Mes ","Año"
"JOSE MANUEL TEJADA GERMAN","M","DIRECCION DEL HOSPITAL","DIRECTOR GENERAL","ACTIVO","170000","10000","180000","31071.19","5168","4879","62245.8","76636.01","Julio","2025"

HDUDDC-Nomina-2025.csv
"Nombres","Departamento","Funcion","Estatus","Sueldo","Mes","Año"
"Luis RafaelOlivo Payano","Finanzas","Gerente Financiero","Contratado","40000","Abril","2021"

HGDVC-Nomina-2025.csv
"Nombre","Apellido","Departamento","Función","Estatus","Sueldo Bruto","Mes","Año"
"HENRRY ARCADIO","PERALTA RODRIGUEZ","DIRECCION GENERAL","MENSAJERO INTERNO","SIMPLIFICADO","16,500.00","enero","2022"