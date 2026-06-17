# Taller: Spark MLlib — Clasificación y Ajuste de Hiperparámetros

**Tecnología sugerida:** PySpark 3.x · Python 3  

---

## Ejercicio 1 — Crear una SparkSession y cargar datos

Crea una `SparkSession` y carga el siguiente dataset de clasificación de crédito. El objetivo es predecir si un cliente pagará su préstamo (`default = 1`) o no (`default = 0`).

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

spark = SparkSession.builder \
    .appName("Taller-Clasificacion") \
    .getOrCreate()

schema = StructType([
    StructField("edad",        IntegerType(), True),
    StructField("educacion",   StringType(),  True),
    StructField("empleo",      StringType(),  True),
    StructField("ingreso",     DoubleType(),  True),
    StructField("deuda",       DoubleType(),  True),
    StructField("default",     IntegerType(), True),
])

data = [
    (25, "universitaria", "empleado",      32000.0,  5000.0,  0),
    (40, "secundaria",    "independiente", 18000.0, 12000.0,  1),
    (35, "posgrado",      "empleado",      60000.0,  2000.0,  0),
    (28, "universitaria", "desempleado",   0.0,      8000.0,  1),
    (52, "primaria",      "empleado",      25000.0,  3000.0,  0),
    (30, "universitaria", "empleado",      45000.0,  6000.0,  0),
    (23, "secundaria",    "desempleado",   0.0,     15000.0,  1),
    (45, "posgrado",      "empleado",      80000.0,  1000.0,  0),
    (38, "universitaria", "independiente", 30000.0,  9000.0,  1),
    (60, "secundaria",    "empleado",      22000.0,  4000.0,  0),
    (33, "universitaria", "empleado",      52000.0,  3500.0,  0),
    (47, "secundaria",    "independiente", 15000.0, 11000.0,  1),
    (29, "posgrado",      "empleado",      70000.0,  1500.0,  0),
    (55, "primaria",      "desempleado",   0.0,     20000.0,  1),
    (42, "universitaria", "empleado",      38000.0,  7000.0,  0),
    (26, "secundaria",    "desempleado",   0.0,     13000.0,  1),
    (50, "posgrado",      "empleado",      90000.0,   500.0,  0),
    (37, "universitaria", "independiente", 28000.0,  8500.0,  1),
    (43, "secundaria",    "empleado",      19000.0,  4500.0,  0),
    (31, "universitaria", "empleado",      41000.0,  6500.0,  0),
]

df = spark.createDataFrame(data, schema=schema)
df.show()
df.printSchema()
```

**Preguntas:**
- ¿Cuántas filas y columnas tiene el DataFrame?
- ¿Cuál es el tipo de dato de cada columna?

---

## Ejercicio 2 — Transformar variables categóricas con StringIndexer

Las columnas `educacion` y `empleo` son cadenas de texto. MLlib necesita valores numéricos. Usa `StringIndexer` para convertirlas a índices enteros. Aplica la transformación sobre el DataFrame y muestra el resultado.

```python
from pyspark.ml.feature import StringIndexer

# StringIndexer para 'educacion'
indexer_edu = StringIndexer(inputCol="educacion", outputCol="educacion_idx")

# Tu código: aplicar fit y transform sobre df
# También indexa 'empleo' -> 'empleo_idx'
```

**Preguntas:**
- ¿Qué valor numérico recibió la categoría más frecuente?
- ¿Por qué no podemos pasar strings directamente a un modelo de MLlib?

---

## Ejercicio 3 — Combinar características con VectorAssembler

Después de indexar las variables categóricas, usa `VectorAssembler` para combinar todas las características numéricas en un único vector `features`.

Las columnas a combinar son: `edad`, `educacion_idx`, `empleo_idx`, `ingreso`, `deuda`.

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["edad", "educacion_idx", "empleo_idx", "ingreso", "deuda"],
    outputCol="features"
)

# Tu código: aplicar transform y mostrar columnas 'features' y 'default'
```

**Preguntas:**
- ¿Qué tipo de objeto es la columna `features` resultante?
- ¿Por qué es necesario tener todas las características en un único vector para MLlib?

---

## Ejercicio 4 — Normalizar con StandardScaler

Las magnitudes de `ingreso` (decenas de miles) y `edad` (decenas) son muy diferentes. Aplica `StandardScaler` sobre la columna `features` para normalizar a media 0 y desviación estándar 1. Llama a la columna resultado `features_scaled`.

```python
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    inputCol="features",
    outputCol="features_scaled",
    withMean=True,
    withStd=True
)

# Tu código: fit sobre datos de entrenamiento, transform sobre todo el df
# Muestra las columnas 'features' y 'features_scaled'
```

**Preguntas:**
- ¿Qué diferencia hay entre `withMean=True` y `withStd=True` en el escalado?
- ¿Por qué el `StandardScaler` es un `Estimator` y no un `Transformer`?

---

## Ejercicio 5 — Construir un Pipeline completo

Integra todos los pasos anteriores en un `Pipeline` de MLlib. El pipeline debe tener los siguientes stages en orden:
1. `StringIndexer` para `educacion`
2. `StringIndexer` para `empleo`
3. `VectorAssembler`
4. `StandardScaler`

Divide el dataset en 70 % entrenamiento y 30 % prueba, luego ajusta el pipeline sobre los datos de entrenamiento y transforma ambos conjuntos.

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

indexer_edu = StringIndexer(inputCol="educacion",  outputCol="educacion_idx")
indexer_emp = StringIndexer(inputCol="empleo",     outputCol="empleo_idx")

assembler = VectorAssembler(
    inputCols=["edad", "educacion_idx", "empleo_idx", "ingreso", "deuda"],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features", outputCol="features_scaled",
    withMean=True, withStd=True
)

pipeline = Pipeline(stages=[indexer_edu, indexer_emp, assembler, scaler])

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Tu código: fit pipeline sobre train_df
# Transformar train_df y test_df
# Mostrar columnas label ('default') y 'features_scaled'
```

**Preguntas:**
- ¿Qué ventaja tiene usar un `Pipeline` en lugar de aplicar cada transformación manualmente?
- ¿Por qué el `fit` del pipeline solo debe hacerse sobre los datos de entrenamiento?

---

## Ejercicio 6 — Inspeccionar los stages del PipelineModel

Una vez ajustado el `PipelineModel`, inspecciona sus stages para extraer la información aprendida durante el entrenamiento.

```python
# Después de pipeline_model = pipeline.fit(train_df)

pipeline_model = pipeline.fit(train_df)

for i, stage in enumerate(pipeline_model.stages):
    print(f"Stage {i}: {type(stage).__name__}")

# Extrae el StandardScalerModel y muestra mean y std de cada característica
scaler_model = pipeline_model.stages[-1]
print("Media de cada característica  :", scaler_model.mean)
print("Desv. estándar de cada caract.:", scaler_model.std)
```

**Preguntas:**
- ¿Cuántos stages tiene el `PipelineModel` ajustado?
- ¿Qué información guarda el `StandardScalerModel` que fue aprendida del training set?

---


# Pipelines

---

## Ejercicio 7 — Regresión Logística en Spark (solver L-BFGS)

Entrena una `LogisticRegression` sobre `features_scaled` para predecir `default`. Usa el solver por defecto (`lbfgs`). Imprime la exactitud (accuracy) sobre el conjunto de prueba.

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lr = LogisticRegression(
    featuresCol="features_scaled",
    labelCol="default",
    solver="lbfgs",
    maxIter=100
)

# Tu código: fit sobre train_ready, transform test_ready
# Calcula el área bajo la curva ROC con BinaryClassificationEvaluator
```

**Preguntas:**
- ¿Qué métrica devuelve por defecto `BinaryClassificationEvaluator`?
- ¿Para qué sirve el parámetro `maxIter` en la regresión logística de Spark?

---

## Ejercicio 8 — Comparar solvers: L-BFGS vs SGD

Entrena dos modelos de `LogisticRegression`, uno con `solver="lbfgs"` y otro con `solver="gd"` (gradiente descendente estocástico). Compara el AUC-ROC y el tiempo de entrenamiento de cada uno.

```python
import time
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="default", metricName="areaUnderROC")

for solver in ["lbfgs", "gd"]:
    lr = LogisticRegression(
        featuresCol="features_scaled", labelCol="default",
        solver=solver, maxIter=100
    )
    start = time.time()
    model = lr.fit(train_ready)
    elapsed = time.time() - start

    predictions = model.transform(test_ready)
    auc = evaluator.evaluate(predictions)
    print(f"Solver={solver:6s} | AUC-ROC={auc:.4f} | Tiempo={elapsed:.2f}s")
```

**Preguntas:**
- ¿Cuál solver converge más rápido en este dataset pequeño?
- ¿En qué escenarios de Big Data podría preferirse SGD sobre L-BFGS?

---

## Ejercicio 9 — Regresión Logística multiclase

Crea un nuevo dataset de tres clases (riesgo: bajo=0, medio=1, alto=2) y entrena una `LogisticRegression` en modo `multinomial`. Evalúa con `MulticlassClassificationEvaluator`.

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

data_3class = [
    (25, "universitaria", "empleado",      32000.0,  5000.0,  0),
    (40, "secundaria",    "independiente", 18000.0, 12000.0,  2),
    (35, "posgrado",      "empleado",      60000.0,  2000.0,  0),
    (28, "universitaria", "desempleado",   0.0,      8000.0,  1),
    (52, "primaria",      "empleado",      25000.0,  3000.0,  1),
    (30, "universitaria", "empleado",      45000.0,  6000.0,  0),
    (23, "secundaria",    "desempleado",   0.0,     15000.0,  2),
    (45, "posgrado",      "empleado",      80000.0,  1000.0,  0),
    (38, "universitaria", "independiente", 30000.0,  9000.0,  1),
    (60, "secundaria",    "empleado",      22000.0,  4000.0,  1),
]

# Tu código:
# 1. Crear DataFrame con este dataset (columna 'riesgo' en lugar de 'default')
# 2. Aplicar el pipeline de features (usa labelCol='riesgo')
# 3. Entrenar LogisticRegression con family='multinomial'
# 4. Evaluar con MulticlassClassificationEvaluator (metricName='accuracy')
```

**Preguntas:**
- ¿Qué diferencia hay entre `family='binomial'` y `family='multinomial'`?
- ¿Por qué se usa `MulticlassClassificationEvaluator` y no `BinaryClassificationEvaluator`?

---

## Ejercicio 10 — DecisionTreeClassifier: impurity y maxDepth

Entrena un `DecisionTreeClassifier` sobre el dataset binario original. Experimenta con `impurity='gini'` y `impurity='entropy'`, y con `maxDepth` de 2 y 4. Muestra el AUC-ROC de cada combinación.

```python
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="default", metricName="areaUnderROC")

configs = [
    ("gini",    2),
    ("gini",    4),
    ("entropy", 2),
    ("entropy", 4),
]

for impurity, depth in configs:
    dt = DecisionTreeClassifier(
        featuresCol="features_scaled",
        labelCol="default",
        impurity=impurity,
        maxDepth=depth
    )
    model = dt.fit(train_ready)
    preds = model.transform(test_ready)
    auc   = evaluator.evaluate(preds)
    print(f"impurity={impurity:8s} maxDepth={depth} | AUC-ROC={auc:.4f}")
```

**Preguntas:**
- ¿Qué combinación obtuvo mejor AUC-ROC?
- ¿Qué riesgo existe al usar `maxDepth=10` en un dataset pequeño?

---

## Ejercicio 11 — ParamGridBuilder: construir la grilla de hiperparámetros

Crea una grilla de hiperparámetros para un `DecisionTreeClassifier` usando `ParamGridBuilder`. La grilla debe explorar:
- `maxDepth`: [2, 4, 6]
- `minInstancesPerNode`: [1, 3]
- `impurity`: ['gini', 'entropy']

Imprime la cantidad de combinaciones totales sin entrenar ningún modelo.

```python
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder

dt = DecisionTreeClassifier(featuresCol="features_scaled", labelCol="default")

param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth,              [2, 4, 6]) \
    .addGrid(dt.minInstancesPerNode,   [1, 3]) \
    .addGrid(dt.impurity,              ["gini", "entropy"]) \
    .build()

print(f"Total de combinaciones: {len(param_grid)}")  # Debe ser 3 × 2 × 2 = 12
```

**Preguntas:**
- ¿Cuántos modelos se entrenarán si se usa validación cruzada con 3 folds?
- ¿Por qué se llama "grilla" (grid) de hiperparámetros?

---

## Ejercicio 12 — CrossValidator: ajuste distribuido de hiperparámetros

Envuelve el `DecisionTreeClassifier` con la grilla del ejercicio anterior en un `CrossValidator` con 3 folds. Entrena el `CrossValidator` y reporta el mejor `maxDepth` encontrado y su AUC-ROC en test.

```python
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

dt = DecisionTreeClassifier(featuresCol="features_scaled", labelCol="default")

param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth,            [2, 4, 6]) \
    .addGrid(dt.minInstancesPerNode, [1, 3]) \
    .addGrid(dt.impurity,            ["gini", "entropy"]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="default", metricName="areaUnderROC")

cv = CrossValidator(
    estimator=dt,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

# Tu código: fit cv sobre train_ready
# Obtén el mejor modelo con cv_model.bestModel
# Evalúa en test_ready y reporta:
#   - Mejor maxDepth
#   - Mejor impurity
#   - AUC-ROC en test
```

**Preguntas:**
- ¿Cómo accedes a los hiperparámetros del mejor modelo encontrado?
- ¿Por qué es importante evaluar el mejor modelo sobre el conjunto de prueba y no sobre el de validación cruzada?


