# Calidad de datos en Machine Learning: 6 problemas, 6 experimentos

Serie de notebooks que reproduce, mide y corrige los seis problemas de datos que más rompen modelos en producción, usando un dataset real: la **Base de Matrícula y Secciones 2023** del Ministerio de Educación de la Nación (35.169 escuelas argentinas).

Cada notebook sigue la misma estructura: **reproducir el problema → medir su impacto → diagnosticarlo → corregirlo → checklist**.

| # | Notebook | Problema | Resultado clave |
|---|---|---|---|
| 1 | [Data leakage](Practica_1_data_leakage.ipynb) | Target leakage, contaminación train-test, target encoding | Una feature derivada del target da **accuracy 1,000**; sin ella el modelo honesto llega a **0,614** (F1 macro 0,567). Pipeline + CV estratificada: F1 0,562 ± 0,005 |
| 2 | [Class imbalance](Practica_2_class_imbalance.ipynb) | Clase minoritaria < 10 % | Sin corrección: accuracy **0,93** pero recall de la minoría **0** y MCC 0. Con `class_weight='balanced'`: recall **0,72**, MCC 0,15 |
| 3 | [Missing data](Practica_3_missing_data.ipynb) | MCAR, MAR y MNAR (Little & Rubin) | Con MNAR ninguna imputación alcanza; agregar un **indicador de ausencia** baja el MAE de 10,30 a **10,16** y supera al baseline sin faltantes |
| 4 | [Outliers](Practica_4_outliers.ipynb) | Efecto sobre MSE/RMSE, modelos robustos | Huber y Random Forest reducen el MAE (9,40 y 9,67 contra 10,53 de la regresión lineal); el ratio RMSE/MAE delata la presencia de outliers |
| 5 | [Multicolinealidad](Practica_5_multicolinealidad.ipynb) | VIF, Ridge, Lasso, maldición de la dimensionalidad | 11 de 12 variables con **VIF > 10**. Sin regularizar, el 13 % de los coeficientes cambia de signo entre submuestras; Ridge baja su desvío de 1,06 a **0,14**. Con 100 features de ruido el R² de test cae a 0,35, y Lasso lo recupera a **0,81** descartando todo el ruido |
| 6 | [Data bias](Practica_6_data_bias.ipynb) | Sesgo de muestreo, métricas desagregadas, fairness | Subrepresentar escuelas rurales casi no cambia el AUC global (0,729 → 0,723), pero la brecha de **Equal Opportunity** entre urbanas y rurales sube de 0,16 a **0,21** |

## Qué muestra este repo

- **Pipelines de scikit-learn** (`ColumnTransformer` + `Pipeline`) para que todo preprocesamiento se ajuste solo con train.
- Elección de **métricas correctas** según el problema: F1 macro, MCC, AUC, MAE contra RMSE.
- **Imputación** simple, KNN e iterativa (MICE), con indicadores de ausencia.
- **Detección de outliers** (IQR, z-score, Isolation Forest) y modelos robustos (Huber).
- **Regularización** (Ridge y Lasso) y diagnóstico con VIF.
- **Fairness**: métricas desagregadas por subgrupo.

## Cómo correrlo

```bash
pip install pandas numpy scikit-learn seaborn matplotlib statsmodels
jupyter notebook
```

Los notebooks leen `data/matricula_secciones_2023.csv` (Relevamiento Anual 2023, Ministerio de Educación de la Nación, datos públicos).

## Stack

Python · pandas · NumPy · scikit-learn · statsmodels · Matplotlib · Seaborn

---

**Juan Cruz Montero** · Estudiante de Lic. en Ciencia de Datos (UCA) · [LinkedIn](https://www.linkedin.com/in/juan-cruz-montero-a14474431)
