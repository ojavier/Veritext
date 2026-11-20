# Veritext

Detector de texto posiblemente generado por IA, en español. Analiza un texto por
párrafo y emite **alertas para revisión humana**. No entrega un veredicto de autoría.

## Cómo funciona

```
texto → segmentar → limpiar → detectar idioma
      → [similitud | matching | zero-shot]   (por segmento)
      → fusión (regresión logística) → nivel bajo / incierto / alto
      → alerta con evidencia → revisión humana
```

| Señal | Qué mide | Módulo |
|---|---|---|
| Similitud | Coseno de embeddings multilingües contra texto humano de referencia | `signals/similarity.py` |
| Matching | Coincidencia de n-gramas/BM25 contra generaciones de LLMs conocidas | `signals/matching.py` |
| Zero-shot | Binoculars: perplexidad / cross-perplexity de Qwen2.5-0.5B base e instruct | `src/pipeline.py` |

La fusión (`fusion/combiner.py`) combina las tres señales y la calibración
(`fusion/calibration.py`) fija el umbral "alto" a una tasa de falsos positivos
objetivo (1 %) y define una zona "incierta" hasta 5 %.

## Instalación

```bash
pip install -e ".[models,similarity,fusion,app,dev]"
```

Para GPU, instala antes `torch` desde pytorch.org.

## Uso

```bash
# 1. Índice de referencia para la señal de similitud
python scripts/build_index.py

# 2. Analizar un texto (json | text | report | html)
python -m src.pipeline texto.txt --format html > alertas.html

# 3. Interfaz web
streamlit run app/streamlit_app.py
```

Entrenar y evaluar la fusión:

```bash
python scripts/train_fusion.py --calculate --zeroshot --device cuda \
    --out data/.cache/fusion.json --calibration-out data/.cache/fusion_calibration.json
python scripts/evaluate.py --calculate --zeroshot --device cuda \
    --holdout-generator qwen3-1.7b --attack typos --attack paraphrase
```

Si no existe `data/.cache/fusion.json`, el pipeline usa solo la señal zero-shot
con umbrales por defecto.

## Datos

- `data/reference/` y `data/eval/`: texto humano en español (Wikipedia y reseñas).
- `data/generated/`: generaciones de tres LLMs (Claude, Qwen2.5-1.5B-Instruct, Qwen3-1.7B).
- Los splits (`evaluation/splits.py`) separan entrenamiento, calibración y prueba por
  dominio y generador. Un generador retenido tampoco aparece en el índice de matching.
- Cada registro se excluye a sí mismo de los índices al calcular señales.

## Evaluación

`scripts/evaluate.py` reporta AUROC y TPR @ FPR 1 % por dominio, con ablaciones
(cada señal sola, promedio, máximo y regresión logística) y ataques
(`typos`, `synonyms`, `paraphrase`) aplicados a los textos de IA del conjunto de prueba.

### Resultados

Prueba con el generador **Qwen3-1.7B retenido** (no visto en entrenamiento ni en el
índice de matching), 956 segmentos de prueba (235 humanos, 721 de IA), objetivo FPR 1 %.
Zero-shot calculado en GPU. Cada celda es AUROC / TPR@FPR 1 %.

| Señal / combinación | Limpio | Typos | Paráfrasis (reglas) |
|---|---|---|---|
| Similitud | 0.475 / 0.003 | 0.534 / 0.001 | 0.482 / 0.001 |
| Matching | 0.901 / 0.209 | 0.831 / 0.075 | 0.894 / 0.187 |
| Zero-shot | 0.926 / 0.530 | **0.147 / 0.000** | 0.911 / 0.406 |
| Promedio | 0.915 / 0.530 | 0.304 / 0.000 | 0.901 / 0.451 |
| Máximo | 0.851 / 0.530 | 0.494 / 0.000 | 0.822 / 0.406 |
| Regresión logística | **0.935** / 0.437 | **0.273 / 0.000** | 0.921 / 0.302 |

Lectura:

- La fusión logística tiene el mejor AUROC en texto limpio (0.935), pero con FPR 1 %
  detecta menos que zero-shot solo (0.437 frente a 0.530).
- La similitud contra texto humano no separa humano de IA (AUROC ≈ azar).
- **Los errores tipográficos rompen el sistema:** el AUROC cae por debajo de 0.5, es decir,
  el texto de IA con typos se puntúa como *más humano* que el humano. Hipótesis (no verificada): los typos
  elevan la perplexidad y el zero-shot lo interpreta como escritura humana. Es la
  principal debilidad conocida.
- La paráfrasis por reglas casi no degrada los resultados, pero es un ataque débil.
- Una sola partición y una sola semilla: no hay intervalos de confianza.

Los resultados por dominio están en el JSON que genera `scripts/evaluate.py --out`.


## Limitaciones

- **El desempeño en español no está validado por la literatura.** Esta evaluación es
  una primera aproximación con un dataset propio y pequeño.
- La paráfrasis de `evaluation/attacks.py` es por reglas; un paráfrasis con modelo
  sería un ataque más fuerte.
- Los detectores zero-shot suelen dar falsos positivos con hablantes no nativos y
  con escritura muy formal.
- Las generaciones provienen de pocos modelos; otros generadores pueden comportarse distinto.
- El resultado es una alerta para revisión humana, nunca una afirmación de autoría.

## Pruebas

```bash
pytest
```
