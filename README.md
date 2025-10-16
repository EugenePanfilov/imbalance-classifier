# Imbalance Classifier

## CI (GitHub Actions)

В репозитории настроен CI (`.github/workflows/ci.yml`), который при каждом push/PR:
1. Устанавливает зависимости.
2. Запускает линтеры (`ruff`, `mypy`).
3. Прогоняет тесты (`pytest`).


![ci](https://github.com/EugenePanfilov/imbalance-classifier/actions/workflows/ci.yml/badge.svg)

---

## Описание проекта

Конвейер обучения для **несбалансированной бинарной классификации**:
- генерация/загрузка данных, стратифицированный train/test split;
- препроцессинг признаков (числовые/категориальные, без доступа к `y`);
- перебор нескольких моделей и честные OOF-прогнозы через CV;
- выбор лучшей модели по **PR-AUC** (можно поменять);
- калибровка вероятностей (Platt / isotonic);
- подбор порога по стоимости ошибок (FN/FP);
- финальная оценка на hold-out тесте;
- сохранение артефактов и графиков;
- CLI для обучения и инференса.

Артефакты в `artifacts/`:
- `preprocessor.pkl`, `model.pkl` — откалиброванный пайплайн;
- `metrics_cv.json` — метрики и бутстрап-CI для всех моделей (OOF);
- `metrics_test.json` — метрики на тесте;
- `thresholds.json` — оптимальный порог и `0.5` для сравнения;
- `test.csv` — сохранённый hold-out сплит (для демо `predict`);
- `pr_curve.png`, `calibration_curve.png`, `cost_vs_threshold.png`.

---

## Установка и запуск

Проект использует локальное виртуальное окружение `.venv` (активировать вручную не обязательно — `Makefile` запускает бинарник Python напрямую).

```bash
# 1) установка
make setup

# 2) обучение (использует configs/imbalance.yaml)
make train

# 3) инференс на сохранённом hold-out
make predict   # сохранит artifacts/preds.csv
```

Полезные цели:
```bash
make test      # pytest
make lint      # ruff + mypy
make clean     # очистка кэшей и artifacts
make help      # подсказка по таргетам
```

> Требования: Python ≥ 3.9. Основные зависимости: scikit-learn (≥1.2), numpy, pandas, matplotlib, pyyaml, joblib.

---

## Структура

```
src/mlc/
  config.py         # загрузка YAML, dataclass/TypedDict, валидация
  logging.py        # единая настройка логгера
  data.py           # генерация/загрузка, train/test split, сохранение test.csv
  features.py       # ColumnTransformer с числ./кат. пайплайнами (y-agnostic)
  models.py         # фабрика моделей: logistic / hist_gbdt / rf
  validation.py     # стратегия CV + OOF-прогнозы
  calibration.py    # CalibratedClassifierCV (sigmoid | isotonic)
  metrics.py        # PR-AUC, ROC-AUC, Brier, Recall@k, F1@thr, bootstrap CI
  cost.py           # функция стоимости и поиск оптимального порога
  plots.py          # PR/Calibration/Cost графики (MPL backend Agg)
  persistence.py    # сохранение/загрузка артефактов
  trainer.py        # оркестратор обучения и отчётов
  infer.py          # загрузка артефактов и предсказания
scripts/
  train.py          # CLI: --config configs/imbalance.yaml
  predict.py        # CLI: --config ..., --input ..., --out ...
configs/
  default.yaml
  imbalance.yaml
tests/
  test_features.py
  test_model.py
  test_cost_calib.py
.github/workflows/
  ci.yml
artifacts/          # создаётся на лету (в .gitignore)
Makefile
pyproject.toml / ruff.toml / mypy.ini / pytest.ini
README.md
```

---

## Конфигурация

См. `configs/default.yaml` и `configs/imbalance.yaml`. Основные секции:
- `data` — источник/генерация, размерность, дисбаланс, `test_size`;
- `validation.cv` — `n_splits`, `n_repeats`;
- `models` — список спецификаций (тип + гиперпараметры);
- `calibration` — метод (`sigmoid` | `isotonic`);
- `cost` — стоимость FN/FP;
- `reports` — параметры отчётов (например, `pr_k`);
- `paths` — каталог артефактов.

---

## Как работает пайплайн

1. **Данные** → `train/test split` (стратифицированный) → сохраняем `artifacts/test.csv`.
2. **Фичи** → `ColumnTransformer` (числовые: imputer+scaler; категориальные: imputer+OHE(handle_unknown='ignore')).
3. **CV/OOF** → для каждой модели считаем честные OOF-прогнозы → `metrics_cv.json` (+ бутстрап-CI).
4. **Выбор модели** → по максимальному **PR-AUC** (OOF).
5. **Графики (OOF)** → `pr_curve.png` и `calibration_curve.png` строятся по OOF лучшей модели.
6. **Калибровка** → CalibratedClassifierCV на всём train.
7. **Порог** → подбор по функции стоимости на train-предсказаниях калиброванной модели → `cost_vs_threshold.png`.
8. **Финальная оценка** → метрики на hold-out test → `metrics_test.json`.
9. **Сохранение артефактов** → `preprocessor.pkl`, `model.pkl`, JSON, PNG, `test.csv`.

---

## Инференс

```bash
python scripts/predict.py \
  --config configs/imbalance.yaml \
  --input artifacts/test.csv \
  --out artifacts/preds.csv
```

Скрипт загрузит артефакты (препроцессор, модель, порог), подготовит вероятности `proba` и метки `label = (proba >= threshold_opt)`.

---

## Примечания

- PR-кривая и калибровка в отчётах — **по OOF**, без утечки на train.
- Графики рисуются с backend’ом `Agg` (без `tkinter`), что устраняет предупреждения в тестах/CI.
- При желании можно сменить критерий выбора модели (например, на Brier/ROC-AUC) и/или считать порог по стоимости на OOF-предсказаниях калиброванной модели.

---

## Три команды для старта

```bash
make setup
make train
make predict
```

Готово. Если нужен бейдж CI — добавьте строку из секции «CI» после пуша в GitHub.
