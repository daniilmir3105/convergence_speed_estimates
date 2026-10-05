# Heavy Ball alternative

Эта версия обучена специально для обратной задачи. В отличие от исходной
модели она предсказывает терминальную скорость сходимости кандидата, а не
продолжает уже известную историю траектории.

## Полный запуск

Из корня проекта:

```powershell
.\venv\Scripts\python.exe hb\pipeline.py
```

Конвейер автоматически:

1. создаст датасет терминальных скоростей без досрочной остановки;
2. обучит внутреннюю forward-модель;
3. выполнит совместный поиск `alpha` и `beta`;
4. обучит две финальные модели с признаками только `L` и `l`;
5. сохранит метрики и сравнительный график.

## Быстрый эксперимент

```powershell
.\venv\Scripts\python.exe hb\pipeline.py `
  --forward-groups 300 `
  --forward-candidates 48 `
  --forward-starts 8 `
  --forward-estimators 200 `
  --meta-pairs 1000 `
  --meta-estimators 300 `
  --n-alpha 24 `
  --n-beta 24 `
  --refine-alpha 12 `
  --refine-beta 12
```

## Результаты

- `models/gradient_boosting_alpha.joblib` — sklearn-модель `(L, l) -> alpha`;
- `models/gradient_boosting_beta.joblib` — sklearn-модель `(L, l) -> beta`;
- `results/metrics.json` — все итоговые метрики;
- `results/meta_vs_theory.png` — модель против теории Поляка;
- `data/inverse_train.csv` — найденные обратной задачей параметры;
- `data/meta_test.csv` — независимая тестовая выборка.

## Использование готовых моделей

```powershell
.\venv\Scripts\python.exe hb\predict.py 40 10
```

Команда выводит рекомендуемые `alpha` и `beta`. Если аргументы переданы в
обратном порядке, скрипт автоматически считает большим значением `L`.
