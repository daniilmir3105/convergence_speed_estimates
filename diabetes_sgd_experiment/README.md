# Diabetes MLP: ML-подбор learning rate

Эксперимент реализует исходную идею ML-пайплайна: зависимость качества
сходимости от learning rate сначала измеряется серией реальных обучений, а затем
аппроксимируется моделью `sklearn.ensemble.GradientBoostingRegressor`. Gauss–Newton-матрица и модели из
`gd` не используются.

## Фиксированные условия

- датасет `sklearn.datasets.load_diabetes`;
- разбиение train/validation/test: 70%/15%/15%;
- стандартизация признаков и целевой переменной только по train;
- MLP `10 -> 16 -> 16 -> 1`, ReLU, линейный выход;
- full-batch `torch.optim.SGD` без momentum;
- постоянный learning rate и MSE;
- одинаковая начальная инициализация сети во всех сравниваемых запусках.

## Логика эксперимента

1. В заданном диапазоне случайно и равномерно по логарифмической шкале
   выбираются learning rate для обучающей выборки мета-модели.
2. Для каждого learning rate MLP заново обучается фиксированное число эпох.
   Целевая метрика — итоговая validation MSE. Test в подборе не участвует.
3. `GradientBoostingRegressor` обучается отображению `log10(learning rate) -> validation MSE`.
   Это один исходный признак — learning rate; логарифм нужен только для удобного
   представления диапазона.
4. Прогноз градиентного бустинга вычисляется на плотной сетке. Центр интервала с минимальным
   прогнозом становится learning rate ML-пайплайна.
5. Независимо выполняются Grid Search на собственной фиксированной сетке и
   Learning Rate Finder с экспоненциально растущим шагом. Для LR Finder берётся
   одна десятая learning rate в точке наиболее быстрого падения сглаженного loss.
6. Три найденных значения сравниваются в финальных запусках из одной и той же
   инициализации. Сохраняются MSE, R² и RMSE в исходной шкале target.

Для градиентного бустинга дополнительно вычисляется 5-fold OOF R². Это проверяет способность
мета-модели интерполировать зависимость, но не используется для выбора шага.

## Запуск

Из корня проекта:

```powershell
.\venv\Scripts\python.exe diabetes_sgd_experiment\experiment.py
```

Быстрый проверочный запуск:

```powershell
.\venv\Scripts\python.exe diabetes_sgd_experiment\experiment.py --surrogate-runs 16 --grid-candidates 17 --selection-epochs 100 --compare-epochs 200
```

Диапазон и объём эксперимента можно менять аргументами `--lr-min`, `--lr-max`,
`--surrogate-runs`, `--grid-candidates`, `--selection-epochs`,
`--lr-finder-steps` и `--compare-epochs`.

## Сохраняемые файлы

- `results/surrogate_training_data.csv` — наблюдения для обучения градиентного бустинга;
- `models/gradient_boosting_lr_surrogate.joblib` — обученная мета-модель;
- `results/surrogate_prediction_curve.csv` — прогноз мета-модели на плотной сетке;
- `results/grid_search.csv` — независимый Grid Search;
- `results/lr_finder.csv` — траектория Learning Rate Finder;
- `results/learning_curves.csv` — финальные train/validation-кривые трёх методов;
- `results/comparison.png` — четыре итоговых графика;
- `results/metrics.json` — конфигурация, выбранные learning rate и метрики;
- `models/surrogate_model.pt`, `models/grid_search_model.pt` и
  `models/lr_finder_model.pt` — финальные веса трёх MLP.

Все процедуры выбора видят только train и validation. Test используется ровно
один раз — для итогового сравнения найденных learning rate.
