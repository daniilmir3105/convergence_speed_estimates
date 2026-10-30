# Boston Housing: подбор learning rate для sklearn MLP

Эксперимент повторяет ML-пайплайн из `diabetes_sgd_experiment` на наборе
Boston Housing с 506 наблюдениями и 13 признаками. Данные загружаются из
OpenML (`data_id=531`) и сохраняются локально в `data/boston_openml_531.csv`.

> Boston Housing удалён из встроенных датасетов sklearn из-за этических и
> методологических проблем исходного набора. Здесь он используется только для
> воспроизведения заданного учебного эксперимента.

## Фиксированная модель

- `sklearn.neural_network.MLPRegressor`;
- вход: 13 признаков;
- скрытые слои: `(32, 32)`;
- выход: один линейный нейрон;
- активация скрытых слоёв: ReLU;
- оптимизатор: Adam;
- постоянный базовый learning rate (`learning_rate="constant"`);
- MSE;
- full-batch обучение без перемешивания;
- одинаковый `random_state` и одинаковая инициализация во всех запусках.

## Предобработка

Сначала данные делятся на train/validation/test в пропорции 70/15/15. Только
после этого на train обучаются:

1. медианный `SimpleImputer` для пропусков;
2. границы winsorization 1% и 99% для каждого признака;
3. отдельные `StandardScaler` для признаков и целевой переменной.

Validation и test преобразуются уже зафиксированными train-параметрами, поэтому
утечки данных нет. В исходной версии OpenML пропусков сейчас нет, однако
импутация является частью воспроизводимого пайплайна. На текущем train-разбиении
ограничивается 61 экстремальное значение признаков.

## Подбор learning rate

1. Для случайной log-uniform выборки learning rate MLP обучается фиксированные
   300 эпох из одной и той же инициализации.
2. `GradientBoostingRegressor` обучается отображению
   `log10(learning rate) -> validation MSE`.
3. Минимум его прогноза сравнивается с независимым Grid Search.
4. Learning Rate Finder экспоненциально увеличивает базовый шаг Adam и выбирает
   точку наиболее быстрого падения сглаженного train MSE.
5. Три learning rate сравниваются в финальном обучении на 500 эпох. Test
   используется только на этом этапе.

## Запуск

Из корня проекта:

```powershell
.\train_boston.cmd
```

Либо напрямую:

```powershell
.\venv\Scripts\python.exe boston_mlp_experiment\experiment.py
```

Быстрый проверочный запуск:

```powershell
.\train_boston.cmd -Quick
```

`-Quick` перезаписывает модели и результаты уменьшенной тестовой версией.
После проверки следует снова выполнить полный запуск.

## Результаты и модели

Полный запуск с `seed=42` дал следующие результаты:

| Метод | Learning rate | Validation R² | Test R² | Test RMSE |
|---|---:|---:|---:|---:|
| Gradient Boosting surrogate | 0.008271 | 0.9154 | 0.7744 | 4.2341 |
| Grid Search | 0.000631 | 0.9017 | 0.7830 | 4.1528 |
| Learning Rate Finder | 0.001440 | 0.8987 | 0.8009 | 3.9782 |

Surrogate-модель имеет `R²=0.9997` на обучающих запусках и
`R²=0.9701` в 5-fold out-of-fold проверке. Различие выбранных learning rate не
является ошибкой: validation-функция имеет широкую область хороших значений, а
финальные метрики считаются после 500 эпох. Test не участвует ни в одном из трёх
способов подбора.

- `data/boston_openml_531.csv` — локальная копия 506 наблюдений;
- `models/preprocessing.joblib` — imputer, границы выбросов и scaler;
- `models/gradient_boosting_lr_surrogate.joblib` — surrogate-модель;
- `models/surrogate_mlp.joblib` — MLP с learning rate от surrogate;
- `models/grid_search_mlp.joblib` — MLP с learning rate от Grid Search;
- `models/lr_finder_mlp.joblib` — MLP с learning rate от LR Finder;
- `results/surrogate_training_data.csv` — обучающие данные surrogate;
- `results/grid_search.csv` и `results/lr_finder.csv` — независимые поиски;
- `results/learning_curves.csv` — финальные кривые обучения;
- `results/comparison.png` — итоговые графики;
- `results/metrics.json` — конфигурация и все метрики.
