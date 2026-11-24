# Расширенный эксперимент гиподифференциального спуска

Это отдельная версия эксперимента из `hypodifferential_rate`. Исходная папка,
её датасет, модели и результаты не изменяются.

Главные изменения:

- датасет увеличен с 1000 до 5000 функций;
- комбинации размерности, количества компонент и логарифмических диапазонов
  `kappa` покрываются равномерно;
- функции с двумя и тремя компонентами в верхних 30% диапазона `kappa`
  получают двойной вес при генерации;
- добавлены спектральные, геометрические и первошаговые признаки;
- `log_l` и `log_L` исключены;
- необходимость `sqrt_kappa` проверяется на validation;
- основной регрессор — `ExtraTreesRegressor`.

## Неизменный таргет

Таргет полностью совпадает с исходным экспериментом:

\[
J_{300}=\frac{1}{16\cdot300}
\sum_{i=1}^{16}
\log_{10}\left[
\max\left(
\frac{f(x_{300}^{(i)})}{f(x_0^{(i)})},
10^{-300}
\right)
\right].
\]

Используются ровно 16 фиксированных начальных точек и бюджет 300 итераций.
В коде эти два значения зафиксированы проверками аргументов, чтобы случайно не
изменить определение таргета.

## Сбалансированная генерация

По умолчанию используются:

- размерности от 2 до 6;
- от 2 до 6 квадратичных компонент;
- 10 равномерных интервалов по `log(kappa)`;
- диапазон `1.1 <= kappa <= 20000`.

Сначала строятся все ячейки
`dimension × n_pieces × log-kappa-bin`. Число функций распределяется между
ними равномерно. Ячейки с `n_pieces` 2 или 3 в трёх верхних интервалах `kappa`
получают вес 2.0. Поэтому сложные режимы представлены чаще, но остальные
комбинации не исчезают.

Файл `data/design_counts.csv` после генерации показывает фактическое число
объектов в каждой ячейке.

## Признаки без подглядывания в J_300

Модель не получает значения из полной 300-шаговой траектории. Признаки
рассчитываются по матрицам задачи, 16 исходным точкам и одному первому шагу:

- квантили нормированных собственных значений;
- статистики чисел обусловленности отдельных матриц;
- относительные числа обусловленности пар матриц;
- расстояния между матрицами и нормы коммутаторов;
- различия главных собственных направлений;
- активные компоненты именно на 16 точках из определения `J_300`;
- зазоры между первой и второй активной компонентой;
- нормы начального гиподифференциального направления;
- длина первого точного линейного поиска;
- относительное уменьшение функции на первом шаге;
- длина первого перемещения и доля смен активной компоненты.

Первый шаг не использует `x_300` и не раскрывает таргет. Это дешёвая
характеристика начального поведения алгоритма.

`log_l` и `log_L` отсутствуют, поскольку все матрицы перед спуском нормируются
на `L`, а точный линейный поиск инвариантен к общему положительному масштабу.

В датасете сохраняются `log_kappa` и диагностический `sqrt_kappa`. Во время
обучения сравниваются два набора:

1. только `log_kappa`;
2. `log_kappa` вместе с `sqrt_kappa`.

Выбор выполняется только по validation MSE. В итоговую модель попадёт набор,
который действительно оказался лучше.

## Разбиение и обучение

Данные делятся на 60% train, 20% validation и 20% test. Стратификация
выполняется одновременно по размерности, числу компонент и интервалу
`log(kappa)`.

Конфигурации Extra Trees и два варианта признаков сравниваются на validation.
После выбора модель обучается на train+validation. Итоговые `R²`, MAE, MSE и
RMSE вычисляются только на test.

## Установка

Из корня проекта:

```powershell
.\venv\Scripts\python.exe -m pip install -r hypodifferential_rate_enhanced\requirements.txt
```

## Запуск

Сначала создайте новый датасет из 5000 функций:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.generate_dataset
```

Генерация содержит 5000 × 16 запусков гиподифференциального спуска и поэтому
может выполняться заметно дольше исходного эксперимента. По умолчанию
используются все доступные ядра. Ограничить их число можно так:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.generate_dataset `
  --n-jobs 4
```

После завершения генерации обучите модель:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.train
```

Или с явным ограничением числа процессов:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.train --n-jobs 4
```

## Альтернативное обучение ЭОПКГ

На том же датасете можно отдельно обучить `EIKGPolynomialRegressor`. Для
каждого из двух вариантов признаков проверяются степени полинома от 1 до 10.
Степень и набор признаков одновременно выбираются по минимальному validation
MSE, после чего итоговая модель обучается на train+validation. Test используется
только для окончательной оценки.

Запуск:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.train_eikg
```

Собственный диапазон степеней и Ridge-регуляризацию можно задать так:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.train_eikg `
  --degrees 1 2 3 4 5 6 `
  --alpha-ridge 1e-5
```

Модель и результаты ЭОПКГ сохраняются отдельно:

```text
models_eikg/eikg_enhanced_J300.joblib

results_eikg/metrics.json
results_eikg/model_selection.csv
results_eikg/split_assignments.csv
results_eikg/test_predictions.csv
results_eikg/model_selection.png
results_eikg/test_predictions.png
```

## Прогноз

Пример для первой функции из созданного набора:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.predict `
  hypodifferential_rate_enhanced\data\instances.npz --index 0
```

## Результаты

После генерации появятся:

```text
data/dataset.csv
data/instances.npz
data/sample_trajectories.csv
data/design_counts.csv
data/dataset_manifest.json
```

После обучения появятся:

```text
models/extra_trees_enhanced_J300.joblib

results/metrics.json
results/model_selection.csv
results/split_assignments.csv
results/test_predictions.csv
results/model_selection.png
results/test_predictions.png
results/feature_importance.csv
results/feature_importance.png
```

Главные итоговые поля находятся в `results/metrics.json`:

```text
test_metrics.J300.r2
test_metrics.J300.mae
test_metrics.rho.r2
test_metrics.rho.mae
model_selection.selected_feature_set
model_selection.sqrt_kappa_selected
```

## Важное ограничение

Новый датасет специально чаще показывает модели трудные режимы. Поэтому его
метрики нельзя напрямую сравнивать с метриками старого теста как два результата
на одной и той же выборке. Корректный вывод делается по новому зафиксированному
test-разбиению и, при необходимости, отдельной проверке обеих моделей на общем
внешнем наборе функций.
