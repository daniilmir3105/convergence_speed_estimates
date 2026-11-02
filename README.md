# ML-оценка оптимальных параметров алгоритмов оптимизации

Проект исследует, можно ли восстанавливать хорошие гиперпараметры алгоритмов
оптимизации при помощи машинного обучения. Основные задачи — подбор шага
градиентного спуска, совместный подбор шага и momentum для метода тяжёлого
шарика, а также практический подбор learning rate нейронной сети.

Во всех актуальных ML-пайплайнах используется
`sklearn.ensemble.GradientBoostingRegressor`. CatBoost и Gauss–Newton-матрица
не используются.

## Состав проекта

| Папка | Задача | Результат |
|---|---|---|
| [`gd`](gd/) | Обратная задача для Gradient Descent | Модель `(L, l) -> alpha` |
| [`hb`](hb/) | Обратная задача для Heavy Ball | Две модели `(L, l) -> alpha` и `(L, l) -> beta` |
| [`diabetes_sgd_experiment`](diabetes_sgd_experiment/) | Эмпирический подбор learning rate для MLP на Diabetes | Модель `learning rate -> validation MSE` и сравнение с Grid Search/LR Finder |
| [`boston_mlp_experiment`](boston_mlp_experiment/) | Подбор learning rate Adam для MLP на Boston Housing | Аналогичное сравнение surrogate, Grid Search и LR Finder после обработки пропусков и выбросов |

Здесь `L` — максимальное, а `l` — минимальное собственное значение гессиана;
`alpha` — шаг оптимизатора; `beta` — коэффициент momentum.

## Основная идея обратной задачи

Для GD и Heavy Ball используется двухэтапная схема:

```text
(параметры задачи + кандидат alpha/beta)
                    ↓
        моделирование сходимости
                    ↓
   GradientBoostingRegressor (forward-модель)
                    ↓
       минимизация её прогноза
                    ↓
     найденные оптимальные alpha/beta
                    ↓
 финальная модель: только (L, l) → параметры
```

Forward-модель предсказывает асимптотическую скорость сходимости

```text
target_rate = 2 · log10(rho),
```

где `rho` — спектральный радиус итерационного отображения для худшего из
крайних собственных значений `L` и `l`. Затем прогноз минимизируется по
кандидатам `alpha` или `(alpha, beta)`. Для равномерной точности на диапазоне
`L/l` от 1.1 до 20000 forward-модель GD обучается на `kappa * target_rate`, а
Heavy Ball — на `sqrt(kappa) * target_rate`. Положительный множитель одинаков
для всех кандидатов одной задачи, поэтому положение минимума не меняется.
Подробный разбор `rho`, характеристических корней, всех множителей таргета и
его отличия от finite-horizon критерия приведён в
[`hb/README.md`](hb/README.md#как-устроен-таргет-forward-модели).

## Математические ориентиры

Для положительно определённой квадратичной функции

```text
f(x) = 1/2 · xᵀHx,    0 < l ≤ λ(H) ≤ L
```

теоретически оптимальный постоянный шаг Gradient Descent равен

```text
alpha* = 2 / (L + l).
```

Для метода тяжёлого шарика Поляка

```text
x[k+1] = x[k] - alpha · grad(f(x[k])) + beta · (x[k] - x[k-1])
```

теоретические параметры имеют вид

```text
alpha* = 4 / (sqrt(L) + sqrt(l))²,
beta*  = ((sqrt(L) - sqrt(l)) / (sqrt(L) + sqrt(l)))².
```

Эти формулы не передаются финальным моделям как целевые значения напрямую:
они используются как независимый ориентир для оценки решения обратной задачи.

## Установка

Проект протестирован на Windows с Python 3.14.4 и следующими версиями:

- NumPy 2.4.6;
- pandas 3.0.3;
- scikit-learn 1.9.0;
- Matplotlib 3.11.0;
- joblib 1.5.3;
- PyTorch 2.14.0 CPU.

Создание нового окружения:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn matplotlib joblib torch
```

Все последующие команды выполняются из корня проекта.

## Быстрый старт с готовыми моделями

Предсказать шаг Gradient Descent для `L=40`, `l=10`:

```powershell
.\venv\Scripts\python.exe gd\predict.py 40 10
```

Скрипт выводит ML-прогноз, теоретическое значение `2/(L+l)` и абсолютную
ошибку.

Предсказать `alpha` и `beta` метода Heavy Ball:

```powershell
.\venv\Scripts\python.exe hb\predict.py 40 10
```

Скрипт также показывает параметры Поляка и абсолютные ошибки. Если `L` и `l`
переданы в обратном порядке, оба скрипта автоматически их переставляют.

Программное использование:

```python
from gd.predict import predict as predict_gd
from hb.predict import predict as predict_hb

alpha_gd = predict_gd(L=40, l=10)
alpha_hb, beta_hb = predict_hb(L=40, l=10)
```

## Полное воспроизведение экспериментов

Для обучения на новом диапазоне подготовлены PowerShell-лаунчеры:

```powershell
.\train_gd.cmd       # только Gradient Descent
.\train_hb.cmd       # только Heavy Ball
.\train_all.cmd      # оба пайплайна последовательно
```

Быстрая проверка с уменьшенными наборами данных:

```powershell
.\train_all.cmd -Quick
```

Режим `-Quick` сохраняет файлы в те же каталоги и поэтому заменяет полные
модели быстрыми тестовыми. После проверки запустите полный режим повторно.

Другой seed можно передать как `-Seed 123`. Файлы `.cmd` сами обходят запрет
на запуск локальных PowerShell-скриптов. Прямой вызов `.ps1` выглядит так:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_all.ps1
```

### Gradient Descent

```powershell
.\venv\Scripts\python.exe gd\pipeline.py
```

Пайплайн:

1. генерирует квадратичные задачи и кандидаты шага;
2. обучает forward-модель скорости сходимости;
3. решает обратную задачу поиском минимума прогноза;
4. обучает итоговую модель `(L, l) -> alpha`;
5. сохраняет данные, модели, метрики и график.

Основные параметры можно посмотреть командой:

```powershell
.\venv\Scripts\python.exe gd\pipeline.py --help
```

### Heavy Ball

```powershell
.\venv\Scripts\python.exe hb\pipeline.py
```

Здесь обратный поиск выполняется совместно по `alpha` и `beta`, после чего для
них обучаются две отдельные финальные модели. Подробное объяснение постановки
находится в [`hb/inverse_problem_idea.txt`](hb/inverse_problem_idea.txt).

Все параметры запуска:

```powershell
.\venv\Scripts\python.exe hb\pipeline.py --help
```

### Практический эксперимент Diabetes

```powershell
.\venv\Scripts\python.exe diabetes_sgd_experiment\experiment.py
```

В этом эксперименте кривизна и модели из `gd` не используются. Фиксируется MLP
`10 -> 16 -> 16 -> 1` с ReLU, MSE и full-batch SGD без momentum. Для разных
постоянных learning rate сеть запускается из одной и той же инициализации.

Целевая переменная surrogate-модели — validation MSE после фиксированных 300
эпох, а единственный исходный признак — learning rate, представленный как
`log10(learning rate)`. Найденный минимум сравнивается с независимыми Grid
Search и Learning Rate Finder. Test-набор используется только в финальном
сравнении.

Быстрый проверочный запуск:

```powershell
.\venv\Scripts\python.exe diabetes_sgd_experiment\experiment.py `
  --surrogate-runs 16 `
  --grid-candidates 17 `
  --selection-epochs 100 `
  --compare-epochs 200
```

### Практический эксперимент Boston Housing

```powershell
.\train_boston.cmd
```

Эксперимент использует 506 наблюдений Boston Housing и фиксированную sklearn MLP
`13 -> 32 -> 32 -> 1`: два скрытых слоя по 32 нейрона, ReLU, MSE и Adam с
постоянным базовым learning rate. Данные делятся на train/validation/test до
предобработки. Медианная импутация, winsorization по квантилям 1%/99% и
стандартизация обучаются только на train. Исходный датасет не содержит пропусков,
но пайплайн корректно обработает их при повторном использовании.

Локальная копия OpenML уже сохранена, поэтому обычный запуск не требует сети.
Быстрая проверка доступна командой `.\train_boston.cmd -Quick`; она
перезаписывает полные результаты сокращёнными.

## Текущие результаты

Результаты ниже получены с `seed=42` и стандартными параметрами скриптов.

### Синтетические обратные задачи

| Модель | R² относительно теории | Корреляция | MAE |
|---|---:|---:|---:|
| GD: итоговая `alpha` | 0.9953 | 0.9981 | 0.000268 |
| Heavy Ball: итоговая `alpha` | 0.9979 | 0.9992 | 0.000293 |
| Heavy Ball: итоговая `beta` | 0.9901 | 0.9984 | 0.027214 |

Полные метрики: [`gd/results/metrics.json`](gd/results/metrics.json) и
[`hb/results/metrics.json`](hb/results/metrics.json).

### Diabetes

Качество surrogate-модели на обучающих запусках:

```text
train R² = 0.9991
5-fold OOF R² = 0.9706
```

| Метод | Learning rate | Test R² | Test RMSE в исходной шкале |
|---|---:|---:|---:|
| Gradient Boosting surrogate | 0.019589 | 0.4673 | 54.999 |
| Grid Search | 0.017550 | 0.4626 | 55.240 |
| Learning Rate Finder | 0.019290 | 0.4664 | 55.043 |

Все три метода выбрали близкую область learning rate. Полный результат находится
в [`diabetes_sgd_experiment/results/metrics.json`](diabetes_sgd_experiment/results/metrics.json),
а графики — в
[`diabetes_sgd_experiment/results/comparison.png`](diabetes_sgd_experiment/results/comparison.png).

### Boston Housing

Качество surrogate-модели: train `R²=0.9997`, 5-fold OOF `R²=0.9701`.

| Метод | Learning rate | Validation R² | Test R² | Test RMSE в исходной шкале |
|---|---:|---:|---:|---:|
| Gradient Boosting surrogate | 0.008271 | 0.9154 | 0.7744 | 4.2341 |
| Grid Search | 0.000631 | 0.9017 | 0.7830 | 4.1528 |
| Learning Rate Finder | 0.001440 | 0.8987 | 0.8009 | 3.9782 |

Подробные метрики и графики сохранены в
[`boston_mlp_experiment/results/metrics.json`](boston_mlp_experiment/results/metrics.json)
и [`boston_mlp_experiment/results/comparison.png`](boston_mlp_experiment/results/comparison.png).

Время полного запуска на текущей CPU-машине составляет примерно 2 минуты для
`gd`, 3 минуты для `hb`, 40 секунд для Diabetes и 40 секунд для Boston. На другом
оборудовании время может заметно отличаться.

## Структура результатов

```text
train_gd.ps1 / train_gd.cmd             # полное или быстрое обучение GD
train_hb.ps1 / train_hb.cmd             # полное или быстрое обучение HB
train_all.ps1 / train_all.cmd            # последовательный запуск обоих
train_boston.ps1 / train_boston.cmd      # полный или быстрый Boston-эксперимент

gd/
├── pipeline.py                         # обучение всего GD-пайплайна
├── predict.py                          # применение готовой alpha-модели
├── data/                               # forward и inverse выборки
├── models/
│   ├── gradient_boosting_terminal_rate.joblib
│   └── gradient_boosting_alpha.joblib
└── results/                            # JSON-метрики и график

hb/
├── pipeline.py                         # обучение Heavy Ball-пайплайна
├── predict.py                          # применение alpha/beta-моделей
├── inverse_problem_idea.txt            # подробная идея обратной задачи
├── data/
├── models/
│   ├── gradient_boosting_terminal_rate.joblib
│   ├── gradient_boosting_alpha.joblib
│   └── gradient_boosting_beta.joblib
└── results/

diabetes_sgd_experiment/
├── experiment.py
├── models/
│   ├── gradient_boosting_lr_surrogate.joblib
│   ├── surrogate_model.pt
│   ├── grid_search_model.pt
│   └── lr_finder_model.pt
└── results/
    ├── surrogate_training_data.csv
    ├── surrogate_prediction_curve.csv
    ├── grid_search.csv
    ├── lr_finder.csv
    ├── learning_curves.csv
    ├── metrics.json
    └── comparison.png

boston_mlp_experiment/
├── experiment.py                       # полный sklearn MLP/Adam-пайплайн
├── data/boston_openml_531.csv          # локальные 506 наблюдений
├── models/                             # preprocessing, surrogate и три MLP
└── results/                            # выборки поиска, метрики и графики
```

Файлы `.joblib` содержат sklearn-модели, а `.pt` — веса итоговых PyTorch MLP.
Загружать сериализованные модели следует только из доверенного источника и с
совместимой версией библиотек.

## Область применимости

- Финальные модели `gd` и `hb` обучены для `l` от 1 до 50 и числа
  обусловленности `L/l` от 1.1 до 20000. Значения `kappa` генерируются
  равномерно по логарифмической шкале.
- Качество вне этого диапазона не проверялось: деревья решений не обеспечивают
  корректную экстраполяцию.
- Теоретическое сравнение относится к положительно определённым квадратичным
  функциям с постоянным гессианом.
- Высокое качество forward-модели не гарантирует идеальное решение обратной
  задачи: минимум чувствителен к локальным ошибкам аппроксимации.
- Результаты Diabetes относятся к фиксированному разбиению данных, архитектуре,
  инициализации, числу эпох и full-batch SGD. При изменении условий surrogate
  нужно обучить заново.

## Воспроизводимость

По умолчанию все эксперименты используют `seed=42`. Сгенерированные датасеты,
готовые модели и итоговые метрики уже находятся в репозитории. Повторный полный
запуск перезаписывает соответствующие файлы в `data`, `models` и `results`.
