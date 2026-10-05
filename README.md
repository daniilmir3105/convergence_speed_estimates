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

Forward-модель предсказывает терминальную скорость сходимости

```text
mean(log10(f_K / f_0)) / K,
```

усреднённую по фиксированным начальным точкам. Затем её прогноз минимизируется
по кандидатам `alpha` или `(alpha, beta)`. Полученные решения становятся
целевыми значениями для финальных моделей, принимающих только два признака:
`L` и `l`.

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

## Текущие результаты

Результаты ниже получены с `seed=42` и стандартными параметрами скриптов.

### Синтетические обратные задачи

| Модель | R² относительно теории | Корреляция | MAE |
|---|---:|---:|---:|
| GD: итоговая `alpha` | 0.9962 | 0.9981 | 0.000558 |
| Heavy Ball: итоговая `alpha` | 0.9873 | 0.9968 | 0.001129 |
| Heavy Ball: итоговая `beta` | 0.8968 | 0.9937 | 0.033269 |

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

Время полного запуска на текущей CPU-машине составляет примерно 40 секунд для
`gd`, 90 секунд для `hb` и 40 секунд для Diabetes. На другом оборудовании время
может заметно отличаться.

## Структура результатов

```text
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
```

Файлы `.joblib` содержат sklearn-модели, а `.pt` — веса итоговых PyTorch MLP.
Загружать сериализованные модели следует только из доверенного источника и с
совместимой версией библиотек.

## Область применимости

- Финальные модели `gd` и `hb` обучены для `l` от 1 до 50 и числа
  обусловленности `L/l` от 1.1 до 20.
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