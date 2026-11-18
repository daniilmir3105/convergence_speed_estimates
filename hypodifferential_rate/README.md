# Прогноз скорости гиподифференциального спуска

Эксперимент обучает одну модель прогнозировать finite-horizon скорость
гиподифференциального спуска на максимумах выпуклых квадратичных функций.
Обратной задачи и подбора постоянного шага здесь нет: на каждой итерации шаг
определяется точным одномерным поиском самого алгоритма.

## Класс задач

Генерируются функции

\[
f(x)=\max_{1\le j\le m}q_j(x),
\qquad
q_j(x)=\frac12x^\mathsf{T}A_jx,
\]

где все матрицы `A_j` симметричны и положительно определены. Поэтому минимум
известен заранее:

\[
x^*=0,
\qquad
f^*=0.
\]

Функция выпукла, но в точках смены активной квадратичной компоненты может быть
негладкой. Число обусловленности генерируется в диапазоне

\[
1.1\le\kappa=\frac{L}{l}\le20000.
\]

## Реализация метода

Для максимума гладких функций используется нормализованный гиподифференциал

\[
C(x)=\operatorname{co}
\left\{
\left(q_j(x)-f(x),\nabla q_j(x)\right):1\le j\le m
\right\}.
\]

Такая конструкция гиподифференциала для max-функции соответствует правилам
кодифференциального исчисления: [источник с формулой](https://dha.spb.ru/PDF/abstractDolgopolik.pdf).
Современное изложение гиподифференциалов и методов спуска опубликовано в
[Journal of Convex Analysis](https://journalofconvexanalysis.com/articles/jca32066/).

На итерации вычисляется

\[
[a_k,g_k]=-\operatorname{proj}_{C(x_k)}(0),
\]

после чего выполняется

\[
\alpha_k=\arg\min_{\alpha\ge0}f(x_k+\alpha g_k),
\qquad
x_{k+1}=x_k+\alpha_kg_k.
\]

Проекция нуля на выпуклую оболочку вычисляется методом минимальной нормы
Gilbert/Frank–Wolfe с контролируемой точностью. Веса предыдущей итерации
используются как тёплый старт.

Одномерный поиск реализован специально для максимума квадратичных функций.
Вдоль направления каждая компонента является одномерной параболой. Минимум
верхней огибающей находится среди:

- `alpha=0`;
- вершин парабол;
- точек пересечения пар парабол.

Поэтому не требуется многократно запускать общий численный оптимизатор.

## Таргет

Для каждой функции используются 16 фиксированных начальных направлений и 300
итераций. Таргет равен

\[
\boxed{
J_{300}=\frac{1}{16\cdot300}
\sum_{i=1}^{16}
\log_{10}\left[
\max\left(
\frac{f(x_{300}^{(i)})}{f(x_0^{(i)})},
10^{-300}
\right)
\right]
}.
\]

Так как `f*=0`, отношение уже является относительным разрывом до оптимума.
Из таргета вычисляется эмпирический средний коэффициент сжатия за итерацию:

\[
\rho=10^{J_{300}}.
\]

Чем отрицательнее `J_300` и чем меньше `rho`, тем быстрее сходимость.

Если условие остановки `||[a_k,g_k]|| <= epsilon` срабатывает раньше 300-го
шага, последняя найденная точка считается неизменной до конца заданного
бюджета. Поэтому в формуле `x_300` используется значение функции в этой
последней точке. Значение `epsilon` сохраняется в манифесте датасета и является
частью численной постановки эксперимента.

Это finite-horizon показатель. Для дополнительной проверки ожидаемой линейной
сходимости сохраняются наклон и `R²` линейной регрессии хвоста графика
`log10(f(x_k)/f(x_0))`.

## Признаки модели

Модель не получает всю траекторию. По матрицам задачи вычисляются компактные
признаки:

- размерность и число квадратичных компонент;
- `l`, `L`, `kappa` и их преобразования;
- статистики следов и норм матриц;
- попарные расстояния между нормированными матрицами;
- нормы коммутаторов, характеризующие различие собственных направлений;
- число фактически активных компонент на выборке направлений;
- энтропия активности, доля доминирующей компоненты и средний зазор между
  двумя крупнейшими квадратичными значениями.

Таким образом учитывается не только обусловленность, но и геометрия
негладкости.

## ML-пайплайн

```text
матрицы A_j
     ↓
точный запуск метода на 16 стартах и 300 итерациях
     ↓
J_300 и rho
     ↓
агрегированные спектральные и геометрические признаки
     ↓
GradientBoostingRegressor: признаки -> J_300
     ↓
rho_pred = 10 ** J_300_pred
```

Данные разделяются на 60% train, 20% validation и 20% test со стратификацией
по `log(kappa)`. Конфигурация градиентного бустинга выбирается по минимальному
validation MSE. После выбора модель переобучается на train+validation, а все
итоговые метрики вычисляются только на test.

## Установка

Из корня проекта:

```powershell
.\venv\Scripts\python.exe -m pip install -r hypodifferential_rate\requirements.txt
```

## Полный запуск

Сначала создаётся датасет:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.generate_dataset
```

По умолчанию генерируется 1000 функций, для каждой выполняются 16 запусков по
300 итераций. Вычисления распараллеливаются по доступным ядрам CPU.

Затем обучается модель:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.train
```

## Альтернативная модель ЭОПКГ

В отдельном скрипте `train_eikg.py` тот же `dataset.csv` и то же разбиение
60/20/20 используются для обучения `EIKGPolynomialRegressor`. Устанавливаемый
пакет называется `eikgpolynomial`, а Python-импорт выполняется из `eikg`.

Степени от 1 до 10 по очереди обучаются только на train. Степень выбирается по
минимальному MSE на validation, после чего итоговая ЭОПКГ-модель переобучается
на train+validation. Метрики `J_300` и `rho` вычисляются только на test.

Запуск:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.train_eikg
```

Свой диапазон степеней и Ridge-регуляризацию можно задать так:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.train_eikg `
  --degrees 1 2 3 4 5 6 `
  --alpha-ridge 1e-5
```

Модель и результаты сохраняются отдельно от градиентного бустинга:

```text
models_eikg/eikg_J300.joblib

results_eikg/metrics.json
results_eikg/degree_selection.csv
results_eikg/split_assignments.csv
results_eikg/test_predictions.csv
results_eikg/degree_selection.png
results_eikg/test_predictions.png
```

## Альтернативный эксперимент Extra Trees

Скрипт `train_extratrees.py` обучает `ExtraTreesRegressor` на том же
`dataset.csv`, с теми же признаками и разбиением 60/20/20, что и основной
Gradient Boosting. Таргет `J_300` и способ вычисления `rho` не изменяются.

Кандидаты обучаются только на train и сравниваются по MSE на validation.
После выбора конфигурации итоговая модель обучается на train+validation,
а метрики `J_300` и `rho` рассчитываются только на test.

Запуск из корня проекта:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.train_extratrees
```

Число используемых процессов можно задать отдельно:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.train_extratrees --n-jobs 4
```

Модель и результаты сохраняются отдельно от остальных экспериментов:

```text
models_extratrees/extra_trees_J300.joblib

results_extratrees/metrics.json
results_extratrees/model_selection.csv
results_extratrees/split_assignments.csv
results_extratrees/test_predictions.csv
results_extratrees/model_selection.png
results_extratrees/test_predictions.png
results_extratrees/feature_importance.csv
results_extratrees/feature_importance.png
```

## Быстрая техническая проверка

Эта команда создаёт слишком маленький датасет для научных выводов, но позволяет
проверить окружение и работоспособность кода:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.generate_dataset `
  --instances 30 `
  --starts 2 `
  --dimension-max 4 `
  --pieces-max 4 `
  --trajectory-instances 2 `
  --projection-max-iterations 100 `
  --n-jobs 1

.\venv\Scripts\python.exe -m hypodifferential_rate.train
```

После технической проверки необходимо заново выполнить полный запуск, потому
что генератор перезаписывает датасет.

## Использование модели

После обучения можно взять, например, задачу с индексом `0` из сохранённого
набора:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.predict `
  hypodifferential_rate\data\instances.npz --index 0
```

Для внешней задачи сохраните массив матриц формы `(m,n,n)` в `.npy`:

```python
import numpy as np

np.save("my_quadratics.npy", matrices)
```

Затем выполните:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.predict my_quadratics.npy
```

Скрипт вернёт прогнозы `J300`, `rho` и рассчитанные признаки задачи.

## Результаты

```text
data/dataset.csv
data/dataset_with_splits.csv
data/instances.npz
data/sample_trajectories.csv
data/dataset_manifest.json

models/gradient_boosting_J300.joblib

results/metrics.json
results/model_selection.csv
results/test_predictions.csv
results/test_predictions.png
results/feature_importance.csv
results/feature_importance.png
results/sample_convergence_curves.png
```

Главные тестовые метрики:

```text
test_metrics.J300.r2
test_metrics.J300.mae
test_metrics.rho.r2
test_metrics.rho.mae
```

## Ограничения

- Эксперимент относится к однородным максимумам положительно определённых
  квадратичных форм с общим минимумом в нуле.
- Проекция вычисляется с конечной численной точностью, параметры которой
  записываются в манифест датасета.
- Агрегированные признаки могут не различать некоторые разные наборы матриц.
- Высокий `R²` модели не является доказательством линейной сходимости. Для
  этого отдельно анализируются сохранённые траектории и линейность их хвостов.
- Модель оценивает скорость и не подбирает параметры метода, поскольку
  `alpha_k` определяется точным поиском внутри каждой итерации.
