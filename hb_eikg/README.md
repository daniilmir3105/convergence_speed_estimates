# Heavy Ball с ЭОПКГ

Независимый вариант finite-horizon эксперимента `hb`, в котором первая
модель и две итоговые модели заменены на `EIKGPolynomialRegressor` —
элементарный образ полинома Колмогорова–Габора (ЭОПКГ).

Численная постановка, генерация данных, oracle и exact-refinement совпадают с
основным экспериментом `hb`. Модели, разбиения и выходные файлы находятся в
отдельной папке `hb_eikg`.

Библиотека: [eikgpolynomial на PyPI](https://pypi.org/project/eikgpolynomial/).
Устанавливаемый пакет называется `eikgpolynomial`, а импорт выглядит так:

```python
from eikg import EIKGPolynomialRegressor
```

## Постановка

Диапазон числа обусловленности:

\[
1.1\leq\kappa=\frac{L}{l}\leq20000.
\]

Критерий сходимости:

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

Первая ЭОПКГ-модель прогнозирует масштабированный таргет:

\[
y_{\mathrm{first}}=\sqrt{\kappa}\,J_{300}.
\]

После ML-ранжирования и точного уточнения формируются finite-horizon метки
`q_found` и `beta_found`, где

\[
q=\frac{\alpha L}{2(1+\beta)}.
\]

Вторая и третья ЭОПКГ-модели обучаются отображениям

```text
(L, l) -> q
(L, l) -> beta
```

После их предсказаний восстанавливается обычный шаг:

\[
\alpha=\frac{2q(1+\beta)}{L}.
\]

Сохранённая публичная модель принимает `[L,l]` и возвращает две колонки
`[alpha,beta]`.

### Метки не получают точку Поляка как готового кандидата

В отличие от исходного `hb.pipeline`, обратный поиск в этой папке не добавляет
теоретические параметры Поляка в набор кандидатов. Метки `q_found` и
`beta_found` выбираются только следующим способом:

1. forward-модель ранжирует глобальную сетку `(q,beta)`;
2. вокруг лучшего прогноза строится локальная сетка;
3. лучшие кандидаты обеих сеток проверяются по точному `J_300`;
4. победитель дополнительно уточняется локальным поиском по точному `J_300`.

Внутри обратного поиска формулы Поляка не вызываются. Теоретические столбцы
рассчитываются отдельно и используются для диагностического oracle, графиков и
итоговых тестовых `R²`. Поэтому совпадение финальных моделей с Поляком больше
не обеспечивается принудительным включением теоретической точки в обратный
поиск.

При этом генератор forward-датасета по-прежнему совпадает с основным
экспериментом `hb`: часть обучающих кандидатов располагается около точки
Поляка для более плотного покрытия практически важной области. Их таргеты не
берутся из теории — каждый `J_300` рассчитывается точной симуляцией. Если нужен
вариант вообще без теоретически ориентированного сэмплирования, генератор
forward-датасета также следует заменить на полностью нейтральный.

## Выбор степени без утечки

Для forward-модели разбиение выполняется по целым группам `kappa`: все
кандидаты одной квадратичной задачи остаются в одной части данных. Для
итоговых `q`- и `beta`-моделей один раз разбиваются строки `(L,l)`, поэтому
обе модели оцениваются на одинаковом тестовом наборе.

Используется схема:

```text
60% train
20% validation
20% test
```

Для каждой из трёх моделей:

1. каждая кандидатная степень обучается только на train;
2. для неё вычисляется MSE на validation;
3. выбирается степень с минимальным validation MSE;
4. при равенстве MSE выбирается меньшая степень;
5. выбранная модель переобучается на train+validation;
6. итоговые `R²` вычисляются только на test.

Test не участвует ни в выборе степени, ни в переобучении. По умолчанию для
каждой модели независимо проверяются степени от 1 до 10. Диапазоны задаются
аргументами:

```text
--forward-degrees
--q-degrees
--beta-degrees
```

## Настройки ЭОПКГ

Каждая модель создаётся с параметрами:

```python
EIKGPolynomialRegressor(
    degree=degree,
    regularization="ridge",
    alpha_ridge=1e-5,
    fit_intercept=True,
    scale=True,
    scale_y=True,
    normalize_latent=True,
)
```

`EIKGPolynomialRegressor` не поддерживает `sample_weight`. Чтобы сохранить
целочисленные веса `1`, `4` и `8` исходного forward-эксперимента, строки train
и validation повторяются соответствующее число раз. На test метрики считаются
по исходным строкам без повторений. Это можно отключить параметром
`--disable-forward-weight-repetition`.

## Установка

Для `eikgpolynomial 0.2.0` требуется Python 3.10 или новее. Из корня проекта:

```powershell
.\venv\Scripts\python.exe -m pip install -r hb_eikg\requirements.txt
```

Если основные зависимости проекта уже установлены:

```powershell
.\venv\Scripts\python.exe -m pip install "eikgpolynomial[sklearn]==0.2.0"
```

## Полный запуск

```powershell
.\venv\Scripts\python.exe -m hb_eikg.pipeline
```

Результаты, созданные старой версией пайплайна, не обновляются автоматически.
После изменения логики меток необходимо заново выполнить полный запуск; он
перезапишет старые данные, модели и `results/metrics.json`.

Последовательно проверяются степени трёх моделей, после чего выбранные модели
переобучаются на train+validation. Поэтому полный запуск может быть дольше
варианта с градиентным бустингом.

## Быстрая техническая проверка

Команда ниже нужна только для проверки окружения и форматов файлов. Её метрики
нельзя использовать как итоговые:

```powershell
.\venv\Scripts\python.exe -m hb_eikg.pipeline `
  --forward-groups 200 `
  --forward-candidates 48 `
  --forward-degrees 1 2 3 4 `
  --q-degrees 1 2 3 4 `
  --beta-degrees 1 2 3 4 `
  --oracle-points 64 `
  --oracle-grid 32 `
  --oracle-refine 17 `
  --meta-pairs 800 `
  --search-grid 32 `
  --predicted-refine 16 `
  --exact-top-k 12 `
  --exact-refine 11 `
  --batch-size 32
```

Быстрый запуск перезаписывает локальные данные и модели. Перед анализом
итоговых метрик после него нужно снова выполнить полный запуск.

## Использование итоговых моделей

После полного обучения, например для `L=1000`, `l=10`:

```powershell
.\venv\Scripts\python.exe -m hb_eikg.predict 1000 10
```

Скрипт выводит прогнозы `alpha` и `beta`, теоретические параметры Поляка и
абсолютные различия.

Программный вызов:

```python
from hb_eikg.predict import predict

alpha, beta = predict(L=1000.0, l=10.0)
```

## Итоговые метрики

Следующие `R²` вычисляются исключительно на test:

```text
metrics.forward_model_test.test_scaled_r2
metrics.forward_model_test.test_near_optimum_scaled_r2
metrics.forward_model_test.test_raw_J300_r2
metrics.final_models_test.test_alpha_vs_polyak_r2
metrics.final_models_test.test_beta_vs_polyak_r2
```

Выбранные степени и validation MSE находятся отдельно в разделе
`model_selection` файла `results/metrics.json` и в CSV. Они являются
диагностикой выбора гиперпараметров, а не тестовыми метриками.

## Сохраняемые файлы

```text
data/initial_points.npy
data/oracle_diagnostic.csv
data/forward_dataset.pkl
data/forward_degree_selection.csv
data/inverse_labels_all.csv
data/q_degree_selection.csv
data/beta_degree_selection.csv
data/final_models_test.csv

models/eikg_forward.joblib
models/eikg_q.joblib
models/eikg_beta.joblib
models/eikg_heavy_ball.joblib

results/metrics.json
results/degree_selection.png
results/oracle_vs_polyak.png
results/eikg_parameters_test.png
```

В `forward_dataset.pkl` и `inverse_labels_all.csv` для каждой строки явно
указано поле `split`.

## Связь с основным экспериментом

Папка переиспользует только детерминированные численные функции из
`hb.pipeline`: генерацию задач, вычисление `J_300`, oracle и точный обратный
поиск. Модели, разбиения, выбор степеней, артефакты и метрики изолированы
внутри `hb_eikg`.
