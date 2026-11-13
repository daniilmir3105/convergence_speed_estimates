# Gradient Descent с ЭОПКГ

Независимый вариант finite-horizon эксперимента `gd`, в котором оба
градиентных бустинга заменены на
`EIKGPolynomialRegressor` — элементарный образ полинома
Колмогорова–Габора (ЭОПКГ).

Численная постановка, генерация данных, oracle и exact-refinement совпадают с
основным экспериментом `gd`. Это позволяет сравнивать модели на одинаковых
данных и при одинаковом критерии качества.

Библиотека: [eikgpolynomial на PyPI](https://pypi.org/project/eikgpolynomial/).
Имя устанавливаемого пакета — `eikgpolynomial`, а Python-импорт выполняется
из модуля `eikg`:

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

Первая ЭОПКГ-модель прогнозирует:

\[
y_{\mathrm{first}}=\sqrt{\kappa}\,J_{300}.
\]

После поиска и точного уточнения кандидатного шага итоговая ЭОПКГ-модель
обучается отображению `(L,l) -> q`, где

\[
q=\frac{\alpha L}{2},
\qquad
\alpha=\frac{2q}{L}.
\]

Сохранённая публичная модель принимает непосредственно `[L,l]` и возвращает
обычный шаг `alpha`.

## Выбор степени без утечки

Для forward-модели разбиение выполняется по целым группам `kappa`, поэтому
кандидаты одной задачи не попадают одновременно в разные части данных.
Для итоговой модели отдельно разбиваются строки `(L,l)`.

В обоих случаях используется схема:

```text
60% train
20% validation
20% test
```

Для каждой кандидатной степени:

1. `EIKGPolynomialRegressor` обучается только на train;
2. вычисляется MSE на validation;
3. выбирается степень с минимальным validation MSE;
4. при точном равенстве MSE выбирается меньшая степень;
5. модель выбранной степени переобучается на train+validation;
6. итоговые `R²` считаются один раз на test.

Test не участвует ни в выборе степени, ни в переобучении.

По умолчанию независимо проверяются степени от 1 до 10 для первой и итоговой
моделей. Диапазоны можно изменить параметрами `--forward-degrees` и
`--final-degrees`.

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
и validation детерминированно повторяются соответствующее число раз. На test
метрики всегда считаются по исходным строкам без повторений. Повторение можно
отключить флагом `--disable-forward-weight-repetition`, но тогда сравнение с
основным экспериментом будет менее прямым.

## Установка библиотеки

Для `eikgpolynomial 0.2.0` требуется Python 3.10 или новее. Из корня проекта
рекомендуется установить полный локальный список зависимостей:

```powershell
.\venv\Scripts\python.exe -m pip install -r gd_eikg\requirements.txt
```

Если основные зависимости проекта уже установлены, достаточно добавить только
библиотеку ЭОПКГ:

```powershell
.\venv\Scripts\python.exe -m pip install "eikgpolynomial[sklearn]==0.2.0"
```

## Полный запуск

```powershell
.\venv\Scripts\python.exe -m gd_eikg.pipeline
```

Пайплайн может работать дольше версии с бустингом: для каждой из двух моделей
последовательно обучаются все кандидатные степени, после чего две выбранные
модели переобучаются на train+validation.

## Быстрая техническая проверка

Эта команда проверяет окружение и форматы файлов. Её результаты нельзя
использовать как финальные:

```powershell
.\venv\Scripts\python.exe -m gd_eikg.pipeline `
  --forward-groups 200 `
  --forward-candidates 48 `
  --forward-degrees 1 2 3 4 `
  --final-degrees 1 2 3 4 `
  --oracle-points 64 `
  --oracle-grid 96 `
  --oracle-refine 25 `
  --oracle-stages 2 `
  --meta-pairs 800 `
  --search-grid 96 `
  --predicted-refine 32 `
  --exact-top-k 16 `
  --exact-refine 21 `
  --exact-stages 2 `
  --batch-size 64
```

Быстрый запуск перезаписывает локальные данные и модели. После него для
получения итоговых результатов нужно снова выполнить полный запуск.

## Использование итоговой модели

После обучения, например для `L=1000`, `l=10`:

```powershell
.\venv\Scripts\python.exe -m gd_eikg.predict 1000 10
```

Скрипт выводит прогноз ЭОПКГ, классический шаг

\[
\alpha_{\mathrm{theory}}=\frac{2}{L+l},
\]

а также абсолютное и относительное различие.

Программный вызов:

```python
from gd_eikg.predict import predict

alpha = predict(L=1000.0, l=10.0)
```

## Итоговые метрики

Все следующие `R²` вычисляются исключительно на test:

```text
metrics.forward_model_test.test_scaled_r2
metrics.forward_model_test.test_near_optimum_scaled_r2
metrics.forward_model_test.test_raw_J300_r2
metrics.final_model_test.test_alpha_vs_classical_theory_r2
```

Степени и validation MSE хранятся отдельно в разделе `model_selection` файла
`results/metrics.json` и в CSV. Они описывают выбор гиперпараметра, но не
выдаются за тестовые метрики.

## Сохраняемые файлы

```text
data/initial_points.npy
data/oracle_diagnostic.csv
data/forward_dataset.pkl
data/forward_degree_selection.csv
data/inverse_labels_all.csv
data/final_degree_selection.csv
data/final_model_test.csv

models/eikg_forward.joblib
models/eikg_alpha.joblib

results/metrics.json
results/degree_selection.png
results/oracle_vs_theory.png
results/eikg_alpha_test.png
```

`forward_degree_selection.csv` и `final_degree_selection.csv` содержат MSE
каждой проверенной степени на validation. В `forward_dataset.pkl` и
`inverse_labels_all.csv` для каждой строки явно указано поле `split`.

## Связь с основным экспериментом

Новая папка переиспользует только детерминированные численные функции из
`gd.pipeline`: генерацию квадратичных задач, вычисление `J_300`, oracle и
точный обратный поиск. Модели, разбиения, выбор степени, артефакты и метрики
полностью изолированы внутри `gd_eikg`.
