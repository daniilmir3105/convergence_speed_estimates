# Learning Convergence Rates

Проект исследует data-driven подход к оценке скорости сходимости и автоматическому подбору параметров алгоритмов оптимизации. Основной критерий — фактическое уменьшение ошибки за фиксированный вычислительный бюджет в 300 итераций. Аналитические формулы используются как независимые ориентиры для сравнения, но не как обучающие метки.

В репозитории представлены три направления:

1. Подбор постоянного шага для градиентного спуска и шага с импульсом для метода тяжёлого шарика.
2. Прогноз скорости гиподифференциального спуска на максимумах выпуклых квадратичных функций.
3. Практический подбор learning rate для MLP на наборе Boston Housing.

## Структура проекта

| Папка | Назначение | Модель или итоговый интерфейс |
|---|---|---|
| [`gd`](gd/) | Gradient Descent на квадратичных задачах, finite horizon | `(L, l) → alpha`, Gradient Boosting |
| [`gd_eikg`](gd_eikg/) | Тот же GD-пайплайн с ЭОПКГ | `(L, l) → alpha`, `EIKGPolynomialRegressor` |
| [`hb`](hb/) | Heavy Ball на квадратичных задачах, finite horizon | `(L, l) → (alpha, beta)`, Gradient Boosting |
| [`hb_eikg`](hb_eikg/) | Тот же HB-пайплайн с ЭОПКГ | `(L, l) → (alpha, beta)`, `EIKGPolynomialRegressor` |
| [`hypodifferential_rate`](hypodifferential_rate/) | Базовый прогноз скорости гиподифференциального спуска | признаки функции → `(J_300, rho)` |
| [`hypodifferential_rate_enhanced`](hypodifferential_rate_enhanced/) | Сбалансированный датасет и расширенные геометрические признаки | признаки функции → `(J_300, rho)` |
| [`hypodifferential_rate_0_1`](hypodifferential_rate_0_1/) | Учет всех индивидуальных пар `L_i, l_i` и широкого диапазона числа компонент | признаки всех компонент → `(J_300, rho)` |
| [`boston_mlp_experiment`](boston_mlp_experiment/) | Подбор learning rate для `MLPRegressor` | learning rate → validation MSE |

Локальные README каждой папки содержат детальное описание конкретного эксперимента.

## Finite-horizon критерий

Для Gradient Descent и Heavy Ball используется показатель

\[
J_{300}=
\frac{1}{16\cdot300}
\sum_{i=1}^{16}
\log_{10}\left[
\max\left(
\frac{f(x_{300}^{(i)})}{f(x_0^{(i)})},
10^{-300}
\right)
\right].
\]

Каждый набор параметров проверяется из 16 фиксированных начальных точек на горизонте 300 итераций. Чем отрицательнее `J_300`, тем быстрее за этот бюджет уменьшилась целевая функция. Первая, или forward-модель, обучается прогнозировать масштабированный таргет

\[
y_{\mathrm{first}}=\sqrt{\kappa}\,J_{300},
\qquad \kappa=\frac{L}{l}.
\]

Масштабирование повышает различимость кандидатов при больших числах обусловленности, но не меняет лучший параметр внутри одной задачи, поскольку множитель зависит только от `kappa`.

В экспериментах с гиподифференциальным спуском используется относительный разрыв

\[
e_k=f(x_k)-f^*,
\]

и тот же finite-horizon принцип:

\[
J_{300}^{\mathrm{HD}}=
\frac{1}{16\cdot300}
\sum_{i=1}^{16}
\log_{10}\left[
\max\left(
\frac{e_{300}^{(i)}}{e_0^{(i)}},
10^{-300}
\right)
\right].
\]

Для исследуемых однородных максимумов квадратичных функций известны `x*=0` и `f*=0`, поэтому `e_k=f(x_k)`. Дополнительно рассчитывается

\[
\rho=10^{J_{300}},
\]

то есть средний коэффициент сохранения ошибки за одну итерацию: чем меньше `rho`, тем быстрее сходимость.

## Пайплайн GD и Heavy Ball

Общая схема синтетических экспериментов:

```text
характеристики задачи + проверяемые параметры оптимизатора
                          ↓
точное вычисление J_300 на синтетической квадратичной задаче
                          ↓
forward-модель скорости сходимости
                          ↓
поиск и точное finite-horizon уточнение лучших кандидатов
                          ↓
метки оптимальных параметров без теоретических кандидатов
                          ↓
итоговая модель: (L, l) → параметры оптимизатора
```

Рабочий диапазон синтетических задач:

\[
1.1\leq\kappa\leq20000.
\]

Для независимого контроля используются классические ориентиры:

\[
\alpha_{\mathrm{GD}}^*=\frac{2}{L+l},
\]

\[
\alpha_{\mathrm P}^*=\frac{4}{(\sqrt L+\sqrt l)^2},
\qquad
\beta_{\mathrm P}^*=\left(
\frac{\sqrt L-\sqrt l}{\sqrt L+\sqrt l}
\right)^2.
\]

Эти значения не добавляются в кандидаты при формировании finite-horizon меток. Поэтому сравнение итоговых моделей с классической теорией является диагностикой, а не следствием прямого копирования формул.

## Гиподифференциальный спуск

Рассматриваются функции

\[
f(x)=\max_{1\leq j\leq m}\frac12x^{\mathsf T}A_jx,
\]

где матрицы `A_j` симметричны и положительно определены. На каждой итерации вычисляется отрицательная проекция нуля на гиподифференциал, после чего шаг определяется точным одномерным поиском. Постоянный `alpha` здесь не является настраиваемым параметром, поэтому решается только прямая задача:

```text
характеристики функции → J_300 → rho
```

Эксперименты последовательно расширяют постановку:

- `hypodifferential_rate` использует глобальные спектральные и агрегированные геометрические признаки;
- `hypodifferential_rate_enhanced` использует 5000 задач, сбалансированный план, спектральные квантили, геометрию компонент и характеристики первого шага, вычисляемые до основного 300-шагового запуска;
- `hypodifferential_rate_0_1` передаёт модели число функций и все индивидуальные пары `L_i,l_i`, упорядоченные и дополненные масками отсутствующих компонент. Диапазон `m` расширен до 2–20.

В `hypodifferential_rate_0_1` дополнительно обучаются контрольные модели только по размерности, числу компонент и глобальному `L/l`. Это позволяет непосредственно измерить пользу сведений об отдельных функциях под максимумом.

## Текущие результаты

Ниже приведены метрики из сохранённых файлов результатов. Для ЭОПКГ и гиподифференциальных моделей итоговые показатели рассчитаны на независимой тестовой части.

### Gradient Descent и Heavy Ball

| Эксперимент | Forward `R²` в исходных единицах `J_300` | Итоговые модели |
|---|---:|---:|
| GD, Gradient Boosting | 0.9992 | `alpha` против классической теории: 0.9985 |
| GD, ЭОПКГ | 0.5060 | `alpha` против классической теории: 0.8485 |
| Heavy Ball, Gradient Boosting | 0.9953 | `alpha`: 0.9953; `beta`: 0.9986 против Поляка |
| Heavy Ball, ЭОПКГ | 0.6667 | `alpha`: 0.8126; `beta`: 0.9160 против Поляка |

### Гиподифференциальный спуск

| Эксперимент | Модель | Test `R²` для `J_300` | Test `R²` для `rho` |
|---|---|---:|---:|
| Базовый | Gradient Boosting | 0.6794 | 0.6703 |
| Базовый | Extra Trees | 0.7663 | 0.7502 |
| Базовый | ЭОПКГ | 0.4517 | 0.4438 |
| Расширенный | Extra Trees | **0.8536** | **0.8491** |
| Расширенный | ЭОПКГ | 0.6367 | 0.6239 |
| Все `L_i,l_i` | Gradient Boosting | 0.7843 | 0.7996 |
| Все `L_i,l_i` | Extra Trees | 0.7515 | 0.7717 |
| Все `L_i,l_i` | ЭОПКГ | 0.2894 | 0.3160 |

В эксперименте со всеми `L_i,l_i` Gradient Boosting повысил тестовый `R²` для `J_300` с 0.4900 у контрольной модели до 0.7843, а Extra Trees — с 0.5977 до 0.7515. Это подтверждает, что глобальные `L`, `l` и количество компонент не полностью описывают внутреннюю структуру максимума. Текущий сохранённый запуск этой папки содержит 1000 функций, хотя генератор по умолчанию настроен на 5000; поэтому его результаты следует считать предварительными.

### Boston Housing

Для MLP `13 → 32 → 32 → 1` с ReLU и Adam модель-суррогат прогнозирует validation MSE по learning rate. Качество суррогата на out-of-fold прогнозах составляет `R²=0.9701`. Тестовые `R²` итоговых MLP равны 0.7744 для learning rate суррогата, 0.7830 для Grid Search и 0.8009 для LR Finder.

## Установка

Пример для Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib joblib
```

Для экспериментов с ЭОПКГ:

```powershell
python -m pip install "eikgpolynomial[sklearn]==0.2.0"
```

Все команды ниже выполняются из корня проекта.

## Запуск GD и Heavy Ball

Gradient Descent с Gradient Boosting:

```powershell
.\venv\Scripts\python.exe -m gd.pipeline
.\venv\Scripts\python.exe -m gd.predict 1000 10
```

Gradient Descent с ЭОПКГ:

```powershell
.\venv\Scripts\python.exe -m gd_eikg.pipeline
.\venv\Scripts\python.exe -m gd_eikg.predict 1000 10
```

Heavy Ball с Gradient Boosting:

```powershell
.\venv\Scripts\python.exe hb\pipeline.py
.\venv\Scripts\python.exe hb\predict.py 1000 10
```

Heavy Ball с ЭОПКГ:

```powershell
.\venv\Scripts\python.exe -m hb_eikg.pipeline
.\venv\Scripts\python.exe -m hb_eikg.predict 1000 10
```

## Запуск гиподифференциальных экспериментов

Базовый датасет и Gradient Boosting:

```powershell
.\venv\Scripts\python.exe -m pip install -r hypodifferential_rate\requirements.txt
.\venv\Scripts\python.exe -m hypodifferential_rate.generate_dataset --n-jobs -1
.\venv\Scripts\python.exe -m hypodifferential_rate.train
```

Дополнительные модели на том же базовом датасете:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate.train_extratrees --n-jobs -1
.\venv\Scripts\python.exe -m hypodifferential_rate.train_eikg
```

Расширенный эксперимент:

```powershell
.\venv\Scripts\python.exe -m pip install -r hypodifferential_rate_enhanced\requirements.txt
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.generate_dataset --n-jobs -1
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.train --n-jobs -1
.\venv\Scripts\python.exe -m hypodifferential_rate_enhanced.train_eikg
```

Эксперимент со всеми индивидуальными `L_i,l_i`:

```powershell
.\venv\Scripts\python.exe -m pip install -r hypodifferential_rate_0_1\requirements.txt
.\venv\Scripts\python.exe -m hypodifferential_rate_0_1.generate_dataset --instances 5000 --n-jobs -1
.\venv\Scripts\python.exe -m hypodifferential_rate_0_1.train --n-jobs -1
```

Прогноз `J_300` и `rho` для собственного NPY-массива матриц:

```powershell
.\venv\Scripts\python.exe -m hypodifferential_rate_0_1.predict --instance path\to\matrices.npy
```

## Запуск Boston Housing

```powershell
.\venv\Scripts\python.exe boston_mlp_experiment\experiment.py
```

Эксперимент использует локальный кэш набора из 506 наблюдений, выполняет медианное заполнение пропусков, winsorization выбросов и стандартизацию, причём параметры предобработки оцениваются только по обучающей части.

## Артефакты и Git

Каждый эксперимент сохраняет данные, модели, CSV с прогнозами, JSON с метриками и графики в собственной папке. Файлы обученных моделей `*.joblib`, `*.cbm`, `*.pkl` и `*.pickle` исключены из Git корневым `.gitignore`, поскольку некоторые из них превышают лимит GitHub в 100 МБ. После клонирования репозитория модели необходимо восстановить соответствующими командами обучения.

Не загружайте `joblib`- и `pickle`-файлы из недоверенных источников: их десериализация может выполнять произвольный код.

## Ограничения

- Основные GD/HB-результаты относятся к синтетическим положительно определённым квадратичным задачам и горизонту 300 итераций.
- Гиподифференциальные эксперименты рассматривают более узкий класс однородных максимумов выпуклых квадратичных функций с известным минимумом.
- Качество моделей за пределами обучающих диапазонов не проверялось; древесные ансамбли не предназначены для надёжной экстраполяции.
- Высокий `R²` forward-модели не гарантирует точного положения минимума по параметрам, поэтому GD/HB-пайплайны используют точную finite-horizon проверку и локальное уточнение кандидатов.
- Сравнение с формулами GD и Поляка не означает, что конечногоризонтные параметры обязаны с ними совпадать: оптимизируются разные критерии.
