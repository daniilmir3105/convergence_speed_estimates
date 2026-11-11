# ML-подбор параметров алгоритмов оптимизации

Проект исследует универсальный двухэтапный ML-подход к настройке параметров
оптимизаторов. Основные синтетические эксперименты посвящены градиентному
спуску и методу тяжёлого шарика на положительно определённых квадратичных
функциях. Качество параметров оценивается на фиксированном горизонте из 300
итераций, а не через спектральный или бесконечно-асимптотический таргет.

Практические эксперименты на Diabetes и Boston Housing показывают родственную
идею: зависимость качества обучения нейросети от learning rate сначала
измеряется серией запусков, а затем аппроксимируется градиентным бустингом.

## Состав проекта

| Папка | Эксперимент | Итоговый интерфейс |
|---|---|---|
| [`gd`](gd/) | Gradient Descent, finite horizon | `(L, l) -> alpha` |
| [`hb`](hb/) | Heavy Ball, finite horizon | `(L, l) -> (alpha, beta)` |
| [`diabetes_sgd_experiment`](diabetes_sgd_experiment/) | Подбор learning rate для PyTorch MLP с SGD | `learning rate -> validation MSE` |
| [`boston_mlp_experiment`](boston_mlp_experiment/) | Подбор learning rate для sklearn MLP с Adam | `learning rate -> validation MSE` |

Здесь `L` и `l` — максимальная и минимальная кривизны квадратичной задачи,
`kappa=L/l` — число обусловленности, `alpha` — шаг, а `beta` — momentum.

## Главный finite-horizon критерий

В папках `gd` и `hb` используется один и тот же фактический показатель
сходимости:

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

- рассматриваются 16 фиксированных начальных точек;
- каждый кандидатный набор параметров проверяется на 300 итерациях;
- чем меньше, то есть чем отрицательнее, `J_300`, тем быстрее уменьшалась
  целевая функция;
- порог `10^-300` защищает вычисление от `log10(0)`.

Первая, или forward-модель, обучается предсказывать масштабированную версию:

\[
y_{\mathrm{first}}=\sqrt{\kappa}\,J_{300}.
\]

Множитель `sqrt(kappa)` улучшает различимость значений при больших числах
обусловленности, но не меняет лучший параметр внутри одной задачи, поскольку
для всех её кандидатов он одинаков и положителен.

## Общая логика GD и Heavy Ball

```text
kappa + кандидатные параметры
              ↓
точное вычисление J_300
              ↓
forward-модель: параметры -> sqrt(kappa) * J_300
              ↓
ML-ранжирование кандидатных параметров
              ↓
точная повторная оценка и локальное уточнение по J_300
              ↓
finite-horizon метки оптимальных параметров
              ↓
финальная модель: (L, l) -> параметры оптимизатора
```

Для контроля дополнительно выполняется exact oracle: прямой поиск минимума
настоящего `J_300` без forward-модели. В Heavy Ball точка Поляка добавляется в
набор проверяемых кандидатов, но побеждает только в том случае, если её
фактический `J_300` не хуже альтернатив. Теоретические формулы не копируются в
датасет как готовые обучающие метки.

Диапазон обоих синтетических экспериментов:

\[
1.1\leq\kappa\leq20000,
\qquad 1\leq l\leq50.
\]

## Теоретические ориентиры

Для Gradient Descent используется классический оптимальный постоянный шаг:

\[
\alpha_{\mathrm{GD}}^*=\frac{2}{L+l}.
\]

Для метода тяжёлого шарика используются параметры Поляка:

\[
\alpha_{\mathrm P}^*=
\frac{4}{(\sqrt L+\sqrt l)^2},
\qquad
\beta_{\mathrm P}^*=
\left(\frac{\sqrt L-\sqrt l}{\sqrt L+\sqrt l}\right)^2.
\]

Эти значения нужны только для независимого итогового сравнения. Метки второй
модели получаются минимизацией finite-horizon критерия `J_300`.

## Какие метрики сохраняются

Вывод синтетических пайплайнов намеренно сокращён до ключевых `R²`.

Первая модель в обоих экспериментах:

```text
metrics.forward_model.validation_scaled_r2
metrics.forward_model.near_optimum_scaled_r2
metrics.forward_model.validation_raw_J300_r2
```

- `validation_scaled_r2` — прогноз `sqrt(kappa) * J_300` на отдельных
  валидационных значениях `kappa`;
- `near_optimum_scaled_r2` — качество на 20% лучших по настоящему `J_300`
  кандидатов внутри каждого валидационного `kappa`;
- `validation_raw_J300_r2` — качество после возврата к исходным единицам
  `J_300`.

Итоговые модели:

```text
GD: metrics.final_model_vs_classical_gd_theory.alpha_r2
HB: metrics.final_models_vs_polyak.alpha_r2
HB: metrics.final_models_vs_polyak.beta_r2
```

Oracle, точное уточнение и дополнительные значения траекторий остаются в CSV
и графиках, но дополнительные диагностические метрики в `metrics.json` не
записываются.

## Установка

Пример для Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn matplotlib joblib torch
```

Все команды ниже выполняются из корня проекта.

## Обучение синтетических пайплайнов

Gradient Descent:

```powershell
.\venv\Scripts\python.exe -m gd.pipeline
```

Heavy Ball:

```powershell
.\venv\Scripts\python.exe hb\pipeline.py
```

Полные варианты быстрых проверочных команд и описание аргументов находятся в
локальных инструкциях: [`gd/README.md`](gd/README.md) и
[`hb/README.md`](hb/README.md).

## Использование готовых моделей

Gradient Descent для `L=1000`, `l=10`:

```powershell
.\venv\Scripts\python.exe -m gd.predict 1000 10
```

Heavy Ball для тех же кривизн:

```powershell
.\venv\Scripts\python.exe hb\predict.py 1000 10
```

С точки зрения пользователя модели получают только `L` и `l`. Внутренние
признаки, такие как `log(kappa)` и `sqrt(kappa)`, вычисляются автоматически.
В Heavy Ball внутренние модели сначала предсказывают устойчивые координаты
`q` и `beta`, после чего восстанавливается обычный шаг `alpha`.

## Практические эксперименты

Diabetes, PyTorch MLP `10 -> 16 -> 16 -> 1`, ReLU, SGD без momentum:

```powershell
.\venv\Scripts\python.exe diabetes_sgd_experiment\experiment.py
```

Boston Housing, sklearn MLP со скрытыми слоями `(32, 32)`, ReLU и Adam:

```powershell
.\venv\Scripts\python.exe boston_mlp_experiment\experiment.py
```

В этих экспериментах forward-модель прогнозирует validation MSE по learning
rate. Найденное значение сравнивается с независимыми Grid Search и Learning
Rate Finder. Подробности находятся в README соответствующих папок.

## Основные артефакты GD и Heavy Ball

```text
gd/
├── pipeline.py
├── alpha_model.py
├── predict.py
├── data/
├── models/
│   ├── hist_gradient_boosting_finite_rate.joblib
│   └── gradient_boosting_alpha.joblib
└── results/
    ├── metrics.json
    ├── oracle_vs_theory.png
    └── alpha_model_vs_finite_labels.png

hb/
├── pipeline.py
├── predict.py
├── data/
├── models/
│   ├── hist_gradient_boosting_finite_rate.joblib
│   ├── gradient_boosting_q.joblib
│   └── gradient_boosting_beta.joblib
└── results/
    ├── metrics.json
    ├── oracle_vs_polyak.png
    └── direct_models_vs_finite_labels.png
```

Повторный запуск перезаписывает соответствующие данные, модели и результаты.
Быстрый режим следует использовать только для проверки работоспособности:
после него необходимо снова выполнить полный запуск перед анализом метрик.

## Ограничения

- Синтетические результаты относятся к двумерным положительно определённым
  квадратичным функциям и фиксированному горизонту `K=300`.
- Качество за пределами `1.1 <= kappa <= 20000` не проверялось.
- Древесные модели не предназначены для надёжной экстраполяции за пределы
  обучающего диапазона.
- Высокий `R²` forward-модели не гарантирует точного положения минимума,
  поэтому после ML-ранжирования применяется точная оценка `J_300`.
- Сериализованные `.joblib` и `.pt` модели следует загружать только из
  доверенного источника и с совместимыми версиями библиотек.
