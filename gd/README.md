# Gradient Descent alternative

Альтернативный конвейер для обратной задачи градиентного спуска. Внутренняя
`sklearn.ensemble.GradientBoostingRegressor` обучается предсказывать терминальную скорость сходимости, после
чего по ней находится оптимальный шаг и обучается финальная модель
`(L, l) -> alpha`.

## Полный запуск

Из корня проекта:

```powershell
.\venv\Scripts\python.exe gd_alter\pipeline.py
```

Конвейер создаст датасеты и сохранит:

- `models/gradient_boosting_terminal_rate.joblib` — внутреннюю forward-модель;
- `models/gradient_boosting_alpha.joblib` — финальную модель `(L,l) -> alpha`;
- `results/metrics.json` — метрики;
- `results/meta_vs_theory.png` — график против теории;
- `data/inverse_train.csv` и `data/meta_test.csv` — результаты эксперимента.

## Использование готовой модели

```powershell
.\venv\Scripts\python.exe gd_alter\predict.py 40 10
```

Скрипт выводит ML-прогноз, теоретический шаг `2/(L+l)` и абсолютную ошибку.

Программный вызов:

```python
from gd_alter.predict import predict, gd_theory

alpha = predict(L=40, l=10)
alpha_theory = gd_theory(L=40, l=10)
```

Модель обучена для `l` от 1 до 50 и `L/l` от 1.1 до 20. Экстраполяция за
пределы этого диапазона не проверялась.
