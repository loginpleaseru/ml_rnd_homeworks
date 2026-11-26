# ДЗ3, проект "Cassava Leaf Disease Classification"
## Логинов Никита

Анализ ошибок и устойчивости full_h1_h2 из ДЗ2.

**Данные:** картинки `../hw_1/data/cassava_train`, чекпоинт `../hw_2/checkpoints/full_h1_h2.pt` (см. `configs/analysis.json`).

Ноутбук - [notebooks/hw3_analysis.ipynb](notebooks/hw3_analysis.ipynb).
В нем пять ошибок с объяснениями, качество по классам, срезы по яркости и контрасту, проверки яркости и шума.

### Запуск

```bash
cd hw_3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hw3 --display-name "Python (hw3)"
jupyter notebook notebooks/hw3_analysis.ipynb
```



### Результаты

Ноутбук выполнен на модели full_h1_h2 и прежних 800 val изображениях.

| Условие | Macro F1 |
|---|---:|
| Исходные фото | 0.4727 |
| Яркость ×0.7 | 0.4112 |
| Яркость ×1.3 | 0.4362 |
| Шум sigma=0.03 | 0.4300 |
| Шум sigma=0.06 | 0.4373 |

Хуже всего CGM: F1=0.2222, найдено 13 из 89 изображений.
Изменение яркости снижает качество \
при шуме заметно страдает Healthy.
Причины ошибок по фотографиям обозначены как гипотезы.

В [reports](reports) сохранены CSV предсказаний, таблицы срезов, интервалы разницы F1 и графики.

Повторный запуск без интерфейса Jupyter:

```bash
python scripts/run_notebook.py --kernel hw3
```

