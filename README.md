# Churn ML Production

ML-проект прогнозирования оттока клиентов, подготовленный к production: Poetry, pre-commit, ruff, mypy, src-layout.

## Структура проекта

```text
.
├── src/churn_model/        # код обучения модели
│   ├── __init__.py
│   └── train.py
├── tests/                  # тесты
├── data/
│   ├── raw/                # исходные данные (не коммитятся)
│   └── processed/          # обработанные данные (не коммитятся)
├── models/                 # артефакты моделей (не коммитятся)
├── model_card.md           # Model Card по шаблону Google
├── Model_BPMN.md           # место модели в бизнес-процессе
├── Model_retrain_BPMN.md   # процесс автоматического переобучения
├── pyproject.toml          # зависимости и настройки инструментов
├── poetry.lock             # зафиксированные версии зависимостей
└── .pre-commit-config.yaml # хуки pre-commit
