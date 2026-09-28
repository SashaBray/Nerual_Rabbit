"""Runtime-hook PyInstaller: проверка собранного приложения без запуска интерфейса.

Выполняется ДО основного скрипта и только если задана переменная окружения
``NR_SMOKE_TEST`` (путь к файлу результата) — обычный запуск программы её не видит.
В проверочном режиме импортирует всё приложение (``app_gui`` со всеми зависимостями:
torch, sklearn, pyarrow, PySide6…) и завершается через ``os._exit``: без окна «Unhandled
exception», которое повесило бы процесс. Код выхода 0 — всё импортируется, 3 — ошибка
(трассировка в файле результата).

Нужен потому, что сборка может пройти успешно, а exe — падать при запуске (так было, когда
из сборки исключили нужный torch модуль). ``make_exe.py`` запускает этот режим после сборки.
"""
import os

_result = os.environ.get('NR_SMOKE_TEST')
if _result:
    import traceback
    try:
        import app_gui  # noqa: F401 — модуль целиком, main() не вызывается
        import torch
        import sklearn.ensemble  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        torch.zeros(3).sum().item()
        with open(_result, 'w', encoding='utf-8') as f:
            f.write('ok\n')
        os._exit(0)
    except BaseException:                                  # noqa: BLE001 — любая ошибка = провал проверки
        with open(_result, 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        os._exit(3)
