# -*- mode: python ; coding: utf-8 -*-
"""Сборка приложения в папку с exe (PyInstaller).

    pip install pyinstaller
    python make_exe.py            # справочники по белому списку + PyInstaller (запускать ЕГО, не spec)

Результат: ``dist/NerualRabbit/NerualRabbit.exe`` + библиотеки рядом. Папку целиком
архивируют и раздают; при первом запуске программа создаёт рядом с exe каталог
``workspace`` и предлагает скачать базу и модели из каталога (см. code/assets.py).

Данные (база, модели, датасеты) в сборку НЕ входят — они приезжают из публичной папки
автора. Торч берётся CPU-сборкой: `pip install torch --index-url
https://download.pytorch.org/whl/cpu` (сборка с CUDA раздула бы дистрибутив в разы).
"""
import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.getcwd())
CODE = os.path.join(ROOT, 'code')

hidden = (collect_submodules('sklearn')          # модели грузятся из pickle — импорты не видны статически
          + ['pyarrow', 'pyarrow.parquet', 'docx', 'openpyxl', 'xlrd'])

a = Analysis(
    [os.path.join(CODE, 'app_gui.py')],
    pathex=[CODE],
    binaries=[],
    datas=[(os.path.join(ROOT, 'logos'), 'logos'),        # логотип/иконка: config.RESOURCES
           # справочники по умолчанию (белый список, без ukey) — собирает make_exe.py
           (os.path.join(ROOT, 'build', 'workspace_defaults'), 'workspace_defaults')],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[os.path.join(ROOT, 'smoke_hook.py')],   # проверочный запуск (NR_SMOKE_TEST), см. make_exe.py
    # НЕ исключать подмодули torch: torch/__init__ импортирует их сам (torch.distributions и т.п.) —
    # сборка при этом проходит, а exe падает при запуске
    excludes=['tkinter', 'PyQt5', 'PyQt6', 'IPython', 'jupyter', 'notebook',
              'torchvision', 'torchaudio'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='NerualRabbit',
    console=False,                                        # оконное приложение, без чёрной консоли
    icon=os.path.join(ROOT, 'logos', 'NR_app_v7.ico'),   # значок exe и ярлыков (кролик из логотипа v7)
)

coll = COLLECT(exe, a.binaries, a.datas, name='NerualRabbit')
