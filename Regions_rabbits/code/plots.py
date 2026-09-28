"""Единый помощник для графиков проекта.

* сохраняет КАЖДЫЙ график в PNG и PDF (PDF — для статей);
* рендерит график в ДВУХ версиях: русской и английской (для статей);
  файлы получают суффикс языка: ``<name>_ru.png/pdf`` и ``<name>_en.png/pdf``.

Использование в коде графиков:
    import plots
    def _draw(lang):
        fig, ax = plt.subplots()
        ax.set_xlabel(plots.tr('Недель', 'Weeks', lang))
        ...
        return fig
    ru_png = plots.bilingual(outdir, 'имя_графика', _draw)   # путь к RU-PNG (для docx)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LANGS = ('ru', 'en')
FORMATS = ('png', 'pdf')


def tr(ru, en, lang):
    """Выбрать строку по языку."""
    return ru if lang == 'ru' else en


def savefig(fig, outdir, name, dpi=120):
    """Сохранить фигуру во все форматы (png+pdf). Вернуть путь к PNG."""
    os.makedirs(outdir, exist_ok=True)
    png = None
    for ext in FORMATS:
        p = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        if ext == 'png':
            png = p
    plt.close(fig)
    return png


def bilingual(outdir, name, draw, dpi=120):
    """draw(lang)->Figure. Рендерит RU и EN, каждую в png+pdf.

    Возвращает путь к RU-PNG (его встраиваем в русские docx-отчёты).
    """
    ru_png = None
    for lang in LANGS:
        fig = draw(lang)
        png = savefig(fig, outdir, f'{name}_{lang}', dpi)
        if lang == 'ru':
            ru_png = png
    return ru_png
