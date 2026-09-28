"""Оформление приложения: единая система цветов -> палитра Qt, таблица стилей (QSS), графики, ячейки.

Все цвета интерфейса задаются здесь, в словарях ``THEMES`` (токены: фон, поверхность, рамка,
текст, акцент...). Из одного набора токенов собираются:

* палитра ``QPalette`` для стиля Fusion — её видят диалоги, меню и всё, что не покрыто QSS;
* таблица стилей QSS — вкладки, кнопки, поля ввода, списки, таблицы, полосы прокрутки;
* параметры matplotlib (фон и подписи графиков в тон окну);
* цвета ячеек таблицы «Проверка урожайности» (``app_gui.C_MISS`` и т.д.).

Фирменные цвета взяты из логотипа (``logos/NR_logo_7.png``): фиолетово-синий неон кролика и
янтарное золото колосьев.

Кнопкам можно задать роль: ``btn.setProperty('role', 'primary' | 'danger')`` — главное действие
вкладки выделяется акцентом, «опасное» (удалить, выйти) — красным контуром. Для уже написанных
вкладок это делает :func:`mark_buttons` по тексту кнопок.

Использование::

    app = QApplication(sys.argv)
    app_theme.apply(app, 'field')      # до создания окон
    win = MainWindow()
    app_theme.mark_buttons(win)
"""

import os
import re
import sys
import tempfile
from string import Template

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QPushButton


# --------------------------------------------------------------------------- #
# Токены тем
# --------------------------------------------------------------------------- #
THEMES = {
    # Светлая: тёплая «бумага», белые карточки, фиолетовый акцент, янтарь колосьев.
    'field': {
        'title': 'Поле (светлая)',
        'dark': False,
        'bg': '#F4F3EF', 'surface': '#FFFFFF', 'surface_alt': '#FAF9F6', 'input': '#FFFFFF',
        'border': '#E4E1D8', 'border_strong': '#CFCABD',
        'hover': '#EFEDE7', 'pressed': '#E5E2DA',
        'text': '#1E2130', 'muted': '#6A6E80', 'disabled': '#A6A8B3',
        'accent': '#5B3FD9', 'accent_hover': '#4D33C4', 'accent_pressed': '#4029A8',
        'accent_soft': '#ECE8FD', 'on_accent': '#FFFFFF',
        'amber': '#E3A21A', 'amber_soft': '#FFF3D6',
        'danger': '#C4314B', 'danger_border': '#E6A3AF', 'danger_soft': '#FCEBEE',
        'scroll': '#CFCABD', 'scroll_hover': '#B3AD9E',
        'tooltip_bg': '#1E2130', 'tooltip_fg': '#FFFFFF',
        'cell_miss': '#FDE4E4', 'cell_user': '#E1F3E4', 'cell_edit': '#FFF0C2', 'cell_plain': '#FFFFFF',
        'plot_cycle': ['#5B3FD9', '#E3A21A', '#1F8FE5', '#1BA37E', '#D9468C', '#7A7F95'],
    },
    # Тёмная: глубокий ночной синий, неоновый фиолетовый акцент, янтарь.
    'night': {
        'title': 'Ночь (тёмная)',
        'dark': True,
        'bg': '#0E0F1A', 'surface': '#161827', 'surface_alt': '#1B1E31', 'input': '#12141F',
        'border': '#262A40', 'border_strong': '#363B58',
        'hover': '#20243A', 'pressed': '#2A2F4A',
        'text': '#E7E8F2', 'muted': '#9398B3', 'disabled': '#5D6180',
        'accent': '#8B6CFF', 'accent_hover': '#9D83FF', 'accent_pressed': '#7657F0',
        'accent_soft': '#2B2552', 'on_accent': '#FFFFFF',
        'amber': '#F5B03A', 'amber_soft': '#3A2E14',
        'danger': '#FF6B81', 'danger_border': '#6E2B3A', 'danger_soft': '#3A1A22',
        'scroll': '#363B58', 'scroll_hover': '#4A5075',
        'tooltip_bg': '#E7E8F2', 'tooltip_fg': '#0E0F1A',
        'cell_miss': '#4A2027', 'cell_user': '#1C3A2A', 'cell_edit': '#4A3A12', 'cell_plain': '#161827',
        'plot_cycle': ['#8B6CFF', '#F5B03A', '#3FA7FF', '#34D399', '#FF6FB5', '#A0A5C0'],
    },
}

# Родной стиль Windows 11 (Qt >= 6.7) + фирменный акцент — минимальное вмешательство.
WIN11 = {'title': 'Windows 11 (родной стиль)', 'accent': '#5B3FD9'}

# Главные действия вкладок и «опасные» кнопки (для mark_buttons). Сравнение по ПОЛНОМУ тексту
# кнопки без значков и стрелок по краям — иначе «Сохранить» зацепило бы «Сохранить как шаблон».
PRIMARY_TEXTS = {'Сделать прогноз', 'Сохранить', 'Сохранить введённое', 'Построить график',
                 'Показать таблицу', 'Сформировать отчёт по партии (CSV + Word)',
                 'Подгрузить урожайность (ЕМИСС)', 'Загрузить урожайности из архива в БД',
                 'Оценить точность (бэктест по выборке)', 'Войти', 'Создать'}
DANGER_TEXTS = {'Удалить партию', 'Выйти из аккаунта', 'Убрать моё значение'}


def _plain(text):
    """Текст кнопки без стрелок и прочих знаков по краям: 'Районы →' -> 'Районы'."""
    return re.sub(r'^[^\w(«]+', '', text).rstrip(' →←↓↑▾').strip()


# Выбранная тема хранится в таблице app_state (ключ 'theme'), как и сессия входа.
DEFAULT_THEME = 'night'
_STATE_KEY = 'theme'


def current_theme():
    """Сохранённая тема оформления; если не выбрана или не читается — «Ночь»."""
    try:
        import db
        df = db.load('app_state')
        if not df.empty:
            sub = df[df['key'].astype(str) == _STATE_KEY]
            if not sub.empty:
                name = str(sub.iloc[-1]['value'])
                if name in THEMES or name == 'win11':
                    return name
    except Exception:                                   # noqa: BLE001 — тема не должна ронять запуск
        pass
    return DEFAULT_THEME


def save_theme(name):
    """Запомнить тему (применится при следующем запуске)."""
    if name not in THEMES and name != 'win11':
        raise ValueError(f'Нет темы {name!r}')
    import pandas as pd
    import db
    df = db.load('app_state')
    if not df.empty:
        df = df[df['key'].astype(str) != _STATE_KEY]
    df = pd.concat([df, pd.DataFrame([{'key': _STATE_KEY, 'value': name}])], ignore_index=True)
    db.save('app_state', df)


# --------------------------------------------------------------------------- #
# Значки (SVG) для стрелок и галочек — QSS умеет брать их только из файлов
# --------------------------------------------------------------------------- #
_SVG = {
    'chevron_down': '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                    '<path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="$c" stroke-width="1.6" '
                    'stroke-linecap="round" stroke-linejoin="round"/></svg>',
    'chevron_up': '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                  '<path d="M2.5 7.5 6 4l3.5 3.5" fill="none" stroke="$c" stroke-width="1.6" '
                  'stroke-linecap="round" stroke-linejoin="round"/></svg>',
    'check': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
             '<path d="M3.5 8.4 6.6 11.4 12.5 4.8" fill="none" stroke="$c" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"/></svg>',
    'grip': '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="4" viewBox="0 0 32 4">'
            '<rect x="0" y="1" width="32" height="2" rx="1" fill="$c"/></svg>',
}


def _write_icons(name, t):
    """Записать значки темы во временный каталог; вернуть {имя: путь для url()}."""
    folder = os.path.join(tempfile.gettempdir(), 'nerual_rabbit_theme', name)
    os.makedirs(folder, exist_ok=True)
    colors = {'chevron_down': t['muted'], 'chevron_up': t['muted'], 'check': t['on_accent'],
              'grip': t['border_strong']}
    paths = {}
    for key, svg in _SVG.items():
        path = os.path.join(folder, key + '.svg')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(Template(svg).substitute(c=colors[key]))
        paths[key] = path.replace('\\', '/')
    return paths


# --------------------------------------------------------------------------- #
# Таблица стилей
# --------------------------------------------------------------------------- #
_QSS = Template("""
* { font-family: "Segoe UI"; font-size: 10pt; }
QWidget { color: $text; }
QMainWindow, QDialog { background: $bg; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

/* ---- вкладки ---- */
QTabWidget::pane { border: none; border-top: 1px solid $border; top: -1px; background: $bg; }
QTabBar { qproperty-drawBase: 0; background: transparent; }
QTabBar::tab { background: transparent; color: $muted; padding: 8px 14px 7px 14px; margin-right: 2px;
               border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: $text; background: $hover; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { color: $text; border-bottom: 2px solid $amber; }

/* ---- кнопки ---- */
QPushButton { background: $surface; border: 1px solid $border_strong; border-radius: 6px;
              padding: 4px 12px; min-height: 18px; }
QPushButton:hover { background: $hover; border-color: $accent; }
QPushButton:pressed { background: $pressed; }
QPushButton:disabled { color: $disabled; background: $surface_alt; border-color: $border; }
QPushButton:flat { border: none; background: transparent; }
QPushButton:flat:hover { background: $hover; }
QPushButton[role="primary"] { background: $accent; color: $on_accent; border: 1px solid $accent; font-weight: 600; }
QPushButton[role="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[role="primary"]:pressed { background: $accent_pressed; }
QPushButton[role="primary"]:disabled { background: $accent_soft; border-color: $accent_soft; color: $disabled; }
QPushButton[role="danger"] { color: $danger; border-color: $danger_border; background: $surface; }
QPushButton[role="danger"]:hover { background: $danger_soft; border-color: $danger; }

/* ---- поля ввода ---- */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox {
    background: $input; border: 1px solid $border_strong; border-radius: 6px; padding: 3px 8px;
    selection-background-color: $accent; selection-color: $on_accent; }
QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover { border-color: $muted; }
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {
    border: 1px solid $accent; }
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled { color: $disabled; background: $surface_alt; }
QComboBox { padding-right: 26px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 24px; border: none; }
QComboBox::down-arrow { image: url($chevron_down); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: $surface; border: 1px solid $border_strong; outline: 0; padding: 4px;
    selection-background-color: $accent_soft; selection-color: $text; }
QAbstractSpinBox { padding-right: 22px; }
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 20px;
    border: none; background: transparent; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 20px;
    border: none; background: transparent; }
QAbstractSpinBox::up-arrow { image: url($chevron_up); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url($chevron_down); width: 10px; height: 10px; }
QDateEdit::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 24px; border: none; }
QDateEdit::down-arrow { image: url($chevron_down); width: 12px; height: 12px; }

/* ---- галочки ---- */
QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator, QAbstractItemView::indicator { width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid $border_strong; background: $input; }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px; border: 1px solid $border_strong;
    background: $input; }
QCheckBox::indicator:hover, QAbstractItemView::indicator:hover, QRadioButton::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked, QAbstractItemView::indicator:checked { background: $accent; border-color: $accent;
    image: url($check); }
QRadioButton::indicator:checked { background: $accent; border: 4px solid $input; outline: none; }

/* ---- списки и таблицы ---- */
QListView, QTreeView, QTableView { background: $surface; alternate-background-color: $surface_alt;
    border: 1px solid $border; border-radius: 8px; gridline-color: $border; outline: 0;
    selection-background-color: $accent_soft; selection-color: $text; }
QListView::item { padding: 2px 6px; border-radius: 4px; }
QListView::item:hover { background: $hover; }
QListView::item:selected { background: $accent_soft; color: $text; }
QTableView::item { padding: 2px 6px; }
QHeaderView { background: $surface; border: none; }
QHeaderView::section { background: $surface_alt; color: $muted; padding: 6px 8px; border: none;
    border-bottom: 1px solid $border; border-right: 1px solid $border; font-weight: 600; }
QTableCornerButton::section { background: $surface_alt; border: none; border-bottom: 1px solid $border; }

/* ---- группы ---- */
QGroupBox { background: $surface; border: 1px solid $border; border-radius: 10px; margin-top: 16px;
    padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 4px;
    color: $text; font-weight: 600; }

/* ---- прокрутка и разделители ---- */
QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: $scroll; border-radius: 4px; min-height: 32px; }
QScrollBar::handle:horizontal { background: $scroll; border-radius: 4px; min-width: 32px; }
QScrollBar::handle:hover { background: $scroll_hover; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QSplitter::handle { background: $bg; }
QSplitter::handle:vertical { image: url($grip); }
QSplitter::handle:hover { background: $hover; }

/* ---- подписи по ролям: label.setProperty('role', ...) ---- */
QLabel[role="title"] { font-size: 16pt; font-weight: 600; color: $text; }
QLabel[role="heading"] { font-size: 13pt; font-weight: 600; color: $text; }
QLabel[role="muted"] { color: $muted; }
QLabel[role="hint"] { color: $muted; font-size: 9pt; }
QLabel[role="success"] { color: $accent; }
QLabel[role="avatar"] { background: $surface_alt; border: 1px solid $border_strong; border-radius: 12px;
    color: $muted; font-size: 36pt; font-weight: 600; }

/* ---- прочее ---- */
QAbstractScrollArea::corner { background: $bg; border: none; }
QWidget#plotCanvas { background: $surface; }
QToolTip { background: $tooltip_bg; color: $tooltip_fg; border: none; padding: 6px 8px; }
QMenu { background: $surface; border: 1px solid $border_strong; padding: 4px; }
QMenu::item { padding: 6px 14px; border-radius: 4px; }
QMenu::item:selected { background: $accent_soft; color: $text; }
QProgressBar { background: $surface_alt; border: 1px solid $border; border-radius: 6px; text-align: center;
    min-height: 14px; }
QProgressBar::chunk { background: $accent; border-radius: 5px; }
""")


def stylesheet(name):
    """QSS для темы ``name``."""
    t = THEMES[name]
    tokens = {k: v for k, v in t.items() if isinstance(v, str)}
    tokens.update(_write_icons(name, t))
    return _QSS.substitute(tokens)


def palette(name):
    """Палитра Fusion из токенов темы (для всего, что не покрыто QSS)."""
    t = THEMES[name]
    p = QPalette()
    c = lambda k: QColor(t[k])                                  # noqa: E731
    for group in (QPalette.Active, QPalette.Inactive):
        p.setColor(group, QPalette.Window, c('bg'))
        p.setColor(group, QPalette.WindowText, c('text'))
        p.setColor(group, QPalette.Base, c('surface'))
        p.setColor(group, QPalette.AlternateBase, c('surface_alt'))
        p.setColor(group, QPalette.Text, c('text'))
        p.setColor(group, QPalette.Button, c('surface'))
        p.setColor(group, QPalette.ButtonText, c('text'))
        p.setColor(group, QPalette.Highlight, c('accent'))
        p.setColor(group, QPalette.HighlightedText, c('on_accent'))
        p.setColor(group, QPalette.ToolTipBase, c('tooltip_bg'))
        p.setColor(group, QPalette.ToolTipText, c('tooltip_fg'))
        p.setColor(group, QPalette.PlaceholderText, c('muted'))
        p.setColor(group, QPalette.Link, c('accent'))
        p.setColor(group, QPalette.Accent, c('accent'))
        # оттенки рамок и теней: их берут palette(mid) в QSS и рамки стиля Fusion
        p.setColor(group, QPalette.Light, c('hover'))
        p.setColor(group, QPalette.Midlight, c('surface_alt'))
        p.setColor(group, QPalette.Mid, c('border_strong'))
        p.setColor(group, QPalette.Dark, c('border'))
        p.setColor(group, QPalette.Shadow, c('bg'))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c('disabled'))
    p.setColor(QPalette.Disabled, QPalette.Base, c('surface_alt'))
    p.setColor(QPalette.Disabled, QPalette.Button, c('surface_alt'))
    return p


def _apply_plots(t):
    """Графики matplotlib в тон окну (действует на фигуры, созданные после вызова)."""
    import matplotlib
    from cycler import cycler
    matplotlib.rcParams.update({
        'figure.facecolor': t['surface'], 'axes.facecolor': t['surface'],
        'savefig.facecolor': t['surface'],
        'axes.edgecolor': t['border_strong'], 'axes.labelcolor': t['muted'],
        'xtick.color': t['muted'], 'ytick.color': t['muted'], 'text.color': t['text'],
        'grid.color': t['border'], 'axes.grid': True, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'font.family': ['Segoe UI', 'DejaVu Sans'],
        'axes.prop_cycle': cycler(color=t['plot_cycle']),
    })


def cell_colors(name=None):
    """Цвета ячеек таблицы «Проверка урожайности» текущей темы + цвет текста в них.

    Вкладка спрашивает цвета в момент заливки таблицы, а не получает их разовым патчем
    модуля: при запуске ``python app_gui.py`` модуль вкладки называется ``__main__``,
    и патч по имени ``app_gui`` молча не срабатывал — ячейки оставались светлыми под
    тёмной темой, а текст на них терялся. Для родного стиля Windows (без своих токенов)
    берём светлый набор.
    """
    t = THEMES.get(name or current_theme()) or THEMES['field']
    return {'miss': QColor(t['cell_miss']), 'user': QColor(t['cell_user']),
            'edit': QColor(t['cell_edit']), 'plain': QColor(t['cell_plain']),
            'fg': QColor(t['text']), 'muted': QColor(t['muted'])}


def apply(app, name='field'):
    """Применить тему к приложению (вызывать ДО создания окон). ``name``: field | night | win11."""
    if name == 'win11':
        app.setStyle('windows11')
        p = app.palette()
        p.setColor(QPalette.Accent, QColor(WIN11['accent']))
        p.setColor(QPalette.Highlight, QColor(WIN11['accent']))
        app.setPalette(p)
        return
    t = THEMES[name]
    app.setStyle('Fusion')
    app.setPalette(palette(name))
    app.setStyleSheet(stylesheet(name))
    _apply_plots(t)


def mark_buttons(root):
    """Выставить роли кнопкам по тексту: главные действия — primary, удаление/выход — danger."""
    for btn in root.findChildren(QPushButton):
        text = _plain(btn.text())
        role = 'danger' if text in DANGER_TEXTS else 'primary' if text in PRIMARY_TEXTS else None
        if role and btn.property('role') != role:
            btn.setProperty('role', role)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
