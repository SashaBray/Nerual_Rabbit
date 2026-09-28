"""Оконное приложение (PySide6) для работы со всеми моделями прогноза урожайности.

Вкладки: личный кабинет, обзор данных (сравнение районов/регионов), создание прогноза
(батч по многим территориям, динамический список моделей, состав ансамбля), обзор моделей,
обзор прогнозов + Word-отчёты. Вход сохраняется до выхода. Запуск: python code/app_gui.py
"""
import datetime
import os
import sys
import traceback

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal, QSize, QDate
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QCheckBox, QLineEdit, QTextEdit, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QMessageBox, QDialog, QListWidget, QListWidgetItem, QFileDialog,
    QGroupBox, QGridLayout, QDateEdit, QInputDialog, QTextBrowser, QScrollArea, QFrame, QProgressBar,
    QSplitter)

import app_accounts as A
import app_theme                                   # оформление (тема «Ночь» и др.)
import app_geo as G
import app_predict as P
import app_reports as R
import app_stats as ST
import app_export as X
import app_preflight as PF
import app_store as S
import assets as AS
import app_yields as Y
import config
import db

YEAR_NOW = datetime.date.today().year
AGG = 'Регион целиком (обобщённо)'

def cell_colors():
    """Заливка клеток «Проверки урожайности» из текущей темы (нет данных / вручную / правка).

    Спрашиваем тему каждый раз при заливке: так цвета верны и после смены темы, и при
    запуске файла напрямую (`python app_gui.py`, модуль называется `__main__`).
    """
    return app_theme.cell_colors()

HELP_HTML = """
<h2>Прогноз урожайности — Nerual Rabbit</h2>
<p>Программа объединяет обученные модели (нейросеть, случайный лес, линейная,
классическая регрессия по NDVI и их ансамбль) и помогает строить прогнозы урожайности
по регионам и районам, формировать отчёты и отслеживать точность моделей.</p>

<h3>Личный кабинет</h3>
<ul>
<li>Имя, эл. почта, <b>ukey</b> (ключ доступа к Vega — нужен для докачки свежих данных), описание, пол,
дата рождения и аватар. Кнопка входа/профиля — справа сверху, с миниатюрой аватара.</li>
<li><b>ukey</b> используется при прогнозе для автозагрузки недостающих рядов из Vega.</li>
<li>Внизу — «Выйти из аккаунта».</li>
<li>Блок кабинета стоит по центру. <b>Ширину блока можно менять</b>, потянув за тонкие полоски
по его бокам; выбранная ширина запоминается и сохраняется между запусками. Двойной щелчок по полоске —
ширина по умолчанию.</li>
</ul>

<h3>1. Данные об урожайности</h3>
<ul>
<li>Выберите регион и культуру. Отметьте районы галочками (или «Регион целиком» — обобщённые данные).</li>
<li>«Добавить в сравнение» накапливает несколько территорий; «Построить график» рисует их вместе.</li>
</ul>

<h3>2. Проверка урожайности (перед прогнозом)</h3>
<ul>
<li>Показывает таблицу <b>территории × годы</b>: где урожайность по культуре уже есть, а где пропуск.
Выберите культуру, диапазон лет и регионы (галочками), нажмите <b>«Показать таблицу»</b>.</li>
<li>Цвет клетки: <b>белая</b> — значение из источника (ЕМИСС / архив БДПМО), <b>зелёная</b> — введено
вами вручную, <b>красная</b> — данных нет, <b>жёлтая</b> — изменено и ещё не сохранено.</li>
<li><b>Впишите недостающие значения прямо в клетки</b> (ц/га, запятая или точка) и нажмите
<b>«Сохранить введённое»</b>. Ваши значения хранятся в отдельной таблице БД <i>user_yields</i>:
они <b>не теряются при переподгрузке источников</b> и используются, когда сервисы-источники
недоступны. Читаются наравне с данными источников (ваше значение имеет приоритет над значением
источника за тот же год).</li>
<li><b>Где лежит таблица:</b> <i>workspace/database/user_yields.csv</i> в папке программы — файл
заводится при первом запуске, даже пока он пуст. Все таблицы базы записаны для Excel
(разделитель «;», запятая в дробных), поэтому открываются сразу по столбцам.</li>
<li><b>Где именно участвуют в прогнозе:</b> в <b>скользящей средней урожайности</b> — это входной
признак нейросети, случайного леса и линейной модели (среднее за последние m лет <i>с данными</i>
до расчётного года); в <b>классической регрессии по NDVI</b> — каждая урожайность даёт точку
регрессии по району (регрессия строится от 3 точек, так что дозаполнение истории включает
прогноз там, где его раньше не было); а также при <b>сборке датасетов</b> для обучения.
Годы без значения в расчёте не участвуют — окно берёт следующий по давности год с данными.</li>
<li><b>«Убрать моё значение»</b> удаляет ваши значения в выделенных клетках (данные источника
остаются нетронутыми — раздел их не портит). <b>«Выгрузить CSV»</b> сохраняет показанную таблицу
для просмотра.</li>
<li>Галочка «только территории с пропусками» оставляет в таблице лишь то, что нужно дозаполнить;
столбец «есть» показывает, сколько лет из диапазона заполнено.</li>
</ul>

<h3>3. Создать прогноз</h3>
<ul>
<li>Выберите культуру и год (или диапазон лет). Список доступных моделей зависит от культуры.</li>
<li>Отметьте регионы галочками. При выделении региона его районы <b>с данными</b> по культуре
отмечаются автоматически; лишние можно снять.</li>
<li><b>«Отмеченные регионы + их районы → задание»</b> — одной кнопкой добавляет в задание все
отмеченные регионы и все их районы с данными, кроме тех, с которых вы сняли галочку.</li>
<li>Кнопки «Регионы целиком →», «Районы →», «ВСЕ районы РФ» — другие способы наполнить задание.</li>
<li><b>Шаблоны</b>: задание можно сохранить как шаблон и потом загрузить из списка.</li>
<li><b>«Проверить урожайность по заданию»</b> открывает раздел 2 с культурой и регионами задания —
посмотреть, хватает ли данных, и дозаполнить пропуски руками.</li>
<li>«Состав ансамбля» задаёт, какие модели усредняются в ансамбль.</li>
<li>Нажмите «Сделать прогноз». <b>Сначала программа обследует базу</b> и покажет план: сколько
прогнозов посчитается сразу, сколько — только после докачки рядов с Vega, а сколько не посчитается
вовсе, и почему (нет модели, нет маски NDVI, мало лет урожайности, район не сопоставлен с Vega, год
ещё не наступил). Там же — какие ряды по параметрам и годам есть в базе, а каких нет, и оценка:
<i>«ожидается R из N; с ukey было бы до M»</i>. Прогноз запускается только после вашего согласия.</li>
<li><b>ukey не обязателен.</b> Без него прогноз считается по данным, которые уже есть в базе (например,
по скачанному из каталога полному снимку); с ukey недостающие ряды докачиваются с Vega.</li>
<li>Рядом показывается прогресс в процентах. Прогноз, уточнённые метрики районов и <b>итог по каждому
запланированному прогнозу с причиной нерасчёта</b> сохраняются в БД (одна партия = один запуск).</li>
</ul>

<h3>4. Модели</h3>
<ul>
<li>Арсенал моделей по культурам: какие сохранены и где. Нейросеть, случайный лес и линейная —
файлами в <i>models/&lt;задача&gt;_&lt;тип&gt;/</i>; классическая регрессия по NDVI считается на лету.</li>
<li><b>«Оценить точность»</b> для выбранной культуры запускает бэктест по выборке районов и
показывает <b>MSE, RMSE, R²</b> по каждой модели (значения сохраняются и видны в таблице).</li>
<li><b>«Обучить RF/линейную»</b> обучает и сохраняет случайный лес и линейную модель (v1) для
выбранной культуры — после этого они появляются в списке моделей на вкладке «Создать прогноз».</li>
<li><b>Комментарий к модели.</b> Столбец «комментарий» — ваша заметка о том, чем эта версия
отличается от соседних. Выберите строку и нажмите <b>«Комментарий к модели»</b> (или дважды
щёлкните по строке): в окне правки показаны параметры из метаданных модели — архитектура,
длительность рядов, окно урожайности, лучшая val-MSE, число эпох, дата. Заметка хранится
файлом <i>comment.txt</i> в папке самой модели, поэтому переезжает вместе с ней и попадает
в выгрузку прогноза. Столбец «создана» — дата файла весов.</li>
</ul>

<h3>5. Прогнозы и отчёты</h3>
<ul>
<li>Верхний список — партии прогнозов (кто и когда); столбец <b>«рассчитано»</b> показывает,
сколько прогнозов посчитано из запланированных. Выберите партию.</li>
<li>Вкладка <b>«План и итог»</b> внизу: сколько рассчитано, что было в плане, сводка причин нерасчёта
и построчно — почему не рассчитан каждый прогноз (например, «нет входных рядов в базе; ukey не указан»
или «Vega не отдала недостающие ряды»). У партий, сделанных до появления учёта, этих сведений нет.</li>
<li><b>«Сформировать отчёт по партии»</b> — спросит папку и имя, создаст CSV (разделитель «;»,
десятичная запятая) с подробной таблицей и Word, разбитый на главы по регионам (график + сводка).</li>
<li>«Глубина лет урожайности» задаёт, сколько последних лет урожайности показывать.</li>
<li><b>«Выгрузить прогноз (папка + ZIP)»</b> собирает прогноз в самостоятельную папку
одного стандартного вида — её можно отправить коллеге, и она читается без программы:
<i>README.txt</i> (что внутри и как читать), <i>meta.json</i> (метаданные: партия, автор,
модели с заметками, годы), <i>predictions.csv</i> и <i>predictions.xlsx</i> (сводная таблица),
<i>predictions_raw.csv</i> (строки прогнозов из БД), <i>territory_metrics.csv</i> (метрики
территорий), <i>models.csv</i> (какие модели и версии участвовали, с комментариями и точностью),
<i>yields_used.csv</i> (урожайности, на которых построен прогноз, с пометкой «ручной ввод»),
<i>report.docx</i> и папки <i>png/pdf</i> с графиками. Галочка <b>ZIP</b> дополнительно
складывает папку в архив рядом; галочка <b>«с Word-отчётом»</b> отключает самую долгую часть,
если нужны только таблицы. По умолчанию выгрузки лежат в <i>workspace/exports</i>.</li>
<li><b>«Удалить партию»</b> — безвозвратно удаляет все прогнозы и метрики выбранной партии.</li>
</ul>

<h3>6. Подгрузка данных</h3>
<ul>
<li>Отдельный раздел для загрузки свежих данных из внешних источников в БД. Источники-разделы
добавляются по мере появления парсеров.</li>
<li><b>ЕМИСС (fedstat.ru) — урожайность регионов.</b> Выберите культуру и отметьте регионы,
нажмите <b>«Подгрузить урожайность»</b> — программа тянет свежую урожайность с ЕМИСС и пишет её
в БД (уровень региона). Значение за год записывается только если оно реально есть у ЕМИСС (без
подмены последним известным). Запросы могут не пройти из-за блокировок по IP (в т.ч. IPv6) — тогда
данные не загрузятся, о чём будет сообщение; результат по каждому региону виден в таблице.</li>
<li><b>Архив урожайностей (tochno.st).</b> Скачайте набор БДПМО на https://tochno.st/ (файл
<i>data_bdmo_*.zip</i>), нажмите «Обзор…», укажите путь и нажмите <b>«Загрузить урожайности из
архива в БД»</b>. Программа сама извлечёт урожайность на убранную площадь и занесёт её в БД по
районам (категория «Хозяйства всех категорий», только известные приложению культуры). Перед записью
делается бэкап <i>yields</i>. Ход виден на шкале и в журнале.</li>
</ul>

<h3>7. Каталог</h3>
<ul>
<li>Готовые модели и снимки базы данных, выложенные автором в общий доступ. <b>Источник</b> —
публичная ссылка на папку (Яндекс.Диск) или локальная папка (раздача по сети или с флешки);
ссылка сохраняется и подставляется при следующих запусках.</li>
<li>В таблице видно: что это (модель / база данных), культура, версия, размер, ваш комментарий
к модели и <b>состояние</b> — «нет», «установлено» или «обновление». Отметьте строки и нажмите
<b>«Скачать / обновить выбранное»</b>: файлы качаются с докачкой при обрыве, сверяется
контрольная сумма, и только потом содержимое подменяется — прерванная загрузка ничего не портит.</li>
<li><b>Снимок базы заменяет только справочные таблицы</b> (территории, регионы, маски, урожайности
источников, ряды). Ваши данные — ручные урожайности, учётные записи, прогнозы, шаблоны — не
трогаются никогда; прежние файлы таблиц сохраняются в <i>workspace/legacy/db_backup_…</i>.</li>
<li>Снимки бывают двух видов: <b>lite</b> (справочники и урожайности, единицы МБ — разделы данных
и проверка урожайности работают сразу, для прогноза ряды докачиваются с Vega) и <b>full</b>
(плюс временные ряды — прогноз работает без обращения к Vega).</li>
<li>При запуске программа тихо проверяет каталог: если появились новые версии, у вкладки
появляется счётчик, например «7. Каталог (2)». Само ничего не скачивается.</li>
<li><b>Датасеты обучения</b> в каталоге нужны только тем, кто собирается переобучать модели —
для прогноза они не требуются. Они тяжёлые (от десятков МБ до гигабайта), поэтому качаются
только по явному выбору. Матрица едет сжатой в float32 и разворачивается в привычный
<i>matrix.csv</i> при установке — это самая долгая часть, примерно минута на 200 МБ.</li>
<li><b>«Удалить»</b> убирает установленную модель или датасет с диска (снимки базы так не
удаляются).</li>
</ul>

<h3>Подсказки</h3>
<ul>
<li>Если по району прогноз пустой — в отчёте есть колонка «примечание» с причиной
(нет истории урожайности по культуре / нет входных рядов за год).</li>
<li>Для прошлых лет данные обычно уже есть в БД; для текущего/будущего года включается
автозагрузка из Vega по вашему ukey.</li>
</ul>
"""
ICON_PATH = os.path.join(config.RESOURCES, 'logos', 'NR_logo_7.png')   # логотип (едет со сборкой)


def app_icon():
    return QIcon(ICON_PATH) if os.path.exists(ICON_PATH) else QIcon()


def df_to_table(tbl, df, maxrows=2000):
    df = df.head(maxrows) if df is not None else None
    if df is None or len(df) == 0:
        tbl.setRowCount(0); tbl.setColumnCount(0); return
    tbl.setColumnCount(len(df.columns)); tbl.setRowCount(len(df))
    tbl.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i in range(len(df)):
        for j, c in enumerate(df.columns):
            v = df.iloc[i, j]
            tbl.setItem(i, j, QTableWidgetItem('' if v is None or v != v else str(v)))
    tbl.resizeColumnsToContents()


def _add_check(lw, text, data, checked=False):
    it = QListWidgetItem(text); it.setData(Qt.UserRole, data)
    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
    it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
    lw.addItem(it)


def _checked(lw):
    return [lw.item(i).data(Qt.UserRole) for i in range(lw.count())
            if lw.item(i).checkState() == Qt.Checked]


def _set_all(lw, state):
    for i in range(lw.count()):
        lw.item(i).setCheckState(Qt.Checked if state else Qt.Unchecked)


def _note(text):
    """Поясняющий абзац: QLabel с переносом строк.

    Без переноса длинная строка задаёт вкладке минимальную ширину по своей длине —
    панель перестаёт помещаться в окно и появляется горизонтальная прокрутка.
    """
    lab = QLabel(text)
    lab.setWordWrap(True)
    return lab


def _status_label():
    """Строка состояния рядом с кнопкой: переносится, чтобы длинный текст не распирал панель."""
    lab = QLabel('')
    lab.setWordWrap(True)
    return lab


def _txt(v):
    """Значение поля как строка; NaN/None/'nan' -> '' (иначе из БД прилетает 'nan')."""
    if v is None or (isinstance(v, float) and v != v):
        return ''
    s = str(v)
    return '' if s.strip().lower() in ('nan', 'none') else s


class Worker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int)                                  # процент выполнения (0..100)
    message = Signal(str)                                   # текстовый комментарий (последняя операция)

    def __init__(self, fn, with_progress=False):
        super().__init__(); self._fn = fn; self._with_progress = with_progress

    def run(self):
        try:
            res = self._fn(self.progress.emit) if self._with_progress else self._fn()
            self.done.emit(res)
        except Exception:                                  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------- #
# Вход / первый запуск
# --------------------------------------------------------------------------- #
class WidthGrip(QWidget):
    """Ручка изменения ширины блока, стоящего по центру: тянуть наружу/внутрь, двойной щелчок — сброс.

    Блок центрирован, поэтому при сдвиге края на dx оба края расходятся симметрично и ширина
    меняется на 2·dx. Ручка лишь сообщает смещение курсора от точки нажатия (наружу — плюс для
    обеих сторон); ширину считает, ограничивает и сохраняет окно.
    """
    pressed = Signal()
    dragged = Signal(int)
    released = Signal()
    reset = Signal()

    WIDTH = 14

    def __init__(self, side, parent=None):
        super().__init__(parent)
        self.side = side                                   # 'left' | 'right'
        self._hover = self._drag = False
        self._x0 = 0.0
        self.setFixedWidth(self.WIDTH)
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip('Потяните, чтобы изменить ширину блока.\nДвойной щелчок — ширина по умолчанию.')

    def enterEvent(self, event):
        self._hover = True; self.update()

    def leaveEvent(self, event):
        self._hover = False; self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag = True
            self._x0 = event.globalPosition().x()
            self.pressed.emit(); self.update()

    def mouseMoveEvent(self, event):
        if self._drag:
            dx = int(event.globalPosition().x() - self._x0)
            self.dragged.emit(dx if self.side == 'right' else -dx)

    def mouseReleaseEvent(self, event):
        if self._drag and event.button() == Qt.LeftButton:
            self._drag = False
            self.released.emit(); self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset.emit()

    def paintEvent(self, event):
        """Тонкая скруглённая полоска по центру: в покое приглушённая, при наведении — акцентная."""
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainter, QPalette
        active = self._hover or self._drag
        color = self.palette().color(QPalette.Highlight if active else QPalette.PlaceholderText)
        if not active:
            color.setAlpha(110)
        h = min(56.0, max(self.height() - 16.0, 12.0))
        w = 4.0 if active else 3.0
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen); painter.setBrush(color)
        painter.drawRoundedRect(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), 2, 2)
        painter.end()


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Вход / первый запуск'); self.setMinimumWidth(440)
        self.setWindowIcon(app_icon())
        lay = QVBoxLayout(self)
        accts = A.list_accounts()
        if accts:
            lay.addWidget(QLabel('Выберите учётную запись или создайте новую:'))
            self.combo = QComboBox()
            for a in accts:
                self.combo.addItem(f'{a["name"]} <{a["email"]}>', a['account_id'])
            self.combo.addItem('— Создать новую —', None)
            self.combo.currentIndexChanged.connect(self._toggle); lay.addWidget(self.combo)
        else:
            self.combo = None
            lay.addWidget(QLabel('Первый запуск. Введите данные учётной записи:'))
        form = QFormLayout()
        self.name = QLineEdit(); self.email = QLineEdit(); self.ukey = QLineEdit()
        self.ukey.setPlaceholderText('необязательно — нужен для докачки рядов с Vega')
        form.addRow('Имя:', self.name); form.addRow('Эл. почта:', self.email); form.addRow('ukey:', self.ukey)
        self.box = QGroupBox('Новая учётная запись'); self.box.setLayout(form); lay.addWidget(self.box)
        btns = QHBoxLayout()
        ok = QPushButton('Войти'); ok.clicked.connect(self._accept)
        cancel = QPushButton('Отмена'); cancel.clicked.connect(self.reject)
        btns.addWidget(ok); btns.addWidget(cancel); lay.addLayout(btns)
        self._toggle()

    def _toggle(self):
        self.box.setVisible(self.combo is None or self.combo.currentData() is None)

    def _accept(self):
        try:
            if self.combo is not None and self.combo.currentData() is not None:
                A.login(self.combo.currentData())
            else:
                A.login(A.create_account(self.name.text(), self.email.text(), self.ukey.text()))
            self.accept()
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, 'Ошибка', str(exc))


class PreflightDialog(QDialog):
    """Перед прогнозом: что есть в базе, чего нет, и сколько прогнозов получится."""

    def __init__(self, parent, survey):
        super().__init__(parent)
        self.setWindowTitle('Перед прогнозом: обследование базы')
        self.setWindowIcon(app_icon())
        self.resize(900, 620)
        s = survey['summary']
        lay = QVBoxLayout(self)
        head = _note(PF.estimate_text(s).replace('\n', '<br>'))
        head.setTextFormat(Qt.RichText)
        lay.addWidget(head)
        if s['planned'] and s['expected'] == 0:
            warn = _note('<b>При текущих условиях не посчитается ни один прогноз.</b> '
                         'Смотрите причины ниже.')
            warn.setTextFormat(Qt.RichText)
            lay.addWidget(warn)

        tabs = QTabWidget()
        reasons = QTableWidget(); reasons.setEditTriggers(QTableWidget.NoEditTriggers)
        df_to_table(reasons, survey['reasons'])
        data = QTableWidget(); data.setEditTriggers(QTableWidget.NoEditTriggers)
        df_to_table(data, survey['data'])
        items = QTableWidget(); items.setEditTriggers(QTableWidget.NoEditTriggers)
        view = survey['items'].assign(статус=survey['items']['status'].map(PF.STATUS_TITLES))[
            ['label', 'year', 'модель', 'статус', 'reason']].rename(
            columns={'label': 'территория', 'year': 'год', 'reason': 'подробности'})
        df_to_table(items, view, maxrows=5000)
        tabs.addTab(reasons, f'Почему не всё посчитается ({s["download"] + s["blocked"]})')
        tabs.addTab(data, 'Данные в базе: есть / нет')
        tabs.addTab(items, f'Все прогнозы плана ({s["planned"]})')
        lay.addWidget(tabs, 1)

        btns = QHBoxLayout(); btns.addStretch(1)
        run = QPushButton('Запустить прогноз'); run.clicked.connect(self.accept)
        run.setProperty('role', 'primary')
        cancel = QPushButton('Отмена'); cancel.clicked.connect(self.reject)
        btns.addWidget(run); btns.addWidget(cancel)
        lay.addLayout(btns)


# --------------------------------------------------------------------------- #
# Главное окно
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.acc = A.current_account()
        self.setWindowTitle('Прогноз урожайности — Nerual Rabbit'); self.resize(1150, 760)
        self.setWindowIcon(app_icon())
        self._worker = None
        top = QWidget(); self.setCentralWidget(top); v = QVBoxLayout(top)
        bar = QHBoxLayout(); bar.addStretch(1)             # вход/профиль — справа сверху
        self.account_btn = QPushButton(); self.account_btn.setFlat(True)
        self.account_btn.setCursor(Qt.PointingHandCursor)
        self.account_btn.setToolTip('Открыть личный кабинет')
        self.account_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.account_btn.setIconSize(QSize(28, 28))
        bar.addWidget(self.account_btn)
        v.addLayout(bar)
        self.tabs = QTabWidget(); v.addWidget(self.tabs)
        self.tabs.addTab(self._scroll(self._tab_profile()), 'Личный кабинет')
        self.tabs.addTab(self._tab_data(), '1. Данные об урожайности')            # уже сплиттер
        self._yields_tab_index = self.tabs.addTab(
            self._tab_yields(), '2. Проверка урожайности')                        # уже сплиттер
        self.tabs.addTab(self._tab_forecast(), '3. Создать прогноз')              # уже сплиттер
        self.tabs.addTab(self._scroll(self._tab_models()), '4. Модели')
        self.tabs.addTab(self._tab_predictions(), '5. Прогнозы и отчёты')         # уже сплиттер
        self.tabs.addTab(self._tab_dataload(), '6. Подгрузка данных')             # уже сплиттер
        self._catalog_tab_index = self.tabs.addTab(self._tab_catalog(), '7. Каталог')  # уже сплиттер
        self.tabs.addTab(self._scroll(self._tab_help()), 'Справка')
        self._refresh_account_btn()
        self._check_catalog_quietly()      # тихо: только отметка на вкладке, ничего не качаем

    @staticmethod
    def _scroll(widget):
        """Обернуть вкладку в прокручиваемую область — окно можно делать уже, контент не распирает его."""
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QFrame.NoFrame)
        sa.setWidget(widget)
        return sa

    @staticmethod
    def _panel(layout):
        """Собрать layout в отдельный виджет-панель (для помещения в QSplitter)."""
        holder = QWidget(); holder.setLayout(layout)
        return holder

    @classmethod
    def _vsplit(cls, panes, sizes=None, scroll_first=True):
        """Вертикальный сплиттер с перетаскиваемыми границами между «модулями».

        panes — список (QWidget | QLayout). Первый модуль (панель управления) при
        scroll_first оборачивается в прокрутку, чтобы на маленьком окне не распирал
        сплиттер, а нижние модули (журнал/таблица) можно было растягивать мышью.
        """
        sp = QSplitter(Qt.Vertical)
        sp.setChildrenCollapsible(False)                   # модуль нельзя схлопнуть в ноль
        sp.setHandleWidth(8)                               # широкая полоса — легче ухватить мышью
        for i, p in enumerate(panes):
            wdg = p if isinstance(p, QWidget) else cls._panel(p)
            sp.addWidget(cls._scroll(wdg) if (i == 0 and scroll_first) else wdg)
        if sizes:
            sp.setSizes(sizes)
        return sp

    def _refresh_account_btn(self):
        self.account_btn.setText('  ' + (self.acc.get('name') or 'аккаунт') + '  ▾')
        p = self.acc.get('avatar')
        if p and os.path.exists(str(p)):
            self.account_btn.setIcon(QIcon(QPixmap(str(p)).scaled(
                28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        else:
            self.account_btn.setIcon(QIcon())

    def _region_combo(self, none_option=False):
        c = QComboBox()
        if none_option:
            c.addItem('—', None)
        for rid, name in G.list_regions():
            c.addItem(f'{name} ({rid})', rid)
        return c

    def _fill_districts(self, lw, id_region, with_aggregate=True):
        lw.clear()
        if with_aggregate:
            _add_check(lw, AGG, None)
        if id_region is not None:
            for did, name in G.list_districts(id_region):
                _add_check(lw, f'{name} ({did})', did)

    # ============================ Личный кабинет (#1) ============================ #
    PROFILE_DEFAULT_WIDTH = 880                            # ширина блока по умолчанию (px)
    PROFILE_MIN_WIDTH = 600                                # уже — поля и шапка начинают тесниться
    PROFILE_WIDTH_KEY = 'profile_block_width'              # app_state: ширина, выбранная пользователем
    PROFILE_PAGE_MARGIN = 24
    PROFILE_GRIP_GAP = 6
    AVATAR_SIZE = 180                                      # как было до редизайна

    def _tab_profile(self):
        """Личный кабинет: блок по центру страницы, ширину меняют ручки по бокам блока."""
        page = QWidget(); page_v = QVBoxLayout(page)
        m = self.PROFILE_PAGE_MARGIN
        page_v.setContentsMargins(m, 20, m, 20)
        self.pf_page = page
        # [пусто | ручка | блок | ручка | пусто]: пустые поля делят остаток поровну -> блок по центру
        self.pf_wrap = QWidget(); wrap = QHBoxLayout(self.pf_wrap)
        wrap.setContentsMargins(0, 0, 0, 0); wrap.setSpacing(self.PROFILE_GRIP_GAP)
        content = QWidget()
        lay = QVBoxLayout(content); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(14)
        grip_l, grip_r = WidthGrip('left'), WidthGrip('right')
        wrap.addWidget(grip_l); wrap.addWidget(content, 1); wrap.addWidget(grip_r)
        row = QHBoxLayout(); row.setSpacing(0)
        row.addStretch(1); row.addWidget(self.pf_wrap, 1000); row.addStretch(1)
        page_v.addLayout(row); page_v.addStretch(1)
        for grip in (grip_l, grip_r):
            grip.pressed.connect(self._profile_width_drag_start)
            grip.dragged.connect(lambda dx: self._set_profile_width(self._pf_width_at_press + 2 * dx))
            grip.released.connect(self._save_profile_width)
            grip.reset.connect(self._reset_profile_width)

        # --- карточка профиля: шапка с аватаром + поля ---
        card = QGroupBox('Профиль'); cl = QVBoxLayout(card); cl.setSpacing(12)
        head = QHBoxLayout(); head.setSpacing(16)
        av_col = QVBoxLayout(); av_col.setSpacing(6)
        self.pf_avatar = QLabel(); self.pf_avatar.setProperty('role', 'avatar')
        self.pf_avatar.setFixedSize(self.AVATAR_SIZE, self.AVATAR_SIZE)
        self.pf_avatar.setAlignment(Qt.AlignCenter)
        pick = QPushButton('Сменить фото…'); pick.clicked.connect(self._pick_avatar)
        pick.setMinimumWidth(self.AVATAR_SIZE)
        av_col.addWidget(self.pf_avatar, 0, Qt.AlignHCenter)
        av_col.addWidget(pick, 0, Qt.AlignHCenter); av_col.addStretch(1)
        head.addLayout(av_col)
        who = QVBoxLayout(); who.setSpacing(2)
        self.pf_head_name = QLabel(); self.pf_head_name.setProperty('role', 'heading')
        self.pf_head_email = QLabel(); self.pf_head_email.setProperty('role', 'muted')
        self.pf_head_since = QLabel(); self.pf_head_since.setProperty('role', 'hint')
        who.addWidget(self.pf_head_name); who.addWidget(self.pf_head_email)
        who.addWidget(self.pf_head_since); who.addStretch(1)
        head.addLayout(who, 1)
        cl.addLayout(head)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet('color: palette(mid);')
        cl.addWidget(line)

        form = QFormLayout(); form.setHorizontalSpacing(14); form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pf_name = QLineEdit(_txt(self.acc.get('name')))
        self.pf_email = QLineEdit(_txt(self.acc.get('email')))
        self.pf_gender = QComboBox(); self.pf_gender.addItems(['Мужской', 'Женский', 'Другой'])
        self.pf_gender.setPlaceholderText('не указан')             # пусто — подсказка, а не пустая строка
        self.pf_gender.setCurrentIndex(self.pf_gender.findText(_txt(self.acc.get('gender'))))
        self.pf_gender.setMinimumWidth(160)
        self.pf_birth = QDateEdit(); self.pf_birth.setCalendarPopup(True)
        self.pf_birth.setDisplayFormat('dd.MM.yyyy'); self.pf_birth.setMinimumWidth(140)
        self.pf_birth.setDateRange(QDate(1900, 1, 1), QDate.currentDate())
        bd = _txt(self.acc.get('birthdate'))
        qd = QDate.fromString(bd, 'yyyy-MM-dd')
        self.pf_birth.setDate(qd if qd.isValid() else QDate(2000, 1, 1))
        pair = QHBoxLayout(); pair.setSpacing(10)                  # короткие значения — в одну строку
        pair.addWidget(self.pf_gender)
        bl = QLabel('Дата рождения'); bl.setProperty('role', 'muted')
        pair.addSpacing(8); pair.addWidget(bl); pair.addWidget(self.pf_birth); pair.addStretch(1)
        self.pf_desc = QTextEdit(_txt(self.acc.get('description'))); self.pf_desc.setFixedHeight(84)
        self.pf_desc.setPlaceholderText('Организация, должность — по желанию')
        form.addRow('Имя', self.pf_name)
        form.addRow('Эл. почта', self.pf_email)
        form.addRow('Пол', pair)
        desc_label = QLabel('О себе'); desc_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        desc_label.setContentsMargins(0, 6, 0, 0)                 # метка по первой строке, а не по центру поля
        form.addRow(desc_label, self.pf_desc)
        cl.addLayout(form)
        lay.addWidget(card)

        # --- доступ к Vega: ключ скрыт, показывается по кнопке ---
        vega = QGroupBox('Доступ к Vega'); vl = QVBoxLayout(vega); vl.setSpacing(8)
        krow = QHBoxLayout(); krow.setSpacing(8)
        klabel = QLabel('ukey'); klabel.setMinimumWidth(60)
        self.pf_ukey = QLineEdit(_txt(self.acc.get('ukey')))
        self.pf_ukey.setEchoMode(QLineEdit.Password)
        self.pf_ukey.setPlaceholderText('не указан')
        self.pf_ukey_show = QPushButton('Показать'); self.pf_ukey_show.setCheckable(True)
        self.pf_ukey_show.setFixedWidth(96)
        self.pf_ukey_show.toggled.connect(self._toggle_ukey)
        krow.addWidget(klabel); krow.addWidget(self.pf_ukey, 1); krow.addWidget(self.pf_ukey_show)
        self.pf_ukey.textChanged.connect(lambda t: self.pf_ukey_show.setEnabled(bool(t)))
        self.pf_ukey_show.setEnabled(bool(self.pf_ukey.text()))   # показывать нечего — кнопка неактивна
        vl.addLayout(krow)
        khint = _note('Необязателен. Без ключа прогноз считается по данным базы, '
                      'с ключом недостающие ряды докачиваются с Vega.')
        khint.setProperty('role', 'hint')
        vl.addWidget(khint)
        lay.addWidget(vega)

        # --- действия: выход слева, сохранение справа, отметка о сохранении без окна ---
        actions = QHBoxLayout(); actions.setSpacing(10)
        logout = QPushButton('Выйти из аккаунта'); logout.setProperty('role', 'danger')
        logout.clicked.connect(self._logout)
        self.pf_saved = QLabel(''); self.pf_saved.setProperty('role', 'success')
        save = QPushButton('Сохранить'); save.setProperty('role', 'primary')
        save.setMinimumWidth(140); save.clicked.connect(self._save_profile)
        actions.addWidget(logout); actions.addStretch(1)
        actions.addWidget(self.pf_saved); actions.addWidget(save)
        lay.addLayout(actions)

        for edit in (self.pf_name, self.pf_email, self.pf_ukey):   # правка снимает отметку «сохранено»
            edit.textEdited.connect(lambda *_: self.pf_saved.setText(''))
        self._show_avatar()
        self._refresh_profile_head()
        self._pf_width = self.PROFILE_DEFAULT_WIDTH
        self._pf_width_at_press = self._pf_width
        self._set_profile_width(self._load_profile_width())
        return page

    # ---- ширина блока «Личного кабинета» ----
    def _load_profile_width(self):
        """Ширина, сохранённая пользователем (в app_state), либо ширина по умолчанию."""
        try:
            return int(float(db.get_state(self.PROFILE_WIDTH_KEY) or self.PROFILE_DEFAULT_WIDTH))
        except (TypeError, ValueError):
            return self.PROFILE_DEFAULT_WIDTH

    def _profile_width_room(self):
        """Сколько места под блок сейчас есть на странице (или None, пока страница не разложена)."""
        page_w = self.pf_page.width()
        if page_w <= 0 or not self.pf_page.isVisible():   # до показа у виджета размер-заглушка (640 px)
            return None
        return page_w - 2 * self.PROFILE_PAGE_MARGIN - 2 * (WidthGrip.WIDTH + self.PROFILE_GRIP_GAP)

    def _set_profile_width(self, width):
        """Задать ширину блока: не уже минимума и не шире страницы; блок остаётся по центру.

        Задаётся МАКСИМАЛЬНАЯ ширина: на узком окне раскладка сама ужмёт блок, а при
        расширении окна он вернётся к выбранной ширине.
        """
        room = self._profile_width_room()
        upper = width if room is None else max(self.PROFILE_MIN_WIDTH, room)
        width = int(max(self.PROFILE_MIN_WIDTH, min(width, upper)))
        self._pf_width = width
        self.pf_wrap.setMaximumWidth(width + 2 * (WidthGrip.WIDTH + self.PROFILE_GRIP_GAP))

    def _profile_width_drag_start(self):
        # тянуть начинаем от того, что видно на экране: на узком окне блок уже сохранённой ширины
        grips = 2 * (WidthGrip.WIDTH + self.PROFILE_GRIP_GAP)
        self._pf_width_at_press = min(self._pf_width, max(self.PROFILE_MIN_WIDTH, self.pf_wrap.width() - grips))

    def _save_profile_width(self):
        db.set_state(self.PROFILE_WIDTH_KEY, self._pf_width)

    def _reset_profile_width(self):
        self._set_profile_width(self.PROFILE_DEFAULT_WIDTH)
        self._save_profile_width()

    def _toggle_ukey(self, shown):
        self.pf_ukey.setEchoMode(QLineEdit.Normal if shown else QLineEdit.Password)
        self.pf_ukey_show.setText('Скрыть' if shown else 'Показать')

    def _refresh_profile_head(self):
        """Шапка карточки: имя, почта и с какого времени есть учётная запись."""
        self.pf_head_name.setText(_txt(self.acc.get('name')) or 'Без имени')
        self.pf_head_email.setText(_txt(self.acc.get('email')))
        created = _txt(self.acc.get('created_at'))[:10]
        qd = QDate.fromString(created, 'yyyy-MM-dd')
        self.pf_head_since.setText(f'Учётная запись с {qd.toString("dd.MM.yyyy")}' if qd.isValid() else '')

    def _show_avatar(self):
        """Фото — квадратом по центру рамки (обрезка по короткой стороне); без фото — инициалы."""
        size = self.AVATAR_SIZE
        p = self.acc.get('avatar')
        if p and os.path.exists(str(p)):
            from PySide6.QtGui import QPainter, QPainterPath
            src = QPixmap(str(p)).scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x, y = (src.width() - size) // 2, (src.height() - size) // 2
            src = src.copy(x, y, size, size)
            out = QPixmap(size, size); out.fill(Qt.transparent)
            painter = QPainter(out); painter.setRenderHint(QPainter.Antialiasing)
            clip = QPainterPath(); clip.addRoundedRect(0, 0, size, size, 12, 12)   # радиус как у рамки
            painter.setClipPath(clip); painter.drawPixmap(0, 0, src); painter.end()
            self.pf_avatar.setPixmap(out)
        else:
            name = _txt(self.acc.get('name')).split()
            initials = ''.join(part[0] for part in name[:2]).upper() or '?'
            self.pf_avatar.setPixmap(QPixmap()); self.pf_avatar.setText(initials)

    def _pick_avatar(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Выбрать фото', '', 'Изображения (*.png *.jpg *.jpeg *.bmp)')
        if path:
            A.set_avatar(self.acc['account_id'], path)
            self.acc = A.current_account(); self._show_avatar(); self._refresh_account_btn()

    def _save_profile(self):
        try:
            A.update_account(self.acc['account_id'], name=self.pf_name.text(), email=self.pf_email.text(),
                             ukey=self.pf_ukey.text(), description=self.pf_desc.toPlainText(),
                             gender=self.pf_gender.currentText(),
                             birthdate=self.pf_birth.date().toString('yyyy-MM-dd'))
            self.acc = A.current_account(); self._refresh_account_btn(); self._refresh_profile_head()
            self._show_avatar()                            # инициалы могли смениться вместе с именем
            self.pf_saved.setText('Сохранено в ' + datetime.datetime.now().strftime('%H:%M'))
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, 'Ошибка', str(exc))

    # ============================ Данные (#2,#3) ============================ #
    def _tab_data(self):
        w = QWidget(); lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.d_reg = self._region_combo()
        self.d_reg.currentIndexChanged.connect(lambda *_: self._fill_districts(self.d_dist, self.d_reg.currentData()))
        self.d_cult = QComboBox(); self.d_cult.addItems(P.list_cultures())
        top.addWidget(QLabel('Регион:')); top.addWidget(self.d_reg)
        top.addWidget(QLabel('Культура:')); top.addWidget(self.d_cult); top.addStretch(1)
        lay.addLayout(top)
        mid = QHBoxLayout()
        self.d_dist = QListWidget(); self.d_dist.setMaximumHeight(150)
        self._fill_districts(self.d_dist, self.d_reg.currentData())
        dbox = QVBoxLayout(); dbox.addWidget(QLabel('Районы (галочками; «Регион целиком» — обобщённые данные):'))
        dbox.addWidget(self.d_dist)
        dbtn = QHBoxLayout()
        allc = QPushButton('Все районы'); allc.clicked.connect(lambda: _set_all(self.d_dist, True))
        nonec = QPushButton('Снять все'); nonec.clicked.connect(lambda: _set_all(self.d_dist, False))
        addc = QPushButton('Добавить в сравнение →'); addc.clicked.connect(self._cmp_add)
        dbtn.addWidget(allc); dbtn.addWidget(nonec); dbtn.addWidget(addc); dbox.addLayout(dbtn)
        mid.addLayout(dbox, 1)
        cbox = QVBoxLayout(); cbox.addWidget(QLabel('Сравнение (несколько районов/регионов):'))
        self.d_cmp = QListWidget(); self.d_cmp.setMaximumHeight(150); cbox.addWidget(self.d_cmp)
        cbtn = QHBoxLayout()
        clr = QPushButton('Очистить'); clr.clicked.connect(lambda: (self._cmp.clear(), self.d_cmp.clear()))
        show = QPushButton('Построить график'); show.clicked.connect(self._show_data)
        cbtn.addWidget(clr); cbtn.addWidget(show); cbox.addLayout(cbtn)
        mid.addLayout(cbox, 1)
        lay.addLayout(mid)
        lay.addStretch(1)                                  # управление прижато кверху внутри своего модуля
        self._cmp = []                                     # [(reg, dist|None, label)]
        # нижние модули с перетаскиваемыми границами (тяните полосу между ними мышью)
        self.d_table = QTableWidget(); self.d_table.setMinimumHeight(80)
        self.d_canvas = FigureCanvas(Figure(figsize=(6.5, 2.9))); self.d_canvas.setMinimumHeight(150)
        self.d_canvas.setObjectName('plotCanvas')                  # фон под графиком — из темы
        self.d_canvas.setAttribute(Qt.WA_StyledBackground, True)
        return self._vsplit([w, self.d_table, self.d_canvas], sizes=[300, 170, 300])

    def _cmp_add(self):
        reg = self.d_reg.currentData()
        if reg is None:
            return
        rname = self.d_reg.currentText()
        for did in _checked(self.d_dist):
            label = f'{rname} — регион' if did is None else f'{rname} / {self._dist_name(self.d_dist, did)}'
            item = (reg, did, label)
            if item not in self._cmp:
                self._cmp.append(item); self.d_cmp.addItem(label)

    def _dist_name(self, lw, did):
        for i in range(lw.count()):
            if lw.item(i).data(Qt.UserRole) == did:
                return lw.item(i).text().rsplit(' (', 1)[0]
        return str(did)

    def _show_data(self):
        import pandas as pd
        cult = self.d_cult.currentText(); cfg = P.resolve_culture(cult)
        targets = list(self._cmp) or [(self.d_reg.currentData(), d, self._dist_name(self.d_dist, d))
                                      for d in _checked(self.d_dist)]
        fig = self.d_canvas.figure; fig.clear(); ax = fig.add_subplot(111)
        rows = []; plotted = 0
        for reg, did, label in targets:
            if reg is None:
                continue
            tid = db.territory_id(reg, did)
            yields = db.get_yields(tid, cfg['bdpmo_culture'])
            if not yields:
                continue
            xs = sorted(yields); ys = [yields[y] for y in xs]
            ax.plot(xs, ys, '-o', ms=3, label=label); plotted += 1
            for y in xs:
                rows.append({'территория': label, 'год': y, 'урожайность': yields[y]})
        if plotted:
            ax.set_xlabel('год'); ax.set_ylabel('ц/га'); ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7); ax.set_title(f'{cult}: история урожайности')
        else:
            ax.text(0.5, 0.5, 'нет данных по выбранным территориям', ha='center')
        fig.tight_layout(); self.d_canvas.draw()
        df_to_table(self.d_table, pd.DataFrame(rows))

    # ==================== Урожайность: проверка и ручной ввод ==================== #
    def _tab_yields(self):
        """Что известно об урожайности культуры по территориям и годам + ручной ввод пропусков."""
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(_note(
            'Проверка данных ПЕРЕД прогнозом: какие урожайности уже есть по культуре в выбранных '
            'территориях и чего не хватает. Пустые клетки можно заполнить руками — ваши значения '
            'хранятся в отдельной таблице БД (user_yields), переживают переподгрузку источников '
            'и используются, когда сервисы-источники недоступны.'))
        top = QHBoxLayout()
        self.y_cult = QComboBox(); self.y_cult.addItems(P.list_cultures())
        self.y_y1 = QSpinBox(); self.y_y1.setRange(1990, YEAR_NOW + 1); self.y_y1.setValue(max(1990, YEAR_NOW - 10))
        self.y_y2 = QSpinBox(); self.y_y2.setRange(1990, YEAR_NOW + 1); self.y_y2.setValue(YEAR_NOW)
        top.addWidget(QLabel('Культура:')); top.addWidget(self.y_cult)
        top.addSpacing(16); top.addWidget(QLabel('Годы с:')); top.addWidget(self.y_y1)
        top.addWidget(QLabel('по:')); top.addWidget(self.y_y2)
        self.y_incl = QCheckBox('включая районы'); self.y_incl.setChecked(True)
        self.y_gaps = QCheckBox('только территории с пропусками')
        top.addSpacing(16); top.addWidget(self.y_incl); top.addWidget(self.y_gaps)
        top.addStretch(1); lay.addLayout(top)

        lay.addWidget(QLabel('Регионы (галочками):'))
        self.y_regs = QListWidget(); self.y_regs.setMaximumHeight(170)
        for rid, name in G.list_regions():
            _add_check(self.y_regs, f'{name} ({rid})', rid)
        lay.addWidget(self.y_regs)
        rb = QHBoxLayout()
        allb = QPushButton('Все'); allb.clicked.connect(lambda: _set_all(self.y_regs, True))
        noneb = QPushButton('Снять'); noneb.clicked.connect(lambda: _set_all(self.y_regs, False))
        showb = QPushButton('Показать таблицу'); showb.clicked.connect(self._y_show)
        rb.addWidget(allb); rb.addWidget(noneb); rb.addWidget(showb); rb.addStretch(1)
        lay.addLayout(rb)

        eb = QHBoxLayout()
        saveb = QPushButton('Сохранить введённое')
        saveb.setToolTip('Записать вписанные вручную значения в таблицу user_yields')
        saveb.clicked.connect(self._y_save)
        delb = QPushButton('Убрать моё значение')
        delb.setToolTip('Удалить свои значения в выделенных клетках (значения из источника остаются)')
        delb.clicked.connect(self._y_clear_selected)
        expb = QPushButton('Выгрузить CSV'); expb.clicked.connect(self._y_export)
        eb.addWidget(saveb); eb.addWidget(delb); eb.addWidget(expb)
        self.y_status = _status_label(); eb.addWidget(self.y_status); eb.addStretch(1)
        lay.addLayout(eb)
        c = cell_colors(); fg = c['fg'].name()
        swatch = lambda bg, txt: (f'<span style="background:{bg.name()}; color:{fg}">'
                                  f'&nbsp;{txt}&nbsp;</span>')
        lay.addWidget(_note(
            '<b>Обозначения:</b> '
            + swatch(c['plain'], '12,3') + ' из источника &nbsp;·&nbsp; '
            + swatch(c['user'], '12,3') + ' введено вручную &nbsp;·&nbsp; '
            + swatch(c['miss'], '&nbsp;&nbsp;&nbsp;') + ' нет данных &nbsp;·&nbsp; '
            + swatch(c['edit'], '12,3') + ' изменено, не сохранено'))
        lay.addStretch(1)                                  # управление прижато кверху внутри своего модуля

        self._y_av = None                                  # последний свод (availability)
        self._y_edits = {}                                 # {(territory_id, year): значение|None} — не сохранено
        self._y_filling = False                            # подавляем itemChanged при программном заполнении
        self.y_table = QTableWidget(); self.y_table.setMinimumHeight(160)
        self.y_table.itemChanged.connect(self._y_item_changed)
        return self._vsplit([w, self.y_table], sizes=[330, 380])

    def _y_show(self):
        """Построить таблицу «территории × годы» по отмеченным регионам и культуре."""
        regs = _checked(self.y_regs)
        if not regs:
            QMessageBox.information(self, 'Урожайность', 'Отметьте галочками хотя бы один регион.'); return
        n_years = abs(self.y_y2.value() - self.y_y1.value()) + 1
        if self.y_incl.isChecked() and len(regs) * 30 * n_years > 80000:
            if QMessageBox.question(self, 'Большая таблица',
                                    f'Выбрано регионов: {len(regs)}, лет: {n_years}. Таблица будет очень '
                                    f'большой и построится не сразу. Продолжить?') != QMessageBox.Yes:
                return
        try:
            av = Y.availability(regs, self.y_cult.currentText(), self.y_y1.value(), self.y_y2.value(),
                                include_districts=self.y_incl.isChecked(),
                                only_gaps=self.y_gaps.isChecked())
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, 'Урожайность', f'Не удалось собрать таблицу: {exc}'); return
        self._y_av = av; self._y_edits = {}
        self._y_fill_table(av)

    def _y_fill_table(self, av):
        years = av['years']; rows = av['rows']; t = self.y_table
        self._cellc = cell_colors()                        # цвета темы — на весь проход заливки
        self._y_filling = True
        t.clear(); t.setRowCount(len(rows)); t.setColumnCount(len(years) + 2)
        t.setHorizontalHeaderLabels(['Территория', 'есть'] + [str(y) for y in years])
        ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable                # столбцы-подписи не редактируются
        for i, r in enumerate(rows):
            name = QTableWidgetItem(r['label']); name.setFlags(ro)
            name.setToolTip(f"territory_id={r['territory_id']}")
            t.setItem(i, 0, name)
            cnt = QTableWidgetItem(f"{r['filled']}/{len(years)}"); cnt.setFlags(ro)
            t.setItem(i, 1, cnt)
            for j, y in enumerate(years):
                hit = av['values'].get((r['territory_id'], y))
                it = QTableWidgetItem('' if hit is None else f'{hit[0]:g}')
                it.setData(Qt.UserRole, (r['territory_id'], y))
                self._y_paint(it, hit)
                t.setItem(i, j + 2, it)
        t.resizeColumnsToContents()
        self._y_filling = False
        s = av['stats']
        pct = 100 * s['filled'] // max(s['cells'], 1)
        self.y_status.setText(f"территорий {len(rows)}, лет {len(years)}: заполнено {s['filled']} из "
                              f"{s['cells']} ({pct}%), из них вручную {s['user']}; пропусков {s['gaps']}")

    def _y_paint(self, item, hit):
        """Закрасить клетку по происхождению значения (нет данных / вручную / из источника).

        Цвет текста задаётся явно: на собственной заливке клетка не наследует цвет темы,
        и под «Ночью» цифры сливались с фоном.
        """
        c = getattr(self, '_cellc', None) or cell_colors()
        item.setForeground(c['fg'])
        if hit is None:
            item.setBackground(c['miss']); item.setToolTip('нет данных — можно вписать значение')
        elif hit[1] == Y.SRC_USER:
            item.setBackground(c['user']); item.setToolTip('введено вручную (таблица user_yields)')
        else:
            item.setBackground(c['plain']); item.setToolTip('значение из источника (ЕМИСС / архив БДПМО)')

    def _y_item_changed(self, item):
        """Проверить введённое, запомнить правку и подсветить клетку как несохранённую."""
        if self._y_filling or self._y_av is None:
            return
        key = item.data(Qt.UserRole)
        if key is None:
            return
        hit = self._y_av['values'].get(key)
        old = None if hit is None else hit[0]
        try:
            val = Y.parse_value(item.text())
        except ValueError:
            QMessageBox.warning(self, 'Значение', f'«{item.text()}» — не число. Введите урожайность '
                                                  f'(ц/га) или очистите клетку.')
            self._y_set_text(item, old); return
        if val is None and hit is not None and hit[1] == Y.SRC_SOURCE:
            QMessageBox.information(self, 'Значение из источника',
                                    'Это значение пришло из источника (ЕМИСС / архив БДПМО) — удалить его '
                                    'в этом разделе нельзя. Можно вписать своё: оно будет использоваться '
                                    'вместо значения источника.')
            self._y_set_text(item, old); return
        if val == old:                                     # вернули как было — правки нет
            self._y_edits.pop(key, None)
            self._y_paint(item, hit)
        else:
            self._y_edits[key] = val
            c = getattr(self, '_cellc', None) or cell_colors()
            item.setBackground(c['edit']); item.setForeground(c['fg'])
            item.setToolTip('изменено, не сохранено')
        self.y_status.setText(f'несохранённых правок: {len(self._y_edits)}')

    def _y_set_text(self, item, value):
        """Молча (без сигнала правки) вернуть в клетку прежнее значение."""
        self._y_filling = True
        item.setText('' if value is None else f'{value:g}')
        self._y_filling = False

    def _y_clear_selected(self):
        """Убрать свои значения в выделенных клетках (значения источника не трогаются)."""
        if self._y_av is None:
            return
        n = 0
        for item in self.y_table.selectedItems():
            key = item.data(Qt.UserRole)
            if key is None:
                continue
            hit = self._y_av['values'].get(key)
            if key not in self._y_edits and (hit is None or hit[1] != Y.SRC_USER):
                continue                                   # источник или пустая клетка — нечего убирать
            item.setText('')                               # обработчик запишет правку «удалить»
            n += 1
        if n == 0:
            QMessageBox.information(self, 'Убрать значение',
                                    'В выделении нет ваших значений (белые клетки — из источника).')

    def _y_save(self):
        """Записать несохранённые правки в таблицу user_yields и перестроить таблицу."""
        if self._y_av is None or not self._y_edits:
            QMessageBox.information(self, 'Сохранение', 'Нет изменений для сохранения.'); return
        edits = [(tid, year, val) for (tid, year), val in self._y_edits.items()]
        try:
            res = Y.save_manual(edits, self._y_av['culture'], account_id=self.acc.get('account_id'))
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, 'Сохранение', f'Не удалось записать: {exc}'); return
        QMessageBox.information(self, 'Сохранение',
                                f'Записано значений: {res["written"]}, удалено своих: {res["deleted"]}.\n'
                                f'Они попадут в прогноз и в сборку датасетов наравне с данными источников.')
        self._y_show()                                     # перечитать из БД (правки станут зелёными)

    def _y_export(self):
        """Выгрузить показанную таблицу в CSV (для просмотра/печати, обратно не читается)."""
        if self._y_av is None:
            QMessageBox.information(self, 'Выгрузка', 'Сначала постройте таблицу.'); return
        default = f'yields_{self._y_av["culture"]}_{self.y_y1.value()}-{self.y_y2.value()}.csv'
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить таблицу', default, 'CSV (*.csv)')
        if not path:
            return
        Y.to_dataframe(self._y_av).to_csv(path, index=False, sep=db.CSV_SEP,
                                          decimal=db.CSV_DECIMAL, encoding='utf-8-sig')
        self.y_status.setText(f'таблица выгружена: {os.path.basename(path)}')

    def _goto_yields_check(self):
        """Открыть раздел проверки урожайности по культуре и регионам текущего задания."""
        idx = self.y_cult.findText(self.f_cult.currentText())
        if idx >= 0:
            self.y_cult.setCurrentIndex(idx)
        regs = {r for r, _d, _l in self._targets} or set(_checked(self.f_regs)) or {self._cur_region()}
        regs.discard(None)
        for i in range(self.y_regs.count()):
            it = self.y_regs.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in regs else Qt.Unchecked)
        self.tabs.setCurrentIndex(self._yields_tab_index)
        if regs:
            self._y_show()

    # ============================ Прогноз (#4,#5,#7) ============================ #
    def _tab_forecast(self):
        w = QWidget(); lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.f_cult = QComboBox(); self.f_cult.addItems(P.list_cultures())
        self.f_cult.currentIndexChanged.connect(self._rebuild_models)
        self.f_cult.currentIndexChanged.connect(lambda *_: self._check_districts_with_data())
        self.f_y1 = QSpinBox(); self.f_y1.setRange(2000, YEAR_NOW + 1); self.f_y1.setValue(YEAR_NOW)
        self.f_range = QCheckBox('диапазон до:')
        self.f_y2 = QSpinBox(); self.f_y2.setRange(2000, YEAR_NOW + 1); self.f_y2.setValue(YEAR_NOW)
        top.addWidget(QLabel('Культура:')); top.addWidget(self.f_cult)
        top.addSpacing(20); top.addWidget(QLabel('Год:')); top.addWidget(self.f_y1)
        top.addWidget(self.f_range); top.addWidget(self.f_y2); top.addStretch(1)
        lay.addLayout(top)

        self._regname = dict(G.list_regions())
        mid = QHBoxLayout()
        # регионы (галочками) — можно отметить несколько; клик по строке показывает районы
        self.f_regs = QListWidget(); self.f_regs.setMaximumHeight(190)
        for rid, name in G.list_regions():
            _add_check(self.f_regs, f'{name} ({rid})', rid)
        self.f_regs.currentItemChanged.connect(lambda *_: self._on_region_changed())
        rcol = QVBoxLayout(); rcol.addWidget(QLabel('Регионы (галочками; клик — показать районы):'))
        rcol.addWidget(self.f_regs)
        rb0 = QHBoxLayout()
        ra = QPushButton('Все'); ra.clicked.connect(lambda: _set_all(self.f_regs, True))
        rn = QPushButton('Снять'); rn.clicked.connect(lambda: _set_all(self.f_regs, False))
        radd = QPushButton('Регионы целиком →'); radd.clicked.connect(self._task_add_regions)
        rb0.addWidget(ra); rb0.addWidget(rn); rb0.addWidget(radd); rcol.addLayout(rb0)
        mid.addLayout(rcol, 1)
        # районы текущего региона
        self._dist_excluded = {}        # {id_region: set(снятых районов)} — для «отмеченные регионы + их районы»
        self._dist_updating = False     # подавляем сигнал itemChanged при программной простановке галочек
        self.f_dist = QListWidget(); self.f_dist.setMaximumHeight(190)
        self.f_dist.itemChanged.connect(self._on_dist_item_changed)
        dcol = QVBoxLayout(); dcol.addWidget(QLabel('Районы текущего региона (галочками):')); dcol.addWidget(self.f_dist)
        b1 = QHBoxLayout()
        a1 = QPushButton('Все'); a1.clicked.connect(lambda: _set_all(self.f_dist, True))
        n1 = QPushButton('Снять'); n1.clicked.connect(lambda: _set_all(self.f_dist, False))
        add1 = QPushButton('Районы →'); add1.clicked.connect(self._task_add_districts)
        b1.addWidget(a1); b1.addWidget(n1); b1.addWidget(add1); dcol.addLayout(b1)
        mid.addLayout(dcol, 1)
        # задание
        tb = QVBoxLayout(); tb.addWidget(QLabel('Задание (территории для прогноза):'))
        self.f_targets = QListWidget(); self.f_targets.setMaximumHeight(190); tb.addWidget(self.f_targets)
        b2 = QHBoxLayout()
        clr = QPushButton('Очистить'); clr.clicked.connect(lambda: (self._targets.clear(), self.f_targets.clear()))
        allreg = QPushButton('ВСЕ районы РФ'); allreg.clicked.connect(self._task_all)
        b2.addWidget(clr); b2.addWidget(allreg); tb.addLayout(b2)
        mid.addLayout(tb, 1)
        lay.addLayout(mid)
        self._targets = []                                 # [(reg, dist|None, label)]

        addrow = QHBoxLayout()
        addall = QPushButton('Отмеченные регионы + их районы → задание')
        addall.setToolTip('Добавить в задание все отмеченные регионы и их районы с данными, '
                          'кроме районов, с которых снята галочка')
        addall.clicked.connect(self._task_add_checked)
        chk = QPushButton('Проверить урожайность по заданию')
        chk.setToolTip('Открыть раздел «2. Проверка урожайности»: какие данные есть по культуре в '
                       'территориях задания и что можно вписать руками')
        chk.clicked.connect(self._goto_yields_check)
        addrow.addWidget(addall); addrow.addWidget(chk); addrow.addStretch(1)
        lay.addLayout(addrow)

        tpl = QHBoxLayout(); tpl.addWidget(QLabel('Шаблон территорий:'))
        self.f_tpl = QComboBox(); self._reload_templates(); tpl.addWidget(self.f_tpl, 1)
        ld = QPushButton('Загрузить'); ld.clicked.connect(self._load_template)
        sv = QPushButton('Сохранить как шаблон'); sv.clicked.connect(self._save_template)
        tpl.addWidget(ld); tpl.addWidget(sv)
        lay.addLayout(tpl)
        if self.f_regs.count():
            self.f_regs.setCurrentRow(0)

        self.f_mbox = QGroupBox('Модели для прогноза (только доступные для культуры)')
        ml = QHBoxLayout(self.f_mbox)
        mc = QVBoxLayout(); mc.addWidget(QLabel('Модели (галочками):'))
        self.f_models_list = QListWidget(); self.f_models_list.setMaximumHeight(150)
        self.f_models_list.itemChanged.connect(lambda *_: self._sync_ensemble_enabled())
        mc.addWidget(self.f_models_list); ml.addLayout(mc, 1)
        ec = QVBoxLayout(); ec.addWidget(QLabel('Состав ансамбля (галочками):'))
        self.f_ens_list = QListWidget(); self.f_ens_list.setMaximumHeight(150)
        ec.addWidget(self.f_ens_list); ml.addLayout(ec, 1)
        lay.addWidget(self.f_mbox)
        self._rebuild_models()

        lay.addStretch(1)                                  # управление прижато кверху внутри своего модуля

        # строка запуска — ВНЕ прокрутки: главное действие вкладки всегда на виду, даже если
        # настройки выше не помещаются и прокручиваются
        bar = QWidget(); bl = QVBoxLayout(bar)
        m = lay.contentsMargins(); bl.setContentsMargins(m.left(), 6, m.right(), m.bottom())
        cm = QHBoxLayout(); cm.addWidget(QLabel('Комментарий:')); self.f_comment = QLineEdit()
        cm.addWidget(self.f_comment, 1); bl.addLayout(cm)
        rb = QHBoxLayout()
        self.f_run = QPushButton('Сделать прогноз'); self.f_run.clicked.connect(self._run_forecast)
        self.f_status = _status_label(); rb.addWidget(self.f_run); rb.addWidget(self.f_status); rb.addStretch(1)
        bl.addLayout(rb)
        head = QWidget(); hl = QVBoxLayout(head); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(0)
        hl.addWidget(self._scroll(w), 1); hl.addWidget(bar)

        # нижние модули с перетаскиваемыми границами (тяните полосу между ними мышью)
        jbox = QVBoxLayout(); jbox.setContentsMargins(0, 0, 0, 0)
        jbox.addWidget(QLabel('Докачка недостающих рядов из Vega:'))
        self.f_dllog = QTextEdit(); self.f_dllog.setReadOnly(True); self.f_dllog.setMinimumHeight(46)
        jbox.addWidget(self.f_dllog, 1)
        self.f_table = QTableWidget(); self.f_table.setMinimumHeight(80)
        return self._vsplit([head, self._panel(jbox), self.f_table], sizes=[580, 100, 240],
                            scroll_first=False)           # head уже содержит прокрутку

    def _f_dllog_line(self, text):
        self.f_dllog.append(text)
        self.f_dllog.ensureCursorVisible()

    def _rebuild_territories(self):
        """Перечитать списки регионов во всех вкладках (после установки базы или подгрузки данных).

        Списки строятся из таблицы ``territories`` при создании вкладок. Если базу поставили
        в уже запущенную программу (каталог, архив БДПМО, ЕМИСС), без этого они остаются
        пустыми до перезапуска. Выбранный регион и отметки сохраняются, если такие регионы
        есть и в новой базе.
        """
        regions = G.list_regions()
        self._regname = dict(regions)
        combo = getattr(self, 'd_reg', None)                # вкладка 1: выпадающий список
        if combo is not None:
            cur = combo.currentData()
            combo.blockSignals(True)                        # иначе районы перестроятся на каждый addItem
            combo.clear()
            for rid, name in regions:
                combo.addItem(f'{name} ({rid})', rid)
            i = combo.findData(cur)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)
            self._fill_districts(self.d_dist, combo.currentData())
        for attr in ('y_regs', 'f_regs', 'dl_regs'):        # вкладки 2, 3, 6: списки с галочками
            lw = getattr(self, attr, None)
            if lw is None:
                continue
            checked = set(_checked(lw))
            lw.clear()
            for rid, name in regions:
                _add_check(lw, f'{name} ({rid})', rid, checked=(rid in checked))

    def _rebuild_models(self):
        """Наполнить списки моделей доступными для культуры (последняя версия каждого типа — отмечена)."""
        avail = P.available_models(self.f_cult.currentText())
        self.f_models_list.clear(); self.f_ens_list.clear()
        seen = set()
        for k in avail:                                    # по умолчанию отмечаем последнюю версию каждого типа
            kind, _v = P.parse_token(k)
            _add_check(self.f_models_list, P.token_label(k), k, checked=(kind not in seen)); seen.add(kind)
        seen2 = set()
        for k in [m for m in avail if m != 'ensemble']:    # состав ансамбля — без самого ансамбля
            kind, _v = P.parse_token(k)
            _add_check(self.f_ens_list, P.token_label(k), k, checked=(kind not in seen2)); seen2.add(kind)
        self._sync_ensemble_enabled()

    def _sync_ensemble_enabled(self):
        self.f_ens_list.setEnabled('ensemble' in _checked(self.f_models_list))

    def _cur_region(self):
        it = self.f_regs.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _on_region_changed(self):
        """При выборе региона показать его районы и отметить только те, где есть данные по культуре."""
        self._fill_districts(self.f_dist, self._cur_region(), with_aggregate=False)
        self._check_districts_with_data()

    def _check_districts_with_data(self):
        """Отметить районы текущего региона с данными по культуре, сняв ранее исключённые вручную."""
        rid = self._cur_region()
        if rid is None:
            return
        try:
            have = P.districts_with_data(rid, self.f_cult.currentText())
        except Exception:                                  # noqa: BLE001 — на отказе отмечаем все
            have = None
        excl = self._dist_excluded.get(rid, set())
        self._dist_updating = True                         # программная простановка — не трогаем исключения
        for i in range(self.f_dist.count()):
            did = self.f_dist.item(i).data(Qt.UserRole)
            on = (have is None or did in have) and did not in excl
            self.f_dist.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._dist_updating = False

    def _on_dist_item_changed(self, item):
        """Запомнить, какие районы пользователь снял/вернул вручную (по текущему региону)."""
        if self._dist_updating:
            return
        rid = self._cur_region()
        if rid is None:
            return
        did = item.data(Qt.UserRole)
        excl = self._dist_excluded.setdefault(rid, set())
        if item.checkState() == Qt.Checked:
            excl.discard(did)
        else:
            excl.add(did)

    def _add_target(self, reg, did, label):
        item = (reg, did, label)
        if item not in self._targets:
            self._targets.append(item); self.f_targets.addItem(label)

    def _task_add_checked(self):
        """Одной кнопкой: в задание все отмеченные регионы + их районы с данными (кроме снятых вручную)."""
        regs = _checked(self.f_regs)
        if not regs:
            QMessageBox.information(self, 'Задание', 'Отметьте галочками хотя бы один регион.'); return
        cult = self.f_cult.currentText(); added_r = added_d = 0
        for rid in regs:
            self._add_target(rid, None, f'{self._regname.get(rid, rid)} — регион'); added_r += 1
            try:
                have = P.districts_with_data(rid, cult)
            except Exception:                              # noqa: BLE001
                have = set(d for d, _ in G.list_districts(rid))
            excl = self._dist_excluded.get(rid, set())
            for did, dname in G.list_districts(rid):
                if did in have and did not in excl:
                    self._add_target(rid, did, f'{self._regname.get(rid, rid)} / {dname}'); added_d += 1
        self.f_status.setText(f'добавлено в задание: регионов {added_r}, районов {added_d}')

    def _task_add_regions(self):
        """Добавить отмеченные регионы целиком (обобщённый уровень региона)."""
        for rid in _checked(self.f_regs):
            self._add_target(rid, None, f'{self._regname.get(rid, rid)} — регион')

    def _task_add_districts(self):
        """Добавить отмеченные районы текущего (выделенного) региона."""
        rid = self._cur_region()
        if rid is None:
            return
        for did in _checked(self.f_dist):
            self._add_target(rid, did, f'{self._regname.get(rid, rid)} / {self._dist_name(self.f_dist, did)}')

    def _task_all(self):
        if QMessageBox.question(self, 'Все территории',
                                'Добавить ВСЕ районы всех регионов? Это может быть очень долго.') != QMessageBox.Yes:
            return
        self._targets.clear(); self.f_targets.clear()
        cult = self.f_cult.currentText(); skipped = 0
        for reg, rname in G.list_regions():
            try:
                have = P.districts_with_data(reg, cult)
            except Exception:                              # noqa: BLE001
                have = None
            for did, dname in G.list_districts(reg):
                if have is not None and did not in have:   # без данных по культуре — пропускаем
                    skipped += 1; continue
                self._targets.append((reg, did, f'{rname} / {dname}'))
        msg = f'(всего территорий: {len(self._targets)}'
        if skipped:
            msg += f'; пропущено без данных по «{cult}»: {skipped}'
        self.f_targets.addItem(msg + ')')

    # ---- шаблоны территорий ----
    def _reload_templates(self):
        self.f_tpl.clear(); self.f_tpl.addItem('—', None)
        for n in S.list_templates():
            self.f_tpl.addItem(n, n)

    def _refresh_targets_list(self):
        self.f_targets.clear()
        for _, _, lab in self._targets:
            self.f_targets.addItem(lab)

    def _load_template(self):
        name = self.f_tpl.currentData()
        if not name:
            return
        self._targets = list(S.load_template(name))
        self._refresh_targets_list()
        self.f_status.setText(f'загружен шаблон «{name}» ({len(self._targets)} террит.)')

    def _save_template(self):
        if not self._targets:
            QMessageBox.warning(self, 'Шаблон', 'Сначала добавьте территории в задание.'); return
        name, ok = QInputDialog.getText(self, 'Сохранить шаблон', 'Название шаблона:')
        name = (name or '').strip()
        if not ok or not name:
            return
        if S.template_exists(name):
            QMessageBox.information(self, 'Шаблон',
                                    f'Шаблон «{name}» уже существует — будет перезаписан.')
        S.save_template(name, self._targets)
        self._reload_templates()
        idx = self.f_tpl.findData(name)
        if idx >= 0:
            self.f_tpl.setCurrentIndex(idx)
        self.f_status.setText(f'шаблон «{name}» сохранён')

    # ============================ Подгрузка данных ============================ #
    def _tab_dataload(self):
        """Отдельный раздел для подгрузки свежих данных из внешних источников (расширяемый)."""
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(_note('Подгрузка свежих данных из внешних источников в БД. '
                            'Разделы-источники добавляются по мере появления парсеров.'))

        # --- Источник 1: ЕМИСС (fedstat) — урожайность регионов ---
        box = QGroupBox('ЕМИСС (fedstat.ru) — урожайность регионов')
        v = QVBoxLayout(box)
        top = QHBoxLayout(); top.addWidget(QLabel('Культура:'))
        self.dl_cult = QComboBox(); self.dl_cult.addItems(P.list_cultures())
        top.addWidget(self.dl_cult); top.addStretch(1); v.addLayout(top)
        v.addWidget(QLabel('Регионы (галочками):'))
        self.dl_regs = QListWidget(); self.dl_regs.setMaximumHeight(180)
        for rid, name in G.list_regions():
            _add_check(self.dl_regs, f'{name} ({rid})', rid)
        v.addWidget(self.dl_regs)
        rb = QHBoxLayout()
        allb = QPushButton('Все'); allb.clicked.connect(lambda: _set_all(self.dl_regs, True))
        noneb = QPushButton('Снять'); noneb.clicked.connect(lambda: _set_all(self.dl_regs, False))
        self.dl_btn = QPushButton('Подгрузить урожайность (ЕМИСС)')
        self.dl_btn.setToolTip('Тянет с ЕМИСС урожайность отмеченных регионов по выбранной культуре '
                               'и пишет в БД (уровень региона). Значение за год — только если есть у ЕМИСС.')
        self.dl_btn.clicked.connect(self._emiss_load)
        self.dl_status = _status_label()
        rb.addWidget(allb); rb.addWidget(noneb); rb.addWidget(self.dl_btn)
        rb.addWidget(self.dl_status); rb.addStretch(1); v.addLayout(rb)
        lay.addWidget(box)

        # --- Источник 2: Архив урожайностей tochno.st («Если быть точным» / переработанный БДПМО) ---
        abox = QGroupBox('Архив урожайностей (tochno.st — переработанный БДПМО)')
        av = QVBoxLayout(abox)
        av.addWidget(_note('Скачайте набор БДПМО на https://tochno.st/ (файл data_bdmo_*.zip) и укажите путь. '
                           'Программа извлечёт урожайность на убранную площадь и занесёт её в БД по районам '
                           '(категория «Хозяйства всех категорий», только известные приложению культуры).'))
        pr = QHBoxLayout(); pr.addWidget(QLabel('Файл архива:'))
        self.dl_arc_path = QLineEdit(); self.dl_arc_path.setPlaceholderText('путь к data_bdmo_*.zip')
        browse = QPushButton('Обзор…'); browse.clicked.connect(self._arc_browse)
        pr.addWidget(self.dl_arc_path, 1); pr.addWidget(browse); av.addLayout(pr)
        ar = QHBoxLayout()
        self.dl_arc_btn = QPushButton('Загрузить урожайности из архива в БД')
        self.dl_arc_btn.clicked.connect(self._archive_load)
        ar.addWidget(self.dl_arc_btn); ar.addStretch(1); av.addLayout(ar)
        lay.addWidget(abox)

        self.dl_bar = QProgressBar(); self.dl_bar.setRange(0, 100); self.dl_bar.setValue(0)
        self.dl_bar.setTextVisible(True); self.dl_bar.setFormat('%p%')
        lay.addWidget(self.dl_bar)
        lay.addStretch(1)                                  # источники прижаты кверху внутри своего модуля

        # нижние модули с перетаскиваемыми границами (тяните полосу между ними мышью)
        jbox = QVBoxLayout(); jbox.setContentsMargins(0, 0, 0, 0)
        jbox.addWidget(QLabel('Журнал операций (последняя — внизу):'))
        self.dl_log = QTextEdit(); self.dl_log.setReadOnly(True); self.dl_log.setMinimumHeight(60)
        jbox.addWidget(self.dl_log, 1)
        tbox = QVBoxLayout(); tbox.setContentsMargins(0, 0, 0, 0)
        tbox.addWidget(QLabel('Итог по регионам:'))
        self.dl_table = QTableWidget(); self.dl_table.setMinimumHeight(80)
        tbox.addWidget(self.dl_table, 1)
        return self._vsplit([w, self._panel(jbox), self._panel(tbox)], sizes=[470, 150, 180])

    def _dl_log_line(self, text):
        self.dl_log.append(text)
        self.dl_log.ensureCursorVisible()

    # ---- Источник 2: архив tochno.st (БДПМО) ----
    def _arc_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Выберите архив tochno.st (БДПМО)', '', 'Архив ZIP (*.zip)')
        if path:
            self.dl_arc_path.setText(path)

    def _archive_load(self):
        path = self.dl_arc_path.text().strip().strip('"')
        if not path or not os.path.exists(path):
            QMessageBox.information(self, 'Архив', 'Укажите путь к скачанному zip-архиву tochno.st (БДПМО).'); return
        self.dl_bar.setValue(0); self.dl_log.clear()
        self._dl_log_line(f'Архив: {os.path.basename(path)}')

        def job(prog):
            import tempfile
            import bdpmo_archive_to_yields as A
            import ingest_bdpmo_table as I
            msg = self._worker.message.emit
            out = os.path.join(tempfile.gettempdir(), 'bdpmo_yields_tmp.csv')
            msg('Извлекаю урожайность из архива (раздел «Сельское хозяйство»)…')

            def cprog(seen, kept):
                prog(min(65, int(65 * seen / 9_000_000)))  # ~65% на извлечение (в разделе ~8.8 млн строк)
                msg(f'   …просмотрено {seen:,}, урожайности {kept:,}')
            _, kept = A.convert(path, out, progress=cprog)
            msg(f'Извлечено строк урожайности: {kept:,}')
            prog(72); msg('Сопоставление районов/культур и запись в БД…')
            stats = I.ingest(out, dry_run=False)
            try:
                os.remove(out)
            except OSError:
                pass
            prog(100)
            return {'kept': kept, 'stats': stats}

        self.dl_arc_btn.setEnabled(False); self.dl_btn.setEnabled(False)
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(self.dl_bar.setValue)
        self._worker.message.connect(self._dl_log_line)
        self._worker.done.connect(self._archive_done)
        self._worker.failed.connect(lambda tb: (self.dl_arc_btn.setEnabled(True), self.dl_btn.setEnabled(True),
                                                self._dl_log_line('ОШИБКА: ' + tb.strip().splitlines()[-1]),
                                                QMessageBox.critical(self, 'Ошибка архива', tb[-1500:])))
        self._worker.start()

    def _archive_done(self, res):
        self.dl_arc_btn.setEnabled(True); self.dl_btn.setEnabled(True); self.dl_bar.setValue(100)
        self._rebuild_territories()                        # в базе могли появиться новые территории
        s = res['stats']
        self._dl_log_line(
            f'Готово. Извлечено {res["kept"]:,}; «Хозяйства всех категорий» {s["after_cat"]:,}; '
            f'культур приложения {s["after_cult"]:,}; районов сопоставлено {s["matched"]:,}; '
            f'записано в БД {s["written"]:,}.')
        if s.get('backup'):
            self._dl_log_line(f'   бэкап yields: {s["backup"]}')
        QMessageBox.information(self, 'Архив',
                               f'Загружено урожайностей в БД: {s["written"]:,}.\n'
                               f'Сопоставлено районов {s["matched"]:,}, не сопоставлено {s["unmatched"]:,} '
                               f'(регионы вне охвата приложения, города, агрегаты).\n'
                               f'Бэкап yields: {s.get("backup") or "—"}')

    def _emiss_load(self):
        regs = list(_checked(self.dl_regs))
        if not regs:
            QMessageBox.information(self, 'ЕМИСС', 'Отметьте галочками хотя бы один регион.'); return
        cult = self.dl_cult.currentText()
        self.dl_bar.setValue(0); self.dl_log.clear()
        self._dl_log_line(f'Старт подгрузки ЕМИСС: культура «{cult}», регионов {len(regs)}.')

        def job(prog):
            def cb(i, total, msg=''):
                prog(int(100 * i / max(total, 1)))
                if msg:
                    self._worker.message.emit(msg)        # живой комментарий (последняя операция)
            return ST.load_fresh_yields(regs, cult, progress=cb)
        self.dl_btn.setEnabled(False); self.dl_status.setText('подгрузка…')
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(self.dl_bar.setValue)
        self._worker.message.connect(self._dl_log_line)
        self._worker.done.connect(self._emiss_done)
        self._worker.failed.connect(lambda tb: (self.dl_btn.setEnabled(True), self.dl_status.setText('ошибка'),
                                                self._dl_log_line('ОШИБКА: ' + tb.strip().splitlines()[-1]),
                                                QMessageBox.critical(self, 'Ошибка ЕМИСС', tb[-1500:])))
        self._worker.start()

    def _emiss_done(self, res):
        import pandas as pd
        self.dl_btn.setEnabled(True); self.dl_bar.setValue(100)
        if not res.get('ok'):
            self.dl_status.setText(res.get('reason', '')); self._dl_log_line('' + res.get('reason', ''))
            QMessageBox.information(self, 'ЕМИСС', res.get('reason', 'нет данных')); return
        self._rebuild_territories()                        # в базе могли появиться новые территории
        rows = res['rows']; ok = sum(1 for r in rows if r.get('лет'))
        df_to_table(self.dl_table, pd.DataFrame(rows))
        summary = f'Готово: регионов с данными {ok}/{len(rows)}, записей в БД: {res["written"]}'
        self.dl_status.setText(summary); self._dl_log_line('' + summary)
        if ok == 0:
            QMessageBox.information(self, 'ЕМИСС',
                                    'Данные не загружены. Возможные причины: блокировка запросов по IP '
                                    '(в т.ч. IPv6), нет данных ЕМИСС по этим регионам, или названия регионов '
                                    'не совпали со справочником ЕМИСС.')

    def _forecast_params(self):
        """Собрать параметры партии из формы (или None, если чего-то не хватает)."""
        targets = list(self._targets)
        if not targets:                                    # задание пусто — берём текущие галочки
            for rid in _checked(self.f_regs):
                targets.append((rid, None, f'{self._regname.get(rid, rid)} — регион'))
            rid = self._cur_region()
            if rid is not None:
                for d in _checked(self.f_dist):
                    targets.append((rid, d, f'{self._regname.get(rid, rid)} / {self._dist_name(self.f_dist, d)}'))
        targets = [t for t in targets if t[0] is not None]
        if not targets:
            QMessageBox.warning(self, 'Прогноз',
                                'Добавьте территории в задание (или отметьте регионы/районы галочками).')
            return None
        y1, y2 = self.f_y1.value(), (self.f_y2.value() if self.f_range.isChecked() else self.f_y1.value())
        models = _checked(self.f_models_list)
        if not models:
            QMessageBox.warning(self, 'Прогноз', 'Выберите хотя бы одну модель в списке.')
            return None
        return {'targets': targets, 'cult': self.f_cult.currentText(),
                'years': list(range(min(y1, y2), max(y1, y2) + 1)), 'models': models,
                'ens_of': _checked(self.f_ens_list) or None,
                'comment': self.f_comment.text().strip() or None,
                'ukey': A.current_ukey(), 'account_id': self.acc['account_id']}

    def _run_forecast(self):
        """Шаг 1: обследовать базу под партию и показать план (что посчитается и почему нет)."""
        params = self._forecast_params()
        if params is None:
            return

        def job(prog):
            return PF.survey(params['targets'], params['cult'], params['years'], params['models'],
                             ensemble_of=params['ens_of'], has_ukey=bool(params['ukey']),
                             progress=prog)

        self.f_run.setEnabled(False)
        self.f_status.setText('обследую базу: 0%')
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(lambda p: self.f_status.setText(f'обследую базу: {p}%'))
        self._worker.done.connect(lambda res: self._preflight_done(params, res))
        self._worker.failed.connect(self._forecast_fail)
        self._worker.start()

    def _preflight_done(self, params, survey):
        """Шаг 2: окно плана; при согласии — запуск расчёта."""
        self.f_run.setEnabled(True)
        s = survey['summary']
        self.f_status.setText(f'план: посчитается {s["expected"]} из {s["planned"]}')
        dlg = PreflightDialog(self, survey)
        app_theme.mark_buttons(dlg)
        if dlg.exec() != QDialog.Accepted:
            self.f_status.setText('прогноз отменён')
            return
        self._start_forecast(params, survey)

    def _start_forecast(self, params, survey):
        """Шаг 3: расчёт партии с учётом плана; по каждому прогнозу пишется итог и причина."""
        targets, cult, years, models = params['targets'], params['cult'], params['years'], params['models']
        ens_of, comment, aid = params['ens_of'], params['comment'], params['account_id']
        ukey = params['ukey']
        has_ukey = bool(ukey)
        plan = {(r.territory_id, int(r.year), r.model): (r.status, r.reason)
                for r in survey['items'].itertuples()}
        batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # одна партия на запуск
        total = len(targets)

        def job(prog):
            import pandas as pd
            cfg = P.resolve_culture(cult); rows = []; outcomes = []; done = 0
            msg = self._worker.message.emit
            _keys = ('requested', 'downloaded', 'empty', 'no_uid', 'net_fail', 'bad_response')
            dl_tot = {k: 0 for k in _keys}                 # общая сводка докачки Vega по партии
            for reg, did, label in targets:
                tid = db.territory_id(reg, did)
                terr = {k: 0 for k in _keys}               # сводка по территории (по всем годам)
                for yr in years:
                    # докачивать есть смысл, только если в плане есть что докачать и есть чем
                    need_dl = has_ukey and any(plan.get((tid, yr, m), ('',))[0] == PF.DOWNLOAD
                                               for m in models)
                    r = P.predict_for_territory(reg, did, cult, yr, models, download=need_dl,
                                                ukey=ukey, ensemble_of=ens_of)
                    d = r.get('download')
                    if d:
                        for k in _keys:
                            terr[k] += int(d.get(k, 0))
                    for m in models:
                        v = r['values'].get(m)
                        db.upsert_prediction(tid, cult, f'{cfg["task_id"]}:{m}',
                                             yr, v, average_productivity=r['prod_hist'],
                                             account_id=aid, comment=comment, batch_id=batch_id)
                        p_status, p_reason = plan.get((tid, yr, m), ('', ''))
                        ok = v is not None
                        reason = '' if ok else PF.run_reason(p_status, p_reason, r['notes'].get(m),
                                                             has_ukey, d)
                        outcomes.append({'territory_id': tid, 'label': label, 'year': yr, 'model': m,
                                         'planned': p_status, 'status': 'ok' if ok else 'failed',
                                         'reason': reason})
                        rows.append({'территория': label, 'год': yr, 'модель': P.token_label(m),
                                     'прогноз, ц/га': round(v, 2) if v is not None else None,
                                     'почему не рассчитан': reason})
                for k in _keys:
                    dl_tot[k] += terr[k]
                if terr['requested']:                      # были докачки — покажем строку по территории
                    extra = ''.join([f', без uid {terr["no_uid"]}' if terr['no_uid'] else '',
                                     f', сбоев сети {terr["net_fail"]}' if terr['net_fail'] else '',
                                     f', битых ответов {terr["bad_response"]}' if terr['bad_response'] else ''])
                    msg(f'   {label}: скачано {terr["downloaded"]}, пусто {terr["empty"]}{extra} '
                        f'(запросов {terr["requested"]})')
                # уточнённые MSE/R² территории считаем здесь и сохраняем в БД ->
                # отчёт по партии потом просто читает их (быстро, без повторного бэктеста)
                try:
                    R.store_territory_metrics(reg, did, cult, models, batch_id, account_id=aid)
                except Exception:                          # noqa: BLE001 — метрики не критичны для прогноза
                    pass
                done += 1; prog(int(100 * done / total))
            if not has_ukey:
                msg('ukey не указан — считали только по данным, которые уже есть в базе.')
            elif dl_tot['requested']:
                msg(f'Докачка Vega по партии: запрошено {dl_tot["requested"]}, '
                    f'скачано {dl_tot["downloaded"]}, пусто {dl_tot["empty"]}, '
                    f'без uid {dl_tot["no_uid"]}, сбоев сети {dl_tot["net_fail"]}, '
                    f'битых ответов {dl_tot["bad_response"]}.')
            else:
                msg('Докачка Vega не потребовалась — все ряды уже в БД.')
            PF.save_outcomes(batch_id, cult, outcomes, has_ukey)
            summary, reasons, _failed = PF.batch_outcomes(batch_id)
            return {'table': pd.DataFrame(rows), 'summary': summary, 'reasons': reasons,
                    'expected': survey['summary']['expected'], 'batch_id': batch_id}

        self.f_run.setEnabled(False)
        self.f_dllog.clear(); self._f_dllog_line('Прогноз запущен…')
        self.f_status.setText(f'считаю прогноз: 0% (0/{total} террит. × {len(years)} лет)')
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(
            lambda p: self.f_status.setText(f'считаю прогноз: {p}% ({total} террит. × {len(years)} лет)'))
        self._worker.message.connect(self._f_dllog_line)
        self._worker.done.connect(self._forecast_done)
        self._worker.failed.connect(self._forecast_fail)
        self._worker.start()

    def _forecast_done(self, res):
        df_to_table(self.f_table, res['table'])
        self.f_run.setEnabled(True)
        s = res['summary']
        if not s:
            self.f_status.setText('готово (сохранено в БД)')
            return
        self.f_status.setText(f'готово: рассчитано {s["done"]} из {s["planned"]} '
                              f'(по плану ожидалось {res["expected"]})')
        self._f_dllog_line('' + PF.outcome_text(s))
        for _, row in res['reasons'].head(8).iterrows():
            self._f_dllog_line(f'   не рассчитано {int(row["прогнозов"])}: {row["причина"]}')
        self._reload_batches()                             # партия сразу видна во вкладке прогнозов
        if s['failed']:
            QMessageBox.information(
                self, 'Прогноз завершён',
                PF.outcome_text(s) + '\n\nПричины по каждому прогнозу — во вкладке '
                '«5. Прогнозы и отчёты» → «План и итог».')

    def _forecast_fail(self, tb):
        self.f_run.setEnabled(True); self.f_status.setText('ошибка')
        QMessageBox.critical(self, 'Ошибка прогноза', tb[-1500:])

    # ============================ Каталог (модели и база) ============================ #
    def _tab_catalog(self):
        """Что автор выложил в общий доступ: модели и снимки базы — скачать и обновить."""
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(_note(
            'Каталог готовых моделей и снимков базы данных. Программа читает список из одной '
            'публичной папки автора и ставит выбранное по кнопке. Снимок базы заменяет только '
            'справочные таблицы: ваши ручные урожайности, учётные записи, прогнозы и шаблоны '
            'не трогаются (прежние файлы сохраняются в workspace/legacy).'))
        srow = QHBoxLayout()
        srow.addWidget(QLabel('Источник:'))
        self.cat_src = QLineEdit(AS.source())
        self.cat_src.setPlaceholderText('публичная ссылка Яндекс.Диска или путь к папке')
        self.cat_src.setToolTip('Публичная ссылка на папку с catalog.json — или локальная папка '
                                '(раздача по сети, флешка, проверка сборки)')
        srow.addWidget(self.cat_src, 1)
        save = QPushButton('Сохранить'); save.clicked.connect(self._cat_save_source)
        ref = QPushButton('Обновить список'); ref.clicked.connect(self._cat_refresh)
        srow.addWidget(save); srow.addWidget(ref)
        lay.addLayout(srow)
        self.cat_show_ds = QCheckBox('показать датасеты обучения (гигабайты; нужны только '
                                     'для переобучения моделей)')
        self.cat_show_ds.setToolTip('Обычному пользователю датасеты не нужны: прогноз считается '
                                    'по базе и моделям. Пока галочка снята, их нет в списке и '
                                    'они не попадут под «выделить всё».')
        self.cat_show_ds.stateChanged.connect(lambda *_: self._cat_show_cached())
        lay.addWidget(self.cat_show_ds)

        brow = QHBoxLayout()
        get = QPushButton('Скачать / обновить выбранное'); get.clicked.connect(self._cat_install)
        rm = QPushButton('Удалить'); rm.clicked.connect(self._cat_remove)
        rm.setToolTip('Удалить установленную модель или датасет с диска '
                      '(снимки базы так не удаляются)')
        self.cat_bar = QProgressBar(); self.cat_bar.setRange(0, 100); self.cat_bar.setValue(0)
        self.cat_bar.setTextVisible(True); self.cat_bar.setFormat('%p%'); self.cat_bar.setMaximumWidth(180)
        self.cat_status = _status_label()
        brow.addWidget(get); brow.addWidget(rm); brow.addWidget(self.cat_bar)
        brow.addWidget(self.cat_status); brow.addStretch(1)
        lay.addLayout(brow)
        lay.addStretch(1)

        self._cat_df = None
        self.cat_table = QTableWidget()
        self.cat_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cat_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.cat_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cat_table.setMinimumHeight(140)
        jbox = QVBoxLayout(); jbox.setContentsMargins(0, 0, 0, 0)
        jbox.addWidget(QLabel('Журнал загрузки (последняя запись — внизу):'))
        self.cat_log = QTextEdit(); self.cat_log.setReadOnly(True); self.cat_log.setMinimumHeight(60)
        jbox.addWidget(self.cat_log, 1)
        return self._vsplit([w, self.cat_table, self._panel(jbox)], sizes=[240, 320, 160])

    def _cat_log_line(self, text):
        self.cat_log.append(text)
        self.cat_log.ensureCursorVisible()

    def _cat_save_source(self):
        AS.set_source(self.cat_src.text())
        self.cat_status.setText('источник сохранён')
        self._cat_refresh()

    def _cat_refresh(self):
        """Перечитать манифест источника и показать, что доступно и что установлено."""
        if not self.cat_src.text().strip():
            QMessageBox.information(self, 'Каталог',
                                    'Укажите источник: публичную ссылку на папку с каталогом '
                                    'или путь к локальной папке.'); return

        show_ds = self.cat_show_ds.isChecked()

        def job():
            cat = AS.fetch_catalog(src=self.cat_src.text().strip())
            return cat, AS.status_frame(cat, include_optional=show_ds)

        self.cat_status.setText('читаю каталог…')
        self._worker = Worker(job)
        self._worker.done.connect(self._cat_show)
        self._worker.failed.connect(lambda tb: (
            self.cat_status.setText('каталог недоступен'),
            self._cat_log_line('' + tb.strip().splitlines()[-1]),
            QMessageBox.critical(self, 'Каталог', 'Не удалось прочитать каталог:\n\n'
                                 + tb.strip().splitlines()[-1])))
        self._worker.start()

    def _cat_show(self, res):
        cat, frame = res
        self._cat_df = frame
        df_to_table(self.cat_table, frame)
        n_upd = int((frame['состояние'] == 'обновление').sum()) if not frame.empty else 0
        n_new = int((frame['состояние'] == 'нет').sum()) if not frame.empty else 0
        note = ' (список из кэша — источник сейчас недоступен)' if cat.get('_offline') else ''
        self.cat_status.setText(f'записей {len(frame)}: доступно {n_new}, обновлений {n_upd}{note}')
        if cat.get('note'):
            self._cat_log_line('' + str(cat['note']))
        self._mark_updates(n_upd)

    def _cat_show_cached(self):
        """Перерисовать список из уже прочитанного каталога (переключение показа датасетов)."""
        cat = AS.cached_catalog()
        if cat is not None:
            self._cat_show((cat, AS.status_frame(cat, include_optional=self.cat_show_ds.isChecked())))

    def _cat_selected_ids(self):
        if self._cat_df is None or self._cat_df.empty:
            return []
        rows = sorted({i.row() for i in self.cat_table.selectedIndexes()})
        return [str(self._cat_df.iloc[r]['id']) for r in rows if r < len(self._cat_df)]

    def _cat_install(self):
        """Скачать и установить выбранные записи каталога (по очереди, с прогрессом)."""
        ids = self._cat_selected_ids()
        if not ids:
            QMessageBox.information(self, 'Каталог', 'Выберите в таблице, что скачать.'); return
        src = self.cat_src.text().strip()
        cat = AS.cached_catalog() or {}
        chosen = [a for a in AS.assets(cat, include_optional=True) if str(a['id']) in set(ids)]
        total_mb = sum(int(a.get('bytes') or 0) for a in chosen) / 1048576
        optional = [a for a in chosen if AS.is_optional(a)]
        if optional:                                       # датасеты — только осознанно
            opt_mb = sum(int(a.get('bytes') or 0) for a in optional) / 1048576
            names = ', '.join(str(a.get('title') or a['id']) for a in optional[:4])
            if len(optional) > 4:
                names += f' и ещё {len(optional) - 4}'
            text = '\n\n'.join([
                f'Среди выбранного — датасеты обучения: {len(optional)} шт., {opt_mb:.0f} МБ.',
                f'({names})',
                'Они нужны только для ПЕРЕОБУЧЕНИЯ моделей. Для прогноза достаточно базы и '
                'моделей, а после установки датасет займёт на диске в несколько раз больше '
                'места, чем скачанный архив.',
                'Всё равно скачать?'])
            if QMessageBox.question(self, 'Датасеты обучения', text) != QMessageBox.Yes:
                return
        if QMessageBox.question(self, 'Каталог',
                                f'Скачать и установить: {len(chosen)} шт., {total_mb:.1f} МБ?') != QMessageBox.Yes:
            return

        def job(prog):
            msg = self._worker.message.emit
            done = []
            for i, a in enumerate(chosen):
                base = int(100 * i / len(chosen))
                span = 100 / len(chosen)
                AS.install(a, src=src, message=msg,
                           progress=lambda p, b=base, s=span: prog(int(b + p * s / 100)))
                done.append(a['id'])
            return done

        self.cat_bar.setValue(0); self.cat_status.setText('загрузка…')
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(self.cat_bar.setValue)
        self._worker.message.connect(self._cat_log_line)
        self._worker.done.connect(self._cat_install_done)
        self._worker.failed.connect(lambda tb: (
            self.cat_status.setText('ошибка'), self.cat_bar.setValue(0),
            self._cat_log_line('' + tb.strip().splitlines()[-1]),
            QMessageBox.critical(self, 'Ошибка загрузки', tb[-1500:])))
        self._worker.start()

    def _cat_install_done(self, done):
        self.cat_bar.setValue(100)
        self.cat_status.setText(f'установлено: {len(done)}')
        self._rebuild_models()                             # новые модели — сразу в списке моделей
        self._rebuild_territories()                        # и регионы из установленного снимка базы
        self._cat_refresh()

    def _cat_remove(self):
        ids = [i for i in self._cat_selected_ids()
               if i.startswith('model:') or i.startswith('dataset:')]
        if not ids:
            QMessageBox.information(self, 'Каталог',
                                    'Выберите установленные модели или датасеты '
                                    '(снимки базы так не удаляются).'); return
        if QMessageBox.question(self, 'Удаление',
                                f'Удалить с диска: {len(ids)} шт.?') != QMessageBox.Yes:
            return
        ok = [i for i in ids if AS.remove(i)]
        failed = [i for i in ids if i not in ok]
        self._cat_log_line('удалено: ' + str(len(ok))
                           + (f', не удалось: {len(failed)}' if failed else ''))
        if failed:
            QMessageBox.warning(self, 'Удаление', 'Не удалось удалить файлы: '
                                + ', '.join(failed) + '. Возможно, папка открыта в проводнике '
                                'или занята другой программой; записи оставлены как установленные.')
        self._rebuild_models()
        self._cat_refresh()

    def _mark_updates(self, n):
        """Отметить вкладку каталога, если есть что обновить (тихо, без окон)."""
        idx = getattr(self, '_catalog_tab_index', None)
        if idx is None:
            return
        base = '7. Каталог'
        self.tabs.setTabText(idx, f'{base} ({n})' if n else base)

    def offer_first_run(self):
        """Пустая база при первом запуске: предложить скачать готовую из каталога."""
        try:
            empty = db.load('territories').empty or db.load('yields').empty
        except Exception:                                  # noqa: BLE001 — не мешаем запуску
            return
        if not empty:
            return
        text = '\n\n'.join([
            'В базе пока нет данных.',
            'Скачать готовую базу (и, если нужно, модели) из каталога?',
            'Иначе данные можно собрать самому — вкладка «6. Подгрузка данных»: '
            'ЕМИСС и архив БДПМО.'])
        if QMessageBox.question(self, 'Первый запуск', text) != QMessageBox.Yes:
            return
        self.tabs.setCurrentIndex(self._catalog_tab_index)
        if AS.source():
            self._cat_refresh()

    def _check_catalog_quietly(self):
        """Фоновая проверка каталога при запуске: только отметка на вкладке, ничего не качаем."""
        if not AS.source():
            return

        def job():
            return AS.updates_available()

        self._cat_worker = Worker(job)
        self._cat_worker.done.connect(lambda n: self._mark_updates(int(n or 0)))
        self._cat_worker.failed.connect(lambda _tb: None)   # нет сети — молчим
        self._cat_worker.start()

    # ============================ Справка ============================ #
    def _tab_help(self):
        w = QWidget(); lay = QVBoxLayout(w)
        view = QTextBrowser(); view.setOpenExternalLinks(False); view.setHtml(HELP_HTML)
        lay.addWidget(view)
        return w

    # ============================ Модели (#6) ============================ #
    def _tab_models(self):
        w = QWidget(); lay = QVBoxLayout(w)
        info = QLabel('Арсенал моделей по культурам. Нейросети, случайный лес и линейная модель '
                      'сохраняются файлами в models/<задача>_<тип>/ с метаданными (тип, культура). '
                      'Классическая регрессия по NDVI считается на лету. «Сохранена» — есть ли файл модели. '
                      'MSE/RMSE/R² — точность по бэктесту (кнопка «Оценить точность»). '
                      'Столбец «комментарий» — ваша заметка о версии модели; она хранится файлом '
                      'comment.txt в папке самой модели и уезжает вместе с ней.')
        info.setWordWrap(True); lay.addWidget(info)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel('Культура:'))
        self.mm_cult = QComboBox(); self.mm_cult.addItems(P.list_cultures()); ctl.addWidget(self.mm_cult)
        est = QPushButton('Оценить точность (бэктест по выборке)')
        est.clicked.connect(self._estimate_model_metrics)
        trn = QPushButton('Обучить RF/линейную')
        trn.setToolTip('Обучить и сохранить случайный лес и линейную модель (v1) для выбранной культуры')
        trn.clicked.connect(self._train_sklearn)
        ref = QPushButton('Обновить'); ref.clicked.connect(self._reload_models)
        cmt = QPushButton('Комментарий к модели')
        cmt.setToolTip('Заметка о выбранной модели: чем эта версия отличается. Хранится файлом '
                       'comment.txt в папке модели (двойной щелчок по строке — то же самое)')
        cmt.clicked.connect(self._edit_model_comment)
        self.mm_status = _status_label()
        ctl.addWidget(est); ctl.addWidget(trn); ctl.addWidget(cmt); ctl.addWidget(ref)
        ctl.addWidget(self.mm_status); ctl.addStretch(1)
        lay.addLayout(ctl)
        self.m_table = QTableWidget()
        self.m_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.m_table.setEditTriggers(QTableWidget.NoEditTriggers)   # правка — через диалог заметки
        self.m_table.doubleClicked.connect(lambda *_: self._edit_model_comment())
        lay.addWidget(self.m_table, 1)
        self._models_inv = None
        self._reload_models()
        return w

    def _reload_models(self):
        import pandas as pd
        inv = pd.DataFrame(P.model_inventory())
        metrics = db.get_model_metrics()

        def met(row, field):
            d = metrics.get((str(row['культура']), str(row['mkey'])))   # join по токену модели
            v = None if not d else d.get(field)
            return None if v is None or v != v else (int(v) if field == 'n' else round(float(v), 2))

        if not inv.empty:
            inv['сохранена'] = inv['сохранена'].map({True: 'да', False: 'нет'})
            inv['MSE'] = inv.apply(lambda r: met(r, 'mse'), axis=1)
            inv['RMSE'] = inv.apply(lambda r: met(r, 'rmse'), axis=1)
            inv['R²'] = inv.apply(lambda r: met(r, 'r2'), axis=1)
            inv['оценка n'] = inv.apply(lambda r: met(r, 'n'), axis=1)
        self._models_inv = inv                              # с mkey: строка таблицы -> токен модели
        cols = [c for c in inv.columns if c != 'mkey'] if not inv.empty else None
        order = ['культура', 'модель', 'сохранена', 'создана', 'комментарий', 'расположение',
                 'MSE', 'RMSE', 'R²', 'оценка n']
        if cols:
            cols = [c for c in order if c in cols] + [c for c in cols if c not in order]
        df_to_table(self.m_table, inv[cols] if cols else inv)

    def _edit_model_comment(self):
        """Заметка о модели: чем эта версия отличается. Хранится в папке модели (comment.txt)."""
        row = self.m_table.currentRow()
        inv = self._models_inv
        if row < 0 or inv is None or inv.empty or row >= len(inv):
            QMessageBox.information(self, 'Комментарий', 'Выберите модель в таблице.'); return
        token = str(inv.iloc[row]['mkey'])
        if P.model_dir(token) is None:
            QMessageBox.information(self, 'Комментарий',
                                    'У этой модели нет папки: классическая регрессия по NDVI и '
                                    'ансамбль считаются на лету, заметку хранить негде.'); return
        auto = P.model_autonote(token)
        prompt = token if not auto else f'{token}\n\nИз метаданных модели: {auto}'
        text, ok = QInputDialog.getMultiLineText(self, 'Комментарий к модели', prompt,
                                                 P.get_model_comment(token))
        if not ok:
            return
        try:
            path = P.set_model_comment(token, text)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, 'Комментарий', str(exc)); return
        self.mm_status.setText('заметка удалена' if path is None
                               else f'заметка сохранена: {os.path.basename(path)} в папке модели')
        self._reload_models()
        self.m_table.selectRow(row)

    def _train_sklearn(self):
        cult = self.mm_cult.currentText()

        def job():
            return P.train_sklearn(cult)
        self.mm_status.setText('обучаю RF/линейную…')
        self._worker = Worker(job)
        self._worker.done.connect(lambda res: (
            self.mm_status.setText('обучение: ' + ', '.join(f'{k}: {s}' for k, s in res)),
            self._reload_models()))
        self._worker.failed.connect(lambda tb: (self.mm_status.setText('ошибка'),
                                                QMessageBox.critical(self, 'Ошибка', tb[-1500:])))
        self._worker.start()

    def _estimate_model_metrics(self):
        cult = self.mm_cult.currentText()

        def job(prog):
            return R.model_accuracy(cult, progress=lambda i, t: prog(int(100 * i / max(t, 1))))
        self.mm_status.setText('оцениваю точность… 0%')
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(lambda p: self.mm_status.setText(f'оцениваю точность… {p}%'))
        self._worker.done.connect(lambda df: (self.mm_status.setText(f'готово ({cult})'), self._reload_models()))
        self._worker.failed.connect(lambda tb: (self.mm_status.setText('ошибка'),
                                                QMessageBox.critical(self, 'Ошибка', tb[-1500:])))
        self._worker.start()

    # ============================ Прогнозы и отчёты ============================ #
    def _tab_predictions(self):
        # модуль 1: список партий
        p1 = QVBoxLayout(); p1.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel('Список прогнозов (партий) — кто и когда делал:')); top.addStretch(1)
        self.p_only_me = QCheckBox('только мои')
        ref = QPushButton('Обновить'); ref.clicked.connect(self._reload_batches)
        top.addWidget(self.p_only_me); top.addWidget(ref); p1.addLayout(top)
        self.b_table = QTableWidget(); self.b_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.b_table.setMinimumHeight(80); p1.addWidget(self.b_table, 1)

        # модуль 2: управление отчётом (компактное)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel('Глубина лет урожайности (0 = все):'))
        self.p_depth = QSpinBox(); self.p_depth.setRange(0, 30); self.p_depth.setValue(6); ctl.addWidget(self.p_depth)
        gen = QPushButton('Сформировать отчёт по партии (CSV + Word)'); gen.clicked.connect(self._make_batch_report)
        exp = QPushButton('Выгрузить прогноз (папка + ZIP)')
        exp.setToolTip('Собрать самостоятельную папку с таблицами, метаданными и отчётом — '
                       'её можно отправить коллеге по почте')
        exp.clicked.connect(self._export_batch)
        self.p_with_report = QCheckBox('с Word-отчётом'); self.p_with_report.setChecked(True)
        self.p_with_report.setToolTip('Самая долгая часть выгрузки: графики и Word по каждой территории')
        self.p_zip = QCheckBox('ZIP'); self.p_zip.setChecked(True)
        self.p_zip.setToolTip('Дополнительно сложить папку в архив рядом с ней')
        dele = QPushButton('Удалить партию'); dele.clicked.connect(self._delete_batch)
        self.p_bar = QProgressBar(); self.p_bar.setRange(0, 100); self.p_bar.setValue(0)
        self.p_bar.setTextVisible(True); self.p_bar.setFormat('%p%'); self.p_bar.setMaximumWidth(160)
        self.p_status = _status_label()
        ctl.addWidget(gen); ctl.addWidget(exp); ctl.addWidget(self.p_with_report); ctl.addWidget(self.p_zip)
        ctl.addWidget(dele); ctl.addWidget(self.p_bar); ctl.addWidget(self.p_status); ctl.addStretch(1)

        # модуль 3: содержание выбранной партии
        p3 = QVBoxLayout(); p3.setContentsMargins(0, 0, 0, 0)
        self.p_tabs = QTabWidget()
        cont = QWidget(); cl = QVBoxLayout(cont); cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(QLabel('Содержание выбранной партии (территории, прогнозы моделей, уточнённые MSE/R²):'))
        self.c_table = QTableWidget(); self.c_table.setMinimumHeight(100); cl.addWidget(self.c_table, 1)
        self.p_tabs.addTab(cont, 'Содержание партии')
        # план и итог: сколько посчитано из запланированного и почему остальное — нет
        outc = QWidget(); ol = QVBoxLayout(outc); ol.setContentsMargins(0, 0, 0, 0)
        self.o_summary = _note('Выберите партию в верхнем списке.')
        ol.addWidget(self.o_summary)
        osplit = QSplitter(Qt.Horizontal); osplit.setChildrenCollapsible(False)
        rbox = QWidget(); rl = QVBoxLayout(rbox); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel('Причины нерасчёта (сколько прогнозов):'))
        self.o_reasons = QTableWidget(); self.o_reasons.setEditTriggers(QTableWidget.NoEditTriggers)
        rl.addWidget(self.o_reasons, 1)
        fbox = QWidget(); fl = QVBoxLayout(fbox); fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel('Каждый нерассчитанный прогноз:'))
        self.o_failed = QTableWidget(); self.o_failed.setEditTriggers(QTableWidget.NoEditTriggers)
        fl.addWidget(self.o_failed, 1)
        osplit.addWidget(rbox); osplit.addWidget(fbox); osplit.setSizes([380, 620])
        ol.addWidget(osplit, 1)
        self.p_tabs.addTab(outc, 'План и итог')
        p3.addWidget(self.p_tabs, 1)
        self._batches = None; self._reload_batches()
        self.b_table.itemSelectionChanged.connect(self._show_batch_outcomes)
        return self._vsplit([self._panel(p1), self._panel(ctl), self._panel(p3)],
                            sizes=[220, 40, 320], scroll_first=False)

    def _reload_batches(self):
        df = S.list_forecast_batches(account_id=(self.acc['account_id'] if self.p_only_me.isChecked() else None))
        if df is not None and not df.empty:                # «рассчитано N из M» по учёту итогов
            oc = db.load(PF.OUTCOMES_TABLE)
            done = {}
            if not oc.empty and 'batch_id' in oc:
                for bid, g in oc.groupby(oc['batch_id'].astype(str)):
                    done[bid] = f'{int((g["status"] == "ok").sum())} из {len(g)}'
            df = df.copy()
            df['рассчитано'] = df['batch_id'].astype(str).map(done).fillna('—')
        self._batches = df
        cols = [c for c in ['когда', 'автор', 'культура', 'территорий', 'рассчитано', 'прогнозов',
                            'комментарий', 'batch_id']
                if df is not None and not df.empty and c in df]
        df_to_table(self.b_table, df[cols] if (df is not None and not df.empty) else df)

    def _show_batch_outcomes(self):
        """План и итог выбранной партии: сколько рассчитано и почему не рассчитано остальное."""
        row = self.b_table.currentRow()
        if row < 0 or self._batches is None or self._batches.empty or row >= len(self._batches):
            return
        bid = str(self._batches.iloc[row]['batch_id'])
        summary, reasons, failed = PF.batch_outcomes(bid)
        self.o_summary.setText(PF.outcome_text(summary))
        df_to_table(self.o_reasons, reasons)
        df_to_table(self.o_failed, failed, maxrows=5000)
        n_failed = 0 if not summary else summary['failed']
        self.p_tabs.setTabText(1, f'План и итог ({n_failed} не рассчитано)' if summary else 'План и итог')

    def _make_batch_report(self):
        row = self.b_table.currentRow()
        if row < 0 or self._batches is None or self._batches.empty:
            QMessageBox.information(self, 'Отчёт', 'Выберите партию прогноза в верхнем списке.'); return
        rec = self._batches.iloc[row]
        bid = str(rec['batch_id'])
        depth = self.p_depth.value() or None

        parent = QFileDialog.getExistingDirectory(self, 'Куда сохранить отчёт (выберите папку)',
                                                  config.REPORTS_DIR)
        if not parent:
            return
        default_name = R._safe_dir(f'Прогноз {rec.get("культура", "")} {bid}')
        folder, ok = QInputDialog.getText(self, 'Имя папки отчёта',
                                          'Папка будет создана внутри выбранного каталога:',
                                          text=default_name)
        if not ok or not folder.strip():
            return
        outdir = os.path.join(parent, R._safe_dir(folder))

        def job():
            return R.batch_report(bid, last_k=depth, outdir=outdir)
        self.p_status.setText('считаю содержание и уточнённые метрики…')
        self._worker = Worker(job)
        self._worker.done.connect(self._batch_report_done)
        self._worker.failed.connect(lambda tb: (self.p_status.setText('ошибка'),
                                                QMessageBox.critical(self, 'Ошибка', tb[-1500:])))
        self._worker.start()

    def _export_batch(self):
        """Выгрузить партию как самостоятельную папку-артефакт (таблицы + метаданные + отчёт)."""
        row = self.b_table.currentRow()
        if row < 0 or self._batches is None or self._batches.empty:
            QMessageBox.information(self, 'Выгрузка', 'Выберите партию прогноза в верхнем списке.'); return
        rec = self._batches.iloc[row]
        bid = str(rec['batch_id'])
        depth = self.p_depth.value() or None
        with_report = self.p_with_report.isChecked()
        make_zip = self.p_zip.isChecked()

        parent = QFileDialog.getExistingDirectory(self, 'Куда выгрузить прогноз (выберите папку)',
                                                  X.exports_dir())
        if not parent:
            return
        default_name = X.suggest_folder_name(bid, rec.get('культура'), rec.get('когда'))
        folder, ok = QInputDialog.getText(self, 'Имя папки выгрузки',
                                          'Папка будет создана внутри выбранного каталога:',
                                          text=default_name)
        if not ok or not folder.strip():
            return

        def job(prog):
            return X.export_batch(bid, out_parent=parent, folder_name=folder.strip(), last_k=depth,
                                  with_report=with_report, make_zip=make_zip,
                                  exported_by=self.acc.get('name'),
                                  progress=prog, message=self._worker.message.emit)

        self.p_bar.setValue(0); self.p_status.setText('выгружаю…')
        self._worker = Worker(job, with_progress=True)
        self._worker.progress.connect(self.p_bar.setValue)
        self._worker.message.connect(self.p_status.setText)
        self._worker.done.connect(self._export_done)
        self._worker.failed.connect(lambda tb: (self.p_status.setText('ошибка'),
                                                QMessageBox.critical(self, 'Ошибка выгрузки', tb[-1500:])))
        self._worker.start()

    def _export_done(self, res):
        self.p_bar.setValue(100)
        df_to_table(self.c_table, res['table'])
        size = sum(f.get('bytes', 0) for f in res['files'])
        self.p_status.setText(f'выгружено: {os.path.basename(res["dir"])}')
        lines = [f'Папка: {res["dir"]}',
                 f'Файлов: {len(res["files"])} ({size / 1048576:.1f} МБ)']
        if res.get('zip'):
            lines.append(f'Архив для отправки: {res["zip"]}')
        lines += ['', 'Открыть папку?']
        if QMessageBox.question(self, 'Выгрузка готова', '\n'.join(lines)) != QMessageBox.Yes:
            return
        try:
            os.startfile(res['dir'])
        except Exception:                                  # noqa: BLE001
            pass

    def _batch_report_done(self, res):
        xlsx_path, docx_path, content = res
        df_to_table(self.c_table, content)
        self.p_status.setText('готово')
        if QMessageBox.question(self, 'Отчёт готов',
                                f'Excel: {xlsx_path}\nWord: {docx_path}\n\nОткрыть Word-отчёт?') != QMessageBox.Yes:
            return
        try:
            os.startfile(docx_path)
        except Exception as exc:                           # noqa: BLE001
            try:
                os.startfile(os.path.dirname(docx_path))
            except Exception:                              # noqa: BLE001
                pass
            QMessageBox.information(self, 'Открыть вручную',
                                    f'Не удалось открыть автоматически ({exc}).\n{docx_path}')

    def _delete_batch(self):
        row = self.b_table.currentRow()
        if row < 0 or self._batches is None or self._batches.empty:
            QMessageBox.information(self, 'Удаление', 'Выберите партию прогноза в списке.'); return
        rec = self._batches.iloc[row]; bid = str(rec['batch_id'])
        if bid == '—':
            QMessageBox.information(self, 'Удаление',
                                    'Это сборная группа старых прогнозов без id партии — удалить её целиком нельзя.')
            return
        if QMessageBox.question(
                self, 'Удалить партию',
                f'Удалить партию {bid} (культура: {rec.get("культура", "")}, '
                f'территорий: {rec.get("территорий", "?")})?\nДействие необратимо.') != QMessageBox.Yes:
            return
        n_pred, n_met = db.delete_batch(bid)
        self._reload_batches()
        self.c_table.setRowCount(0); self.c_table.setColumnCount(0)
        self.p_status.setText(f'удалено: прогнозов {n_pred}, метрик {n_met}')

    def _logout(self):
        A.logout()
        QMessageBox.information(self, 'Выход', 'Вы вышли из аккаунта. Приложение закроется.')
        self.close()


def main():
    config.ensure_workspace()                          # первый запуск: каталоги + справочники по умолчанию
    db.ensure_user_tables()                            # таблица ручного ввода видна сразу, до первой записи
    db.convert_csv_dialect()                           # таблицы прежнего формата -> «;» (читаются в Excel)
    try:                                               # чтобы своя иконка показывалась и на панели задач
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('NerualRabbit.Forecast.App')
    except Exception:                                  # noqa: BLE001
        pass
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    app_theme.apply(app, app_theme.current_theme())    # оформление — до создания любых окон
    if not A.is_logged_in():
        dlg = LoginDialog(); app_theme.mark_buttons(dlg)
        if dlg.exec() != QDialog.Accepted:
            return
    win = MainWindow(); app_theme.mark_buttons(win); win.show()
    win.offer_first_run()                              # пустая база -> предложить каталог
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
