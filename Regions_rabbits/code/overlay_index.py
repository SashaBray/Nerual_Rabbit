"""Индекс смещений оверлея скачанных рядов — чтение по одному ряду вместо всего файла.

Оверлей ``time_series_downloaded.csv`` — это append-лог: каждый скачанный ряд
дописывается в конец ОДНИМ непрерывным блоком строк (см. :func:`db.put_time_series`).
Раньше ``db._downloads()`` при первом же чтении разбирал ВЕСЬ файл в словарь
``(tid, parm, year) -> [[day, value], ...]``. Для оверлея на 3.7 ГБ (92.9 млн точек,
83 тыс. рядов) это ~2-4 минуты и ~11 ГБ RAM — то есть первый прогноз/отчёт в сессии
фактически не завершался.

Здесь вместо этого ОДИН раз строится индекс смещений

    (territory_id, parm, year) -> (offset, nbytes, npoints, vega_uid)

который сохраняется рядом с оверлеем (``*.index.csv`` + ``*.index.json``). Дальше
любой ряд читается одним ``seek``+``read`` (десятки КБ) и разбирается в
``[[day, value], ...]``; разобранные ряды держатся в LRU-кэше с бюджетом в точках.

Индекс — производный кэш, а не источник данных:

* файл вырос (дописали ряды)  -> досканировать только хвост;
* файл переписан/уменьшился (``purge_overlay.py``) -> пересобрать целиком;
* индекс не читается/битый    -> пересобрать целиком.

``OverlayIndex`` реализует протокол ``Mapping``, поэтому объект подставляется вместо
прежнего словаря: ``key in idx``, ``len(idx)``, ``idx.keys()``, ``idx.get(key)``
работают как раньше (``__contains__`` и ``keys`` — только по индексу, без чтения
данных; ``get``/``items``/``values`` читают точки с диска).

Порядок точек. Блок ряда в файле уже отсортирован по дню (``put_time_series``
сортирует перед записью), а разбор дополнительно применяет УСТОЙЧИВУЮ сортировку по
дню. Поэтому порядок точек внутри одного дня — исходный порядок скачивания
(хронологический). Это важно: Вега отдаёт часть параметров (mean_temp, mean_rh, ...)
4 раза в сутки, а ``put_time_series`` теряет дробную часть номера дня, так что в
файле лежат 4 точки с одинаковым ``day``, и порядок внутри дня влияет на
интерполяцию в ``TimeRow.get_array_by_size``.
"""

import csv
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from operator import itemgetter


# Порядок столбцов оверлея (задаётся db.put_time_series)
HEADER = 'territory_id,parm,year,vega_uid,day,value'

# Конец строки при дозаписи. Прежде оверлей писался ``pandas.to_csv(mode='a')``, а у него
# ``lineterminator`` по умолчанию — ``os.linesep`` (на Windows CRLF). Пишем так же, чтобы
# дописанные байты были ровно такими же, как раньше, а файл не стал смешанным.
EOL = os.linesep

# Столбцы файла индекса
INDEX_HEADER = ['territory_id', 'parm', 'year', 'vega_uid', 'offset', 'nbytes', 'npoints']

_SCAN_CHUNK = 1 << 24          # 16 МБ на чтение при сканировании
_CACHE_POINTS = 1_000_000      # бюджет LRU-кэша разобранных рядов, в точках (~120 МБ)
_COMMA = ord(',')


def index_paths(csv_path):
    """Пути производных файлов индекса для оверлея ``csv_path``."""
    base = csv_path[:-4] if csv_path.endswith('.csv') else csv_path
    return base + '.index.csv', base + '.index.json'


def _fmt_value(value):
    """Значение в том же виде, в каком его писал pandas.to_csv (NaN -> пусто)."""
    value = float(value)
    return '' if value != value else repr(value)


def _third_comma(line):
    """Позиция третьей запятой в строке (конец ключа ``tid,parm,year``); -1 если нет."""
    i = line.find(_COMMA)
    if i < 0:
        return -1
    i = line.find(_COMMA, i + 1)
    if i < 0:
        return -1
    return line.find(_COMMA, i + 1)


def _parse_block(raw):
    """Разобрать байты блока строк оверлея в ``[[day, value], ...]`` по возрастанию дня."""
    out = []
    for line in raw.split(b'\n'):
        if len(line) < 6:                       # пустая строка / одинокий '\r'
            continue
        i3 = _third_comma(line)
        if i3 < 0:
            continue
        i4 = line.find(_COMMA, i3 + 1)          # после vega_uid
        i5 = line.find(_COMMA, i4 + 1)          # после day
        if i4 < 0 or i5 < 0:
            continue
        day = line[i4 + 1:i5]
        value = line[i5 + 1:].rstrip()          # rstrip убирает '\r' при CRLF
        try:
            day = int(day)
        except ValueError:
            day = int(float(day))
        out.append([day, float(value) if value else float('nan')])
    out.sort(key=itemgetter(0))                 # Timsort устойчив: порядок внутри дня сохраняется
    return out


class OverlayIndex(Mapping):
    """Ленивый доступ к рядам оверлея: индекс смещений + LRU-кэш разобранных рядов.

    Правило при повторной загрузке одного ряда (в файле появляется второй блок с тем
    же ключом): ПОБЕЖДАЕТ ПОСЛЕДНИЙ блок — так же, как в памяти вело себя прежнее
    ``_downloads()[key] = pts``. Старые байты остаются в append-логе мёртвым грузом
    до следующей перезаписи файла (``purge_overlay.py``).

    Объект потокобезопасен: GUI считает в рабочем потоке (``QThread``), пока главный
    поток тоже читает БД, а у нас общий файловый дескриптор с ``seek`` — все обращения
    к нему, к индексу и к кэшу идут под одним реентерабельным замком.
    """

    def __init__(self, csv_path, cache_points=_CACHE_POINTS, verbose=True):
        self.csv_path = csv_path
        self.index_path, self.meta_path = index_paths(csv_path)
        self.cache_points = int(cache_points)
        self.verbose = verbose
        self._entries = None            # key -> (offset, nbytes, npoints, uid|None)
        self._covered = 0               # сколько байт оверлея описано индексом
        self._cache = OrderedDict()     # key -> [[day, value], ...] (LRU)
        self._cached_points = 0
        self._fh = None                 # открытый на чтение оверлей
        self._lock = threading.RLock()  # общий дескриптор + индекс + кэш

    # ------------------------------------------------------------------ #
    # Протокол Mapping (подстановка вместо прежнего словаря _overlay['dl'])
    # ------------------------------------------------------------------ #
    def __contains__(self, key):
        """Есть ли ряд — только по индексу, без чтения точек."""
        return key in self._index()

    def __len__(self):
        return len(self._index())

    def __iter__(self):
        return iter(self._index())

    def keys(self):
        return self._index().keys()

    def __getitem__(self, key):
        pts = self.points(key)
        if pts is None:
            raise KeyError(key)
        return pts

    # ------------------------------------------------------------------ #
    # Чтение
    # ------------------------------------------------------------------ #
    def points(self, key):
        """Точки ряда ``[[day, value], ...]`` либо ``None``, если ряда нет в оверлее."""
        with self._lock:
            entry = self._index().get(key)
            if entry is None:
                return None
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
            offset, nbytes, _npoints, _uid = entry
            fh = self._reader()
            fh.seek(offset)
            raw = fh.read(nbytes)
        pts = _parse_block(raw)                 # разбор — вне замка
        with self._lock:
            self._cache_put(key, pts)
        return pts

    def uids(self):
        """Провенанс: ``(tid, parm, year) -> vega_uid`` (только там, где он записан)."""
        return {k: e[3] for k, e in self._index().items() if e[3] is not None}

    def stats(self):
        """Сводка по индексу (для диагностики)."""
        entries = self._index()
        return {'series': len(entries),
                'points': sum(e[2] for e in entries.values()),
                'covered_bytes': self._covered,
                'cached_series': len(self._cache),
                'cached_points': self._cached_points}

    # ------------------------------------------------------------------ #
    # Запись (append-лог + поддержание индекса)
    # ------------------------------------------------------------------ #
    def append_series(self, key, points, vega_uid=None):
        """Дописать ряд в оверлей и обновить индекс (offset берётся из файла).

        Байты пишутся ровно в том же формате, что раньше писал ``pandas.to_csv``,
        поэтому файл остаётся совместимым (``purge_overlay``, ``scan_overlay_corrupt``).
        """
        tid, parm, year = str(key[0]), str(key[1]), int(key[2])
        uid_txt = '' if vega_uid is None or vega_uid == '' else str(int(vega_uid))
        prefix = f'{tid},{parm},{year},{uid_txt},'
        raw = ''.join(f'{prefix}{int(d)},{_fmt_value(v)}{EOL}' for d, v in points).encode('utf-8')
        uid = None if uid_txt == '' else int(uid_txt)

        with self._lock:
            entries = self._index()
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            with open(self.csv_path, 'ab') as fh:
                fh.seek(0, os.SEEK_END)
                if fh.tell() == 0:
                    fh.write((HEADER + EOL).encode('utf-8'))
                offset = fh.tell()
                fh.write(raw)

            entries[key] = (offset, len(raw), len(points), uid)
            self._covered = offset + len(raw)
            self._cache_put(key, [[int(d), float(v)] for d, v in points])
            self._append_index_rows([(key, entries[key])])
            self._write_meta()

    def invalidate(self):
        """Забыть индекс и удалить его файлы (после внешней перезаписи оверлея)."""
        with self._lock:
            self._invalidate()

    def _invalidate(self):
        self.close()
        self._entries = None
        self._covered = 0
        self._cache.clear()
        self._cached_points = 0
        for path in (self.index_path, self.meta_path):
            if os.path.exists(path):
                os.remove(path)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------ #
    # Внутреннее: построение / загрузка индекса
    # ------------------------------------------------------------------ #
    def _reader(self):
        if self._fh is None:
            self._fh = open(self.csv_path, 'rb')
        return self._fh

    def _cache_put(self, key, pts):
        self._cache[key] = pts
        self._cached_points += len(pts)
        while self._cached_points > self.cache_points and len(self._cache) > 1:
            _k, old = self._cache.popitem(last=False)
            self._cached_points -= len(old)

    def _index(self):
        """Индекс (строится/подгружается лениво, при необходимости достраивается)."""
        with self._lock:
            return self._build_index()

    def _build_index(self):
        if self._entries is not None:
            return self._entries

        self._entries = {}
        self._covered = 0
        if not os.path.exists(self.csv_path):
            return self._entries
        size = os.path.getsize(self.csv_path)
        if size == 0:
            return self._entries

        covered = self._load_saved(size)
        if covered >= size:
            self._covered = covered
            return self._entries

        if covered == 0:
            self._entries.clear()
        new_rows = self._scan(covered, size)
        self._covered = size
        if covered == 0:
            self._write_index_rows(sorted(self._entries.items(), key=lambda kv: kv[1][0]))
        else:
            self._append_index_rows(new_rows)
        self._write_meta()
        return self._entries

    def _load_saved(self, size):
        """Прочитать сохранённый индекс. Возвращает число описанных им байт (0 — негоден)."""
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return 0
        try:
            with open(self.meta_path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return 0
        covered = int(meta.get('covered_bytes', 0))
        if meta.get('header') != HEADER or not 0 < covered <= size:
            return 0                      # другой формат оверлея или файл усох -> пересборка

        entries = {}
        try:
            with open(self.index_path, 'r', encoding='utf-8', newline='') as fh:
                reader = csv.reader(fh)
                head = next(reader, None)
                if head != INDEX_HEADER:
                    return 0
                for row in reader:        # дубликаты ключей: побеждает последняя строка
                    if len(row) != 7:
                        return 0
                    uid = row[3]
                    entries[(row[0], row[1], int(row[2]))] = (
                        int(row[4]), int(row[5]), int(row[6]), None if uid == '' else int(uid))
        except (OSError, ValueError):
            return 0
        if not entries:
            return 0
        if not self._spot_check(entries, covered):
            return 0

        self._entries = entries
        return covered

    def _spot_check(self, entries, covered):
        """Проверить, что байты оверлея на месте: у крайних записей ключ совпадает."""
        by_offset = sorted(entries.items(), key=lambda kv: kv[1][0])
        try:
            fh = self._reader()
            for key, (offset, nbytes, _n, _uid) in (by_offset[0], by_offset[-1]):
                if offset + nbytes > covered:
                    return False
                fh.seek(offset)
                head = fh.read(min(nbytes, 4096)).split(b'\n', 1)[0]
                i3 = _third_comma(head)
                if i3 < 0:
                    return False
                want = f'{key[0]},{key[1]},{key[2]}'.encode('utf-8')
                if head[:i3] != want:
                    return False
        except OSError:
            return False
        return True

    def _scan(self, start, size):
        """Просканировать оверлей с байта ``start``, заполнив индекс. -> список новых записей."""
        entries = self._entries
        new_rows = []
        label = 'построение' if start == 0 else 'дополнение'
        if self.verbose:
            print(f'Индекс оверлея: {label} по {(size - start) / 1e6:.0f} МБ '
                  f'({os.path.basename(self.csv_path)})...', flush=True)
        t0 = time.time()

        fh = open(self.csv_path, 'rb')
        try:
            if start == 0:
                header = fh.readline()          # шапку не индексируем
                pos = len(header)
            else:
                pos = start
                fh.seek(pos)

            def flush(key, offset, nbytes, npoints, uid):
                entry = (offset, nbytes, npoints, uid)
                entries[key] = entry
                new_rows.append((key, entry))

            ckey = None
            coff = cn = 0
            cuid = None
            tail = b''
            next_report = t0 + 15.0
            while True:
                chunk = fh.read(_SCAN_CHUNK)
                if not chunk:
                    break
                if tail:
                    chunk = tail + chunk
                lines = chunk.split(b'\n')
                tail = lines.pop()              # последняя строка может быть неполной
                for line in lines:
                    step = len(line) + 1
                    if len(line) < 6:
                        pos += step
                        continue
                    i3 = _third_comma(line)
                    if i3 < 0:
                        pos += step
                        continue
                    key = line[:i3]
                    if key != ckey:
                        if ckey is not None:
                            flush(_key_of(ckey), coff, pos - coff, cn, cuid)
                        i4 = line.find(_COMMA, i3 + 1)
                        uid = line[i3 + 1:i4] if i4 > 0 else b''
                        ckey, coff, cn = key, pos, 0
                        cuid = int(float(uid)) if uid else None
                    cn += 1
                    pos += step
                if self.verbose and time.time() > next_report:
                    print(f'  ...{(pos - start) / 1e6:.0f} / {(size - start) / 1e6:.0f} МБ, '
                          f'рядов {len(entries)}', flush=True)
                    next_report = time.time() + 15.0

            if len(tail) >= 6:                  # файл без завершающего перевода строки
                i3 = _third_comma(tail)
                if i3 >= 0:
                    key = tail[:i3]
                    if key != ckey:
                        if ckey is not None:
                            flush(_key_of(ckey), coff, pos - coff, cn, cuid)
                        i4 = tail.find(_COMMA, i3 + 1)
                        uid = tail[i3 + 1:i4] if i4 > 0 else b''
                        ckey, coff, cn = key, pos, 0
                        cuid = int(float(uid)) if uid else None
                    cn += 1
                    pos += len(tail)
            if ckey is not None:
                flush(_key_of(ckey), coff, pos - coff, cn, cuid)
        finally:
            fh.close()

        if self.verbose:
            print(f'Индекс оверлея: рядов {len(entries)}, '
                  f'{time.time() - t0:.1f} с -> {os.path.basename(self.index_path)}', flush=True)
        return new_rows

    # ------------------------------------------------------------------ #
    # Внутреннее: файлы индекса
    # ------------------------------------------------------------------ #
    def _write_index_rows(self, items):
        tmp = self.index_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(INDEX_HEADER)
            writer.writerows(_index_row(key, entry) for key, entry in items)
        os.replace(tmp, self.index_path)

    def _append_index_rows(self, items):
        if not items:
            return
        new_file = not os.path.exists(self.index_path) or os.path.getsize(self.index_path) == 0
        with open(self.index_path, 'a', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(INDEX_HEADER)
            writer.writerows(_index_row(key, entry) for key, entry in items)

    def _write_meta(self):
        meta = {'source': os.path.basename(self.csv_path),
                'header': HEADER,
                'covered_bytes': self._covered,
                'series': len(self._entries or {}),
                'built_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        tmp = self.meta_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.meta_path)


def _key_of(key_bytes):
    """``b'13_1337,mean_temp,2015'`` -> ``('13_1337', 'mean_temp', 2015)``."""
    tid, parm, year = key_bytes.decode('utf-8').split(',')
    return tid, parm, int(year)


def _index_row(key, entry):
    offset, nbytes, npoints, uid = entry
    return [key[0], key[1], key[2], '' if uid is None else uid, offset, nbytes, npoints]
