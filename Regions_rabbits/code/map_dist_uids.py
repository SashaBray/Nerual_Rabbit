"""Сопоставление наших территорий с АКТУАЛЬНЫМИ районными uid Веги.

Вход:  ``workspace/database/vega_districts.csv`` (скан: uid, region, district) —
       см. :mod:`scan_dist_uids`. Текущий uid района = (uid региона)*1000 + номер,
       поэтому ``uid // 1000`` совпадает с нашим ``id_region``.
Логика: для каждой нашей территории берём районы скана того же региона
       (``uid//1000 == id_region``) и подбираем по названию (difflib, с нормализацией
       «муниципальный/район/город»).
Выход: ``workspace/database/territory_uids.csv`` — ``territory_id, vega_uid,
       scan_district, similarity``.

Запуск:  ``python code/map_dist_uids.py``
"""

import collections
import csv
import difflib
import os
import re

import pandas as pd

import config
import db


def _safe_to_csv(df, path):
    """Записать CSV; если файл занят (открыт в Excel) — предупредить, но не падать."""
    try:
        df.to_csv(path, index=False, encoding='utf-8-sig')
        return True
    except PermissionError:
        print(f'  ! не удалось записать {os.path.basename(path)} — файл занят (открыт в Excel?). Пропускаю.')
        return False


# Родовые слова, одинаковые у разных районов — убираем по ГРАНИЦЕ слова (\b),
# чтобы не покалечить названия типа «Городовиковский» (в нём 'город' — не отдельное слово).
_DROP_WORDS = [
    'муниципальный', 'муниципальное', 'муниципального', 'образование',
    'городской', 'городское', 'сельское', 'сельский', 'поселение', 'поселок', 'пгт', 'рп',
    'район', 'районный', 'округ', 'город', 'зато', 'имени', 'им',
    'область', 'край', 'края', 'республика', 'автономный', 'автономная', 'автономное', 'аобл',
]
_DROP_RE = re.compile(r'\b(' + '|'.join(_DROP_WORDS) + r')\b')

# Прилагательные окончания района (одинаковые у разных названий) -> приводим к основе,
# чтобы «Чеховский»↔«Чехов», «Котласский»↔«Котлас», «Клинский»↔«Клин» совпадали:
# Вега часто называет район по центру в им. падеже, а БДПМО — прилагательным.
_ADJ_SUFFIXES = ('овский', 'евский', 'инский', 'ынский',
                 'цкий', 'цкая', 'цкое', 'ский', 'ская', 'ской', 'ское')


def _norm(name):
    """Каноничный ключ названия: ё→е, убрать скобки/родовые слова, привести к основе."""
    s = str(name).lower().replace('ё', 'е')
    s = re.sub(r'\(.*?\)', ' ', s)        # содержимое скобок
    s = _DROP_RE.sub(' ', s)              # родовые слова по границам
    s = re.sub(r'[^а-яa-z0-9]', '', s)    # оставить только буквы/цифры
    for suf in _ADJ_SUFFIXES:             # срезать прилагательное окончание к основе
        if s.endswith(suf) and len(s) - len(suf) >= 4:
            s = s[:-len(suf)]
            break
    return re.sub(r'[оаеияй]$', '', s)    # затем одиночную хвостовую гласную (Ступино→ступин)


def _is_rayon(name):
    """Тип объекта: True — район/округ (с/х земли), False — город (застройка).

    Нужно, чтобы при одинаковой основе («Джанкой» город vs «Джанкойский район»)
    наш район предпочитал вегинский РАЙОН, а не город.
    """
    s = str(name).lower()
    if 'район' in s or 'округ' in s:
        return True
    if s.startswith('город') or 'г.' in s:
        return False
    return False        # голое имя (Джанкой, Симферополь) считаем городом


# Алиас полосы региона: наш id_region -> region_uid Веги, когда они НЕ совпадают.
# Крым: у нас 1501, у Веги 101 (районы 101001…). Дополняйте при необходимости.
_REGION_UID_ALIAS = {1501: 101}

REPORT_NAME = 'district_match.csv'          # постоянный отчёт (оператор правит колонку error)
REV_REPORT_NAME = 'vega_unmatched.csv'      # обратный отчёт (районы Веги)


def _load_operator_marks(path):
    """Прочитать прошлый отчёт: (frozen_rows, rejected).

    frozen_rows — строки, где оператор поставил error=1 (его память; сохраняем дословно и
    оставляем в отчёте). rejected[territory_id] = множество забракованных vega_uid — их
    программа больше НЕ предлагает этому району и пробует следующего кандидата выше порога.
    """
    frozen, rejected = [], collections.defaultdict(set)
    if not os.path.exists(path):
        return frozen, rejected
    try:
        df = pd.read_csv(path, dtype=str, encoding='utf-8-sig').fillna('')
    except Exception:                       # noqa: BLE001
        return frozen, rejected
    seen = set()
    for _, r in df.iterrows():
        if str(r.get('error', '')).strip() not in ('1', '1.0'):
            continue
        tid = str(r.get('territory_id', '')).strip()
        uid_raw = str(r.get('vega_uid', '')).strip()
        if not tid or not uid_raw:
            continue
        try:
            uid = int(float(uid_raw))
        except ValueError:
            continue
        if (tid, uid) in seen:
            continue
        seen.add((tid, uid))
        rejected[tid].add(uid)
        frozen.append({'territory_id': tid, 'region': str(r.get('region', '')),
                       'bdpmo_district': str(r.get('bdpmo_district', '')), 'vega_uid': uid,
                       'vega_district': str(r.get('vega_district', '')),
                       'similarity': r.get('similarity', ''), 'status': str(r.get('status', '')),
                       'error': 1})
    return frozen, rejected


def build(threshold=0.8):
    config.ensure_dirs()
    scan_path = os.path.join(config.DB_DIR, 'vega_districts.csv')
    if not os.path.exists(scan_path):
        print('Нет vega_districts.csv — сначала запустите scan_dist_uids.py')
        return

    # скан, сгруппированный по региону (uid//1000) + сведения по uid (регион/название)
    by_region = {}
    by_region_name = {}    # запасная группировка по ИМЕНИ региона (для Крыма и т.п.,
    #                        где наш id_region не равен vega_uid//1000)
    scan_uid_info = {}      # uid -> (region_name, vega_district)
    with open(scan_path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['status'] != 'ok':
                continue
            uid = int(r['uid'])
            by_region.setdefault(uid // 1000, []).append((uid, r['district']))
            by_region_name.setdefault(_norm(r.get('region', '')), []).append((uid, r['district']))
            scan_uid_info[uid] = (r.get('region', ''), r['district'])

    terr = db.load('territories')
    # сгруппировать наши районные территории по региону
    bdpmo_by_region = {}
    for _, t in terr.iterrows():
        if t['level'] != 'district':
            continue
        ru = db._norm_id(t['id_region'])
        if not ru.isdigit():
            continue
        bdpmo_by_region.setdefault(int(ru), []).append(t)
    bdpmo_region_uids = set(bdpmo_by_region)

    # отметки оператора из прошлого отчёта: забракованные пары не предлагаем повторно
    report_path = os.path.join(config.REPORTS_DIR, REPORT_NAME)
    frozen_rows, rejected = _load_operator_marks(report_path)

    rows = []           # рабочая карта (только сопоставленные)
    report_rows = []    # отчёт по ВСЕМ районам (включая несопоставленные)
    matched = unmatched = 0
    # внутри каждого региона — жадное ВЗАИМНО-ОДНОЗНАЧНОЕ сопоставление:
    # пары (похожесть, наш_район, uid_веги) сортируем по убыванию и забираем сверху,
    # каждый район Веги и каждый наш район используется НЕ БОЛЕЕ одного раза -> дублей нет.
    for ru, terrs in bdpmo_by_region.items():
        vega_ru = _REGION_UID_ALIAS.get(ru, ru)     # Крым: 1501 -> 101
        cands = by_region.get(vega_ru, [])
        if not cands:   # полоса пуста -> запасной матч по имени региона
            cands = by_region_name.get(_norm(terrs[0]['region']), [])
        cand_norm = [(uid, dname, _norm(dname), _is_rayon(dname)) for uid, dname in cands]
        best_raw = {}   # ti -> (score, uid, dname): ближайший кандидат (для отчёта о несопоставленных)
        elig = []       # (приоритет, похожесть, ti, uid, dname) только при похожести >= порога
        for ti, t in enumerate(terrs):
            tn = _norm(t['district'])
            o_rayon = _is_rayon(t['district'])
            rej = rejected.get(t['territory_id'], ())   # забраковано оператором
            br = (0.0, None, None)
            for uid, dname, dn, c_rayon in cand_norm:
                if uid in rej:
                    continue                            # не предлагаем повторно
                sc = difflib.SequenceMatcher(None, tn, dn).ratio()
                if sc > br[0]:
                    br = (sc, uid, dname)
                if sc >= threshold:
                    # типовой бонус: район↔район выше, район↔город ниже (Крым: и город,
                    # и район дают одну основу -> при ничьей берём именно РАЙОН, а не город)
                    bonus = 0.05 if o_rayon == c_rayon else -0.05
                    elig.append((sc + bonus, sc, ti, uid, dname))
            best_raw[ti] = br
        elig.sort(key=lambda x: -x[0])
        assign = {}                 # ti -> (uid, dname, score)
        used_t, used_uid = set(), set()
        for _prio, sc, ti, uid, dname in elig:
            if ti in used_t or uid in used_uid:
                continue
            assign[ti] = (uid, dname, round(sc, 3))   # в карту/отчёт — чистая похожесть
            used_t.add(ti)
            used_uid.add(uid)
        for ti, t in enumerate(terrs):
            if ti in assign:
                uid, dname, sc = assign[ti]
                rows.append({'territory_id': t['territory_id'], 'vega_uid': uid,
                             'scan_district': dname, 'similarity': sc})
                report_rows.append({
                    'territory_id': t['territory_id'], 'region': t.get('region', ''),
                    'bdpmo_district': t['district'], 'vega_uid': uid,
                    'vega_district': dname, 'similarity': sc, 'status': 'matched', 'error': 0})
                matched += 1
            else:
                sc, _uid, dname = best_raw[ti]   # ближайший (ниже порога или уже занят)
                report_rows.append({
                    'territory_id': t['territory_id'], 'region': t.get('region', ''),
                    'bdpmo_district': t['district'], 'vega_uid': '',
                    'vega_district': dname or '', 'similarity': round(sc, 3),
                    'status': 'unmatched', 'error': 0})
                unmatched += 1

    out = pd.DataFrame(rows, columns=['territory_id', 'vega_uid', 'scan_district', 'similarity'])
    db.save('territory_uids', out)

    # постоянный отчёт о сопоставлении (БДПМО -> Вега): новые активные строки (error=0) +
    # сохранённые строки оператора (error=1, его пометки о неверных сопоставлениях).
    # Худшие совпадения сверху -> оператору удобнее искать подозрительные.
    cols = ['territory_id', 'region', 'bdpmo_district', 'vega_uid',
            'vega_district', 'similarity', 'status', 'error']
    rep = pd.DataFrame(report_rows + frozen_rows, columns=cols)
    rep['_sim'] = pd.to_numeric(rep['similarity'], errors='coerce').fillna(-1.0)
    rep = rep.sort_values(['_sim', 'territory_id', 'error']).drop(columns='_sim').reset_index(drop=True)
    rep_path = report_path
    _safe_to_csv(rep, rep_path)
    n_rejected = sum(len(v) for v in rejected.values())

    # обратная проверка: районы Веги в наших регионах и сколько раз каждый сопоставлен
    # (times_matched == 0 -> не сопоставлен ни с одним БДПМО; > 1 -> дубль)
    assigned = collections.Counter(int(r['vega_uid']) for r in rows)
    # полосы Веги, соответствующие нашим регионам (с учётом алиаса: 1501 -> 101)
    vega_bands = {_REGION_UID_ALIAS.get(ru, ru) for ru in bdpmo_region_uids}
    rev_rows = []
    for uid, (reg_name, vdist) in scan_uid_info.items():
        if uid // 1000 not in vega_bands:
            continue
        rev_rows.append({'vega_uid': uid, 'region': reg_name, 'vega_district': vdist,
                         'times_matched': assigned.get(uid, 0)})
    rev = pd.DataFrame(rev_rows, columns=['vega_uid', 'region', 'vega_district', 'times_matched'])
    rev = rev.sort_values(['times_matched', 'region', 'vega_district']).reset_index(drop=True)
    rev_path = os.path.join(config.REPORTS_DIR, REV_REPORT_NAME)
    _safe_to_csv(rev, rev_path)

    print(f'Сопоставлено территорий: {matched}, не сопоставлено: {unmatched}')
    if n_rejected:
        print(f'Учтено пометок оператора (error=1): {n_rejected} забракованных пар — '
              f'предложены другие кандидаты выше порога либо район оставлен без пары.')
    print(f'Карта: {db.path_of("territory_uids")}')
    print(f'Отчёт (БДПМО->Вега): {rep_path}  '
          f'[поставьте 1 в колонке error у неверных строк — на следующем прогоне '
          f'программа попробует другой район]')
    print(f'Отчёт (обратный, районы Веги): {rev_path}  '
          f'[не сопоставлено: {int((rev.times_matched == 0).sum())}, '
          f'дублей: {int((rev.times_matched > 1).sum())}]')
    if matched:
        low = out[out['similarity'] < 0.8]
        print(f'Совпадений с similarity<0.8: {len(low)} (см. отчёт, отсортирован по возрастанию похожести)')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Сопоставление наших районов с текущими uid Веги')
    ap.add_argument('--threshold', type=float, default=0.8,
                    help='порог похожести (по умолчанию 0.8; ниже -> район остаётся несопоставленным)')
    build(threshold=ap.parse_args().threshold)
