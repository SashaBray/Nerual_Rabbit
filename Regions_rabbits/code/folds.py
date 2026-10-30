"""Фолды по годам и общая тестовая выборка — единое разбиение для всех опытов статьи.

Зачем. Прежде разбиение считалось случайно (``explore_models.split_indices``): соседние годы
одного района попадали и в обучение, и в тест, а у разных заданий тестовые выборки вообще не
совпадали — 652 примера против 687, разные районы и годы. Сравнивать метрики моделей в таких
условиях нельзя.

Здесь разбиение задаётся **годом**, одинаково для любого задания:

    фолд Y:  обучение — примеры с годом <= Y-2,  валидация — Y-1,  тест — Y

Будущее в обучение не попадает: пример года t опирается на ряды t-2…t, поэтому пример
обучения года Y-2 не видит ни одного дня тестового года.

Дополнительно задаётся **общая выборка пар «район, год»**: пересечение того, что есть во всех
сравниваемых заданиях. Тогда 12-канальная и 23-канальная сети меряются на одних и тех же
районах и годах, и их цифры сопоставимы.

Определение фолдов сохраняется в ``workspace/tasks/folds/<имя>.csv`` — чтобы опыт можно было
повторить и сослаться на него в статье.

    python code/folds.py --tasks winter_wheat_raw_v2 winter_wheat_all_v2 --name winter_wheat
"""
import argparse
import os

import numpy as np
import pandas as pd

import config

YIELD_MAX = 5000.0                                    # сентинел/выброс урожайности, как в обучении
FOLDS_DIR = os.path.join(config.TASKS_DIR, 'folds')


def load_plan(task):
    """План датасета с номером строки: territory_id, year, productive, row."""
    path = os.path.join(config.dataset_dir(task), 'plan.csv')
    if not os.path.exists(path):
        raise SystemExit(f'Нет плана датасета {task}: {path}')
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['row'] = np.arange(len(df))
    df['territory_id'] = df['territory_id'].astype(str)
    df['year'] = df['year'].astype(int)
    return df


def valid_mask(plan):
    """Пример годен: есть метка урожайности и она в разумных пределах (как при обучении)."""
    y = plan['productive'].to_numpy(dtype=float)
    return ~np.isnan(y) & (y >= 0) & (y <= YIELD_MAX)


def common_pairs(tasks):
    """Пары «район, год», которые есть во ВСЕХ перечисленных заданиях. -> DataFrame."""
    sets = []
    for t in tasks:
        plan = load_plan(t)
        ok = plan[valid_mask(plan)]
        sets.append(set(zip(ok['territory_id'], ok['year'])))
    common = set.intersection(*sets) if sets else set()
    out = pd.DataFrame(sorted(common), columns=['territory_id', 'year'])
    return out


def year_folds(pairs, test_years=None, gap=1):
    """Разбиение по годам. -> [{'fold', 'train_years', 'val_year', 'test_year'}, ...].

    ``gap=1`` — между обучением и тестом остаётся год валидации (обучение <= Y-2).
    Фолд берётся только если на обучение остаётся не меньше трёх лет.
    """
    years = sorted(pairs['year'].unique())
    folds = []
    for y in (test_years or years):
        val_y = y - gap
        train_years = [t for t in years if t <= y - gap - 1]
        if val_y not in years or len(train_years) < 3:
            continue
        folds.append({'fold': int(y), 'train_years': train_years,
                      'val_year': int(val_y), 'test_year': int(y)})
    return folds


def indices(task, fold, pairs=None):
    """Строки датасета ``task`` для фолда: (train, val, test) как массивы индексов.

    ``pairs`` — общая выборка (``common_pairs``); если задана, берутся только эти пары,
    поэтому разные задания меряются на одинаковых районах и годах.
    """
    plan = load_plan(task)
    plan = plan[valid_mask(plan)]
    if pairs is not None:
        keep = set(zip(pairs['territory_id'].astype(str), pairs['year'].astype(int)))
        plan = plan[[(t, y) in keep for t, y in zip(plan['territory_id'], plan['year'])]]
    tr = plan[plan['year'].isin(fold['train_years'])]['row'].to_numpy()
    va = plan[plan['year'] == fold['val_year']]['row'].to_numpy()
    te = plan[plan['year'] == fold['test_year']]['row'].to_numpy()
    return tr, va, te


def read_definition(name):
    """Прочитать сохранённую схему: {'name', 'folds': {год: {'роль': [пары]}}}. -> DataFrame."""
    path = os.path.join(FOLDS_DIR, f'{name}.csv')
    if not os.path.exists(path):
        raise SystemExit(f'Нет схемы фолдов «{name}»: {path}. Создайте её: python code/folds.py …')
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    df['territory_id'] = df['territory_id'].astype(str)
    df['year'] = df['year'].astype(int)
    return df


def fold_spec(scheme, fold):
    """Описание одного фолда: наборы пар «район, год» для каждой части.

    Роли заданы явно, пара за парой, поэтому схема описывает любое разбиение — по годам,
    по районам или по тому и другому сразу (см. ``ladder_definition``). Годы частей
    возвращаются для наглядности и для записи в метаданные модели.
    """
    key = str(fold)
    sub = scheme[scheme['fold'].astype(str) == key]
    if sub.empty:
        have = ', '.join(str(x) for x in sorted(scheme['fold'].astype(str).unique()))
        raise SystemExit(f'В схеме нет фолда «{fold}»; есть: {have}')
    part = lambda role: set(zip(sub[sub['role'] == role]['territory_id'],
                                sub[sub['role'] == role]['year']))
    train_pairs, val_pairs, test_pairs = part('train'), part('val'), part('test')
    years = lambda pp: sorted({int(y) for _t, y in pp})
    test_years = years(test_pairs)
    val_years = years(val_pairs)
    return {'fold': key, 'train_years': years(train_pairs),
            'val_year': int(val_years[0]) if val_years else None,
            'test_year': int(test_years[0]) if test_years else None,
            'train_pairs': train_pairs, 'val_pairs': val_pairs, 'test_pairs': test_pairs,
            'pairs': train_pairs | val_pairs | test_pairs}


def district_groups(pairs, plan, n_groups=2, strata_years=(2015, 2016, 2017), seed=42):
    """Раскидать районы по группам РОВНО по уровню урожайности. -> {район: номер группы}.

    Уровень считается по ранним годам, которые при любом режиме попадают в обучение, — чтобы
    тестовые годы не участвовали в построении самого разбиения. Районы сортируются по уровню
    и раздаются по группам «змейкой», поэтому в каждой группе поровну слабых, средних и
    сильных районов.
    """
    have = {t for t, _y in pairs}
    early = plan[plan['year'].isin(list(strata_years)) & plan['territory_id'].isin(have)]
    level = early.groupby('territory_id')['productive'].mean()
    rest = sorted(have - set(level.index))                  # нет ранних лет — раскидываем ровно
    rng = np.random.default_rng(seed)
    order = list(level.sort_values().index) 
    groups = {}
    for i, tid in enumerate(order):
        groups[tid] = i % n_groups
    for i, tid in enumerate(rng.permutation(rest)):
        groups[str(tid)] = i % n_groups
    return groups


def year_halves(pairs, plan, mode='interleaved'):
    """Две половины лет. -> ({год: половина}, средний уровень урожайности по годам).

    ``interleaved`` — годы сортируются по среднему уровню урожайности и раздаются «змейкой»
    (0,1,1,0,0,1,1,0), поэтому в каждую половину попадают и урожайные, и неурожайные годы, и
    средние уровни половин сходятся. Это нужно, чтобы требование «половины схожи по диапазону
    урожайности» выполнялось не только по районам, но и по годам.

    ``chrono`` — ранние годы против поздних. Так проверяется перенос ВПЕРЁД по времени, но
    уровни половин при этом не совпадают: поздние годы в нашем наборе хуже.
    """
    have = set(pairs)
    sub = plan[[(t, int(y)) in have for t, y in zip(plan['territory_id'], plan['year'])]]
    level = {int(y): float(v) for y, v in sub.groupby('year')['productive'].mean().items()}
    if mode == 'chrono':
        years = sorted(level)
        half = len(years) // 2
        return {y: (0 if i < half else 1) for i, y in enumerate(years)}, level
    order = sorted(level, key=lambda y: -level[y])
    return {y: (0 if i % 4 in (0, 3) else 1) for i, y in enumerate(order)}, level


def block_definition(pairs, plan, mode='interleaved', n_groups=2, seed=42):
    """Разбиение блоками: группы районов × половины лет. -> (DataFrame ролей, сведения).

    Решётка 2×2: две группы районов (ровные по уровню урожайности) на две половины лет
    (тоже ровные). Обучение берётся из одного блока решётки, тест — из блока, который не
    пересекается с ним НИ ПО РАЙОНАМ, НИ ПО ГОДАМ::

        половина лет →        h0        h1
        группа районов g0   [ блок A ] [ блок B ]
        группа районов g1   [ блок C ] [ блок D ]

    Диагональ A↔D и антидиагональ B↔C дают по два фолда каждая: блок по очереди служит
    обучением и тестом. Четыре фолда вместо двенадцати, каждый пример участвует, и
    появляется разброс, по которому видно, насколько одному числу можно верить.

    Валидация — один год обучающего блока, тот, чей уровень ближе всех к среднему по блоку.
    Год, а не случайная доля примеров: ранняя остановка тогда отбирает модель по переносу на
    НЕЗНАКОМЫЙ год — то же условие, что и на тесте. Случайная доля отбирала бы наоборот,
    поощряя запоминание года.
    """
    groups = district_groups(pairs, plan, n_groups, seed=seed)
    halves, level = year_halves(pairs, plan, mode)
    blocks = {}
    for (t, y) in pairs:
        g, h = groups.get(t), halves.get(int(y))
        if g is None or h is None:
            continue
        blocks.setdefault((g, h), set()).add((t, int(y)))

    rows = []
    for tag, dirs in (('d', [((0, 0), (1, 1)), ((1, 1), (0, 0))]),
                      ('a', [((0, 1), (1, 0)), ((1, 0), (0, 1))])):
        for i, (btr, bte) in enumerate(dirs):
            train_pairs, test_pairs = set(blocks.get(btr, ())), set(blocks.get(bte, ()))
            if not train_pairs or not test_pairs:
                continue
            tr_years = sorted({y for _t, y in train_pairs})
            mean_lv = np.mean([level[y] for y in tr_years])
            val_year = min(tr_years, key=lambda y: abs(level[y] - mean_lv))
            val_pairs = {(t, y) for (t, y) in train_pairs if y == val_year}
            train_pairs -= val_pairs
            fold = f'{mode[:5]}|{tag}{i}'
            for role, pp in (('train', train_pairs), ('val', val_pairs), ('test', test_pairs)):
                rows.extend({'fold': fold, 'territory_id': t, 'year': int(y), 'role': role}
                            for (t, y) in sorted(pp))
    return pd.DataFrame(rows), {'groups': groups, 'halves': halves, 'level': level}


def loo_definition(pairs, plan, n_groups=2, seed=42):
    """Один год из всех × группа районов. -> (DataFrame ролей, сведения).

    Требование то же, что у ``block_definition``: обучение и тест не пересекаются ни по
    районам, ни по годам. Разница в том, как резать год. Решётка 2×2 отдаёт тесту половину
    лет, и обучению остаётся всего три года (четвёртый уходит в валидацию). Но эффективный
    размер выборки определяется почти целиком числом ЛЕТ: район-годы одного года — это
    примерно одно независимое наблюдение. Поэтому выгоднее изымать по ОДНОМУ году::

        тест   = группа g, год Y                      (районы чужие, год чужой)
        обуч   = группа 1-g, все годы кроме Y и года валидации
        валид  = группа 1-g, один год, не Y

    Фолдов получается «лет × групп» = 16 вместо четырёх, обучающих лет шесть вместо трёх.
    Цена: тест каждого фолда — один год, то есть один розыгрыш уровня года, и по одному
    фолду судить нельзя. Судить надо по среднему и разбросу всех шестнадцати, зато разброс
    тогда измеряется по восьми РАЗНЫМ годам, а не по двум половинам.
    """
    groups = district_groups(pairs, plan, n_groups, seed=seed)
    have = set(pairs)
    sub = plan[[(t, int(y)) in have for t, y in zip(plan['territory_id'], plan['year'])]]
    level = {int(y): float(v) for y, v in sub.groupby('year')['productive'].mean().items()}
    by_group = {}
    for (t, y) in pairs:
        g = groups.get(t)
        if g is not None and int(y) in level:
            by_group.setdefault(g, set()).add((t, int(y)))

    rows = []
    for test_year in sorted(level):
        for g in sorted(by_group):
            test_pairs = {(t, y) for (t, y) in by_group[g] if y == test_year}
            pool = {(t, y) for gg in by_group if gg != g for (t, y) in by_group[gg]
                    if y != test_year}
            if not test_pairs or not pool:
                continue
            pool_years = sorted({y for _t, y in pool})
            mean_lv = np.mean([level[y] for y in pool_years])
            val_year = min(pool_years, key=lambda y: abs(level[y] - mean_lv))
            val_pairs = {(t, y) for (t, y) in pool if y == val_year}
            train_pairs = pool - val_pairs
            fold = f'{test_year}|g{g}'
            for role, pp in (('train', train_pairs), ('val', val_pairs), ('test', test_pairs)):
                rows.extend({'fold': fold, 'territory_id': t, 'year': int(y), 'role': role}
                            for (t, y) in sorted(pp))
    return pd.DataFrame(rows), {'groups': groups, 'level': level}


def ladder_definition(pairs, plan, test_years, n_groups=2, gap=1, seed=42):
    """«Лесенка утечек»: четыре режима обучения на ОДНОМ И ТОМ ЖЕ тесте. -> DataFrame ролей.

    Тест фолда — пары (районы тестовой группы × тестовый год). Меняется только то, что
    разрешено брать в обучение::

        random   — любые пары, кроме теста (как в прежних опытах: район знаком, год знаком)
        district — другие районы, любые годы, включая тестовый   (район незнаком, год знаком)
        year     — любые районы, но только годы <= Y-2           (район знаком, год незнаком)
        double   — другие районы И только годы <= Y-2            (незнакомо ни то, ни другое)

    Разница между режимами и есть вклад каждого вида утечки.
    """
    groups = district_groups(pairs, plan, n_groups, seed=seed)
    years = sorted({int(y) for _t, y in pairs})
    by_year = {}
    for t, y in pairs:
        by_year.setdefault(int(y), set()).add(t)

    rows = []
    for gi, test_year in enumerate(test_years):
        if test_year not in years:
            continue
        val_year = test_year - gap
        train_years = [y for y in years if y <= test_year - gap - 1]
        if val_year not in years or len(train_years) < 3:
            continue
        g_test = gi % n_groups                              # тестовая группа меняется по годам
        test_pairs = {(t, y) for (t, y) in pairs
                      if int(y) == test_year and groups.get(t) == g_test}
        other = lambda t: groups.get(t) != g_test
        regimes = {
            'random':   ({(t, y) for (t, y) in pairs} - test_pairs, None),
            'district': ({(t, y) for (t, y) in pairs if other(t)}, None),
            'year':     ({(t, y) for (t, y) in pairs if int(y) in train_years}, None),
            'double':   ({(t, y) for (t, y) in pairs if other(t) and int(y) in train_years}, None),
        }
        for name, (train_pairs, _x) in regimes.items():
            if name in ('random', 'district'):
                val_pairs = {(t, y) for (t, y) in train_pairs if int(y) == val_year}
            else:
                val_pairs = {(t, y) for (t, y) in pairs if int(y) == val_year
                             and (name == 'year' or other(t))}
            train_pairs = train_pairs - val_pairs - test_pairs
            fold = f'{test_year}|{name}'
            for role, pp in (('train', train_pairs), ('val', val_pairs), ('test', test_pairs)):
                rows.extend({'fold': fold, 'territory_id': t, 'year': int(y), 'role': role}
                            for (t, y) in sorted(pp))
    return pd.DataFrame(rows)


def save_definition(name, tasks, folds, pairs):
    """Сохранить определение опыта: по строке на пару «район, год» с ролью в каждом фолде."""
    os.makedirs(FOLDS_DIR, exist_ok=True)
    rows = []
    for f in folds:
        for t, y in zip(pairs['territory_id'], pairs['year']):
            role = ('train' if y in f['train_years'] else
                    'val' if y == f['val_year'] else
                    'test' if y == f['test_year'] else None)
            if role:
                rows.append({'fold': f['fold'], 'territory_id': t, 'year': y, 'role': role})
    df = pd.DataFrame(rows)
    path = os.path.join(FOLDS_DIR, f'{name}.csv')
    df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    meta = pd.DataFrame([{'name': name, 'tasks': ' '.join(tasks), 'pairs': len(pairs),
                          'folds': ' '.join(str(f['fold']) for f in folds)}])
    meta.to_csv(os.path.join(FOLDS_DIR, f'{name}_meta.csv'), index=False, sep=';',
                encoding='utf-8-sig')
    return path


def summary(tasks, folds, pairs):
    """Таблица: сколько примеров в каждой части каждого фолда (по общей выборке)."""
    rows = []
    for f in folds:
        row = {'фолд (тест)': f['test_year'],
               'годы обучения': f"{f['train_years'][0]}–{f['train_years'][-1]}",
               'валидация': f['val_year']}
        for t in tasks:
            tr, va, te = indices(t, f, pairs)
            row[f'{t}: обуч/вал/тест'] = f'{len(tr)} / {len(va)} / {len(te)}'
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Фолды по годам и общая тестовая выборка')
    ap.add_argument('--tasks', nargs='+', required=True, help='задания (датасеты) опыта')
    ap.add_argument('--name', required=True, help='имя схемы, например winter_wheat')
    ap.add_argument('--test-years', nargs='*', type=int, help='годы тестов (по умолчанию все возможные)')
    ap.add_argument('--ladder', action='store_true',
                    help='«лесенка утечек»: четыре режима обучения на одном тесте '
                         '(random / district / year / double)')
    ap.add_argument('--groups', type=int, default=2, help='на сколько групп делить районы')
    ap.add_argument('--blocks', action='store_true',
                    help='разбиение блоками: группы районов × половины лет, обучение и тест '
                         'не пересекаются ни по районам, ни по годам')
    ap.add_argument('--loo', action='store_true',
                    help='один год из всех × группа районов: то же требование, что у --blocks, '
                         'но обучающих лет шесть вместо трёх, а фолдов шестнадцать')
    ap.add_argument('--year-mode', default='interleaved', choices=['interleaved', 'chrono'],
                    help='как делить годы: interleaved — поровну урожайных и неурожайных '
                         '(уровни половин совпадают), chrono — ранние против поздних')
    a = ap.parse_args()

    pairs = common_pairs(a.tasks)
    for t in a.tasks:
        plan = load_plan(t)
        print(f'{t}: примеров {len(plan)}, годных {int(valid_mask(plan).sum())}')
    print(f'общая выборка (есть во всех заданиях): {len(pairs)} пар «район, год», '
          f'районов {pairs["territory_id"].nunique()}, годы '
          f'{pairs["year"].min()}–{pairs["year"].max()}')

    if a.loo:
        plan = load_plan(a.tasks[0])
        plan = plan[valid_mask(plan)]
        pp = set(zip(pairs['territory_id'], pairs['year']))
        df, info = loo_definition(pp, plan, n_groups=a.groups)
        os.makedirs(FOLDS_DIR, exist_ok=True)
        path = os.path.join(FOLDS_DIR, f'{a.name}.csv')
        df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        gr = pd.Series(info['groups'])
        print('\nрайонов в группах: ' + ', '.join(
            f'g{g} — {int(n)}' for g, n in gr.value_counts().sort_index().items()))
        print('уровень по годам, ц/га: ' + ', '.join(
            f'{y} {v:.1f}' for y, v in sorted(info['level'].items())))
        cnt = df.pivot_table(index='fold', columns='role', values='year', aggfunc='count')
        cnt['лет_обуч'] = df[df.role == 'train'].groupby('fold')['year'].nunique()
        print('\nсостав фолдов (пар на роль):')
        print(cnt[['train', 'val', 'test', 'лет_обуч']].to_string())
        print(f'\nсхема сохранена: {path}')
        return

    if a.blocks:
        plan = load_plan(a.tasks[0])
        plan = plan[valid_mask(plan)]
        pp = set(zip(pairs['territory_id'], pairs['year']))
        df, info = block_definition(pp, plan, a.year_mode, n_groups=a.groups)
        os.makedirs(FOLDS_DIR, exist_ok=True)
        path = os.path.join(FOLDS_DIR, f'{a.name}.csv')
        df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')

        halves, level = info['halves'], info['level']
        print('\nгоды по половинам (в скобках — средняя урожайность года, ц/га):')
        for h in (0, 1):
            ys = sorted(y for y in halves if halves[y] == h)
            lv = np.mean([level[y] for y in ys])
            print(f'  h{h}: ' + ', '.join(f'{y} ({level[y]:.1f})' for y in ys)
                  + f'   -> средний уровень {lv:.2f}')
        gr = pd.Series(info['groups'])
        print(f'\nрайонов в группах: ' + ', '.join(f'g{g} — {int(n)}'
                                                  for g, n in gr.value_counts().sort_index().items()))
        # проверка: совпадают ли блоки по уровню и разбросу урожайности
        lv = plan.set_index(['territory_id', 'year'])['productive']
        chk = []
        for (t, y) in sorted(pp):
            g, h = info['groups'].get(t), halves.get(int(y))
            if g is None or h is None:
                continue
            chk.append({'блок': f'g{g}h{h}', 'урожайность': float(lv.loc[(t, int(y))])})
        chk = pd.DataFrame(chk)
        print('\nраспределение урожайности по блокам решётки:')
        print(chk.groupby('блок')['урожайность'].describe()[
            ['count', 'mean', 'std', '25%', '50%', '75%']].round(2).to_string())
        t = df.pivot_table(index='fold', columns='role', values='territory_id', aggfunc='count')
        print('\nразмеры фолдов:')
        print(t[['train', 'val', 'test']].to_string())
        print('\nопределение сохранено:', path)
        return

    if a.ladder:
        plan = load_plan(a.tasks[0])
        plan = plan[valid_mask(plan)]
        pp = set(zip(pairs['territory_id'], pairs['year']))
        test_years = a.test_years or [y for y in sorted(pairs['year'].unique())][3:]
        df = ladder_definition(pp, plan, test_years, n_groups=a.groups)
        os.makedirs(FOLDS_DIR, exist_ok=True)
        path = os.path.join(FOLDS_DIR, f'{a.name}.csv')
        df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        t = df.pivot_table(index='fold', columns='role', values='territory_id', aggfunc='count')
        print()
        print(t[['train', 'val', 'test']].to_string())
        print('\nопределение сохранено:', path)
        return

    folds = year_folds(pairs, a.test_years)
    if not folds:
        raise SystemExit('Не набралось ни одного фолда: мало лет для обучения')
    print()
    print(summary(a.tasks, folds, pairs).to_string(index=False))
    path = save_definition(a.name, a.tasks, folds, pairs)
    print(f'\nопределение опыта сохранено: {path}')


if __name__ == '__main__':
    main()
