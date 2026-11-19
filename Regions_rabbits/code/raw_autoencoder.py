"""Сжатие СЫРЫХ рядов: мешает ли разрешение или отсутствие меток.

Что уже известно. ``curve_autoencoder`` сжимал ГЛАДКИЕ кривые — накопленные аномалии на
двухсуточной сетке — и дал чистый диагноз: восстановление R² около 0.9, а урожайности в коде
нет. Отрицательный результат вида «персептрон не обошёл линию» сам по себе слаб: всегда можно
сказать, что плохо сжали или слишком сгладили. Поэтому здесь разведены два подозрения, и
каждое проверяется отдельным плечом опыта.

**Подозрение первое: сгладили лишнее.** Интеграл выбрасывает всё, что быстрее нескольких
суток, а после докачки шестичасовых рядов внутрисуточный ход у нас есть. Плечо: то же
сжатие на двух разрешениях — шестичасовом (1296 точек на сезон) и суточном (324 точки).
Одинаковая глубина сжатия обеспечена числом слоёв со шагом 2: пять для шестичасового, три
для суточного, так что узкое место в обоих случаях одинаково.

**Подозрение второе: сжатие не знает, что искать.** Автоэнкодер ищет направления
максимального разброса, а не связанные с целью. На сырых рядах самый крупный разброс — это
синоптические качели в 3-7 суток, и доля дисперсии, связанная с урожаем, там МЕНЬШЕ, чем на
сглаженных интегралах. Плечо: на тех же данных рядом считаются PLS (линейное сжатие, знающее
цель) и свёрточная сеть, обученная прямо на остаток (нелинейное сжатие, знающее цель).

Разложение опыта: 2 разрешения × 3 сжатия × 2 направления переноса, и каждое с контролем на
перемешанном остатке. Читается так:

* не бьёт ни одно сжатие ни на одном разрешении — вывод про сырые ряды сильный: в них нет
  того, чего не было в интегралах;
* бьют супервизорные, но не автоэнкодер — дело не в разрешении, а в отсутствии меток;
* бьёт шестичасовое, но не суточное — внутри суток действительно что-то есть.

Нормировка и честность. Аномалия — ряд года минус средняя района ПО ЕГО ЭПОХЕ. Эпохи
нормируются раздельно, поэтому знание о проверочных годах в обучение не протекает; норма по
прошлым годам (как в работе) здесь не взята намеренно: шестичасовые ряды начинаются с 2007,
и при норме по четырём прошлым годам от ранней эпохи остаётся три года — проверять не на чем.
Цель центрируется внутри года: модель упорядочивает районы внутри года, уровня года она не
знает, и обучение на нецентрированном остатке вбирает более сильную связь «каким выдался
год» (измерено в ``curve_autoencoder``). Автоэнкодер учится ТОЛЬКО на эпохе обучения, хотя
меток и не видит.

Что вышло (озимая пшеница, 9679 строк, 685 районов, три затравки).

Устояло ровно одно плечо: СУТОЧНАЯ сетка, обучение на поздней эпохе (12 лет), проверка на
ранней (5 лет) — r внутри года +0.13 при шуме +0.018. Затравка этот результат НЕ закрепляет:
свёртки на процессоре в несколько потоков складываются в непредсказуемом порядке, и один и
тот же номер затравки даёт разные числа в разных прогонах. Поэтому единица отчёта — не
отдельное число, а разброс: двенадцать обучений этого плеча в четырёх прогонах дали
+0.078…+0.171, среднее +0.131, ПОЛОЖИТЕЛЬНЫ ВСЕ ДВЕНАДЦАТЬ, и в каждом знак верен в 4-5
проверочных годах из 5. Обратное направление (учить на 5-6 годах) неустойчиво: затравки дали
+0.086, +0.142 и −0.029. Шестичасовое разрешение не воспроизводится вовсе. Значит, граница
проходит не по направлению переноса, а по ЧИСЛУ ЛЕТ В ОБУЧЕНИИ, и подозрение первое снято
отрицательно: внутри суток для урожая ничего нет.

Подозрение второе снято тоже, но наоборот: супервизорные сжатия переносятся ХУЖЕ. PLS даёт
на отложенном годе эпохи обучения до +0.52, а на другой эпохе +0.005…+0.06; сеть на остаток
неустойчива, и её контроль на шуме местами сам +0.05. Отбор направлений под цель находит
то, что годно только для своих лет.

Восстановление сырых аномалий почти нулевое: влага почвы R² 0.36-0.57, влажность воздуха
0.04-0.15, температура грунта 0.02-0.13, температура воздуха 0.01-0.06, осадки отрицательно
во всех двенадцати прогонах (против 0.9 на сглаженных интегралах). Сглаживание было УСЛОВИЕМ
работы сжатия, а не потерей.

Разбор кода (``raw_ae_probe.py``) показал, на что опирается гребень: температура грунта
0.40 вклада в разброс, воздуха 0.34, влага почвы 0.25. Предсказание коррелирует внутри года
с максимумом снега −0.51, с днями снега ≥30 см −0.45, с днями ниже −10 °C на грунте +0.38 —
то есть восстанавливает «снежность зимы», которой В КАНАЛАХ НЕТ, по температуре грунта: под
глубоким снегом грунт держится около нуля. Знак обратен литературному, и совпадает с тем,
что показала панель в ``agro_panel``.

Но это НЕ переоткрытая литература: связь двух предсказаний внутри года всего +0.20…+0.35, на
том же плече литературный гребень переносится в нуль (−0.011), а объединение двух блоков
ХУЖЕ кода отдельно во всех шести обучениях (+0.03…+0.09). Ближе всего содержательно
сравнение с одиночными признаками по грунту: ``зима_дней_м10`` и ``зима_грунт_мин`` — те же
данные, но через пороги руками — дают внутри эпохи 0.02-0.06 при шуме 0.047, тогда как код
по тому же каналу переносится на другую эпоху с +0.13. Из температуры грунта сжатие достаёт
заметно больше, чем три пороговых правила. Литературный блок здесь только
разбавляет. Побочное наблюдение, важное для всей работы: отложенный год эпохи обучения
переносу не указчик — у литературы +0.130 «в себя» против −0.011 на проверке, у кода затравка
2 дала +0.069 против +0.161.

В центнерах результата нет: на устойчивом плече dRMSE +0.004 ц/га при линии района 7.2,
в остальных минус. Корреляция 0.13 внутри года стоит около 1 % RMSE, и ошибка оценки
масштаба съедает её целиком — то же, что вышло на литературных признаках.

Запуск:
    python code/raw_autoencoder.py --culture "Озимая пшеница"
    python code/raw_autoencoder.py --culture "Озимая пшеница" --res daily --code 8
"""
import argparse
import datetime as dt
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from torch import nn

import config
import peak_regression as P
import series_6h

CHANNELS = ('mean_temp', 'mean_rh', 'mean_tmpgr10', 'mean_soilw10', 'mean_prec')
SEASON_DAYS = 324        # 1 сентября года Y-1 … 20 июля года Y
MIN_PER_DISTRICT = 3     # меньше трёх лет в эпохе — норму района считать не из чего
MIN_PER_YEAR = 30        # меньше тридцати районов в году — корреляция не считается
MIN_COVER = 0.90         # строка годна, если столько доли точек на месте
CODE = 8                 # размер кода на канал
EMB = 4                  # обучаемый признак канала
AE_EPOCHS, AE_PATIENCE = 40, 5
SUP_EPOCHS, SUP_PATIENCE = 60, 8
BATCH = 512
LAMBDAS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
PLS_COMP = (2, 4, 8, 16)
SUB = {'6h': 4, 'daily': 1}
DEPTH = {'6h': 5, 'daily': 3}    # слоёв со шагом 2: узкое место одинаково на двух сетках


# --------------------------------------------------------------------------- #
# Куб сырых рядов
# --------------------------------------------------------------------------- #
def year_grid(tid, parm, year, sub):
    """Ряд года на сетку: sub=4 — шестичасовые срезы как есть, sub=1 — суточные средние."""
    a = np.asarray(series_6h.get(tid, parm, year), dtype=float)
    out = np.full(366 * sub, np.nan, dtype=np.float32)
    if a.size == 0:
        return out
    v = a[:, 1]
    if sub == 4:
        i = np.rint(a[:, 0] * 4.0).astype(int)
        m = np.isfinite(v) & (i >= 0) & (i < out.size)
        out[i[m]] = v[m]
    else:
        d = np.floor(a[:, 0] + 1e-9).astype(int)
        m = np.isfinite(v) & (d >= 0) & (d < 366)
        s = np.bincount(d[m], weights=v[m], minlength=366)
        c = np.bincount(d[m], minlength=366)
        out[:366] = np.where(c > 0, s / np.maximum(c, 1.0), np.nan)
    return out


def season_of(arrs, year, sub):
    """Каналы сезона на оси «от 1 сентября года Y-1». -> (каналов, SEASON_DAYS*sub).

    От 1 сентября до 31 декабря всегда 122 суток, поэтому ось одинаковой длины в любой год;
    в високосный целевой год последние сутки сдвигаются на один день, что для сезонного
    хода безразлично.
    """
    L = SEASON_DAYS * sub
    out = np.full((len(CHANNELS), L), np.nan, dtype=np.float32)
    s0 = dt.date(year - 1, 9, 1)
    for j, p in enumerate(CHANNELS):
        for y in (year - 1, year):
            a = arrs.get((y, p))
            if a is None:
                continue
            i0 = (dt.date(y, 1, 1) - s0).days * sub
            lo, hi = max(0, -i0), min(a.size, L - i0)
            if hi > lo:
                out[j, i0 + lo:i0 + hi] = a[lo:hi]
    return out


def build_cube(keys, sub, path):
    """Сырые ряды панели. -> (A [строк, каналов, точек], покрытие [строк, каналов])."""
    if os.path.exists(path):
        z = np.load(path)
        print(f'куб: беру из {os.path.basename(path)}, {z["A"].shape}')
        return z['A'], z['cover']
    L = SEASON_DAYS * sub
    A = np.full((len(keys), len(CHANNELS), L), np.nan, dtype=np.float32)
    t0 = time.time()
    for i, (tid, grp) in enumerate(keys.groupby('territory_id'), 1):
        years = sorted(set(int(y) for y in grp['year']))
        arrs = {}
        for y in sorted(set(years) | {y - 1 for y in years}):
            for p in CHANNELS:
                arrs[(y, p)] = year_grid(tid, p, y, sub)
        for ix, y in zip(grp.index, grp['year']):
            A[ix] = season_of(arrs, int(y), sub)
        if i % 100 == 0:
            print(f'    куб: районов {i}, {time.time() - t0:.0f} c', flush=True)
    cover = np.isfinite(A).mean(axis=2).astype(np.float32)
    np.savez(path, A=A, cover=cover)
    print(f'куб: {A.shape}, {time.time() - t0:.0f} c -> {os.path.basename(path)}')
    return A, cover


def anomalies(A, tid, era_of):
    """Ряд года минус средняя района ПО ЕГО ЭПОХЕ; масштаб канала — по всей панели.

    Эпохи нормируются раздельно, поэтому средняя района не знает лет чужой эпохи. Каналы
    приводятся к одному масштабу, иначе общий энкодер описывал бы в основном влажность:
    у неё разброс в десятки раз больше температурного.
    """
    An = np.full_like(A, np.nan)
    warnings.filterwarnings('ignore', 'Mean of empty slice')
    for e in np.unique(era_of):
        for s in np.unique(tid[era_of == e]):
            k = np.flatnonzero((tid == s) & (era_of == e))
            if k.size < MIN_PER_DISTRICT:
                continue
            An[k] = A[k] - np.nanmean(A[k], axis=0)
    sd = np.nanstd(An, axis=(0, 2))
    sd = np.where(sd > 1e-9, sd, 1.0).astype(np.float32)
    An /= sd[None, :, None]
    print('масштаб каналов: ' + ' '.join('%s %.2f' % (c.replace('mean_', ''), v)
                                        for c, v in zip(CHANNELS, sd)))
    return An


# --------------------------------------------------------------------------- #
# Сети
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """Свёрточный энкодер одной кривой; номер канала подаётся обучаемым признаком.

    Энкодер один на все каналы: тогда ему достаётся по кривой с каждого канала каждой
    строки, и примеров становится в C раз больше, чем строк.
    """

    def __init__(self, n_ch, code, length, depth, emb=EMB):
        super().__init__()
        step = 2 ** depth
        self.pad = (-length) % step
        self.length, self.inner = length, (length + self.pad) // step
        self.emb = nn.Embedding(n_ch, emb)
        layers, cin = [], 1 + emb
        for i in range(depth):
            cout = 16 if i == 0 else 32
            layers += [nn.Conv1d(cin, cout, 5, stride=2, padding=2), nn.ReLU()]
            cin = cout
        self.enc = nn.Sequential(*layers)
        self.to_code = nn.Linear(32 * self.inner, code)
        self.depth, self.code = depth, code

    def forward(self, x, ch):
        e = self.emb(ch)
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad), mode='replicate')
        h = torch.cat([x, e[:, :, None].expand(-1, -1, x.shape[2])], dim=1)
        return self.to_code(self.enc(h).flatten(1))


class AE(nn.Module):
    """Автоэнкодер: энкодер плюс зеркальный декодер, знающий номер канала."""

    def __init__(self, n_ch, code, length, depth, emb=EMB):
        super().__init__()
        self.e = Encoder(n_ch, code, length, depth, emb)
        self.from_code = nn.Linear(code + emb, 32 * self.e.inner)
        layers = []
        for i in range(depth):
            cout = 1 if i == depth - 1 else 32
            layers.append(nn.ConvTranspose1d(32, cout, 5, 2, padding=2, output_padding=1))
            if i < depth - 1:
                layers.append(nn.ReLU())
        self.dec = nn.Sequential(*layers)

    def forward(self, x, ch):
        c = self.e(x, ch)
        h = self.from_code(torch.cat([c, self.e.emb(ch)], dim=1))
        y = self.dec(h.view(-1, 32, self.e.inner))
        return y[:, :, :self.e.length], c


class Supervised(nn.Module):
    """Тот же энкодер, но обучаемый прямо на остаток: сжатие знает, что искать."""

    def __init__(self, n_ch, code, length, depth):
        super().__init__()
        self.e = Encoder(n_ch, code, length, depth)
        self.head = nn.Linear(n_ch * code, 1)
        self.n_ch = n_ch

    def forward(self, x):
        n, c, L = x.shape
        ch = torch.arange(c).repeat(n)
        z = self.e(x.reshape(n * c, 1, L), ch).reshape(n, -1)
        return self.head(z)[:, 0]


def flat(X, rows):
    """Строки -> (кривые [N*C,1,L], номера каналов [N*C]) для обучения энкодера."""
    a = X[rows]
    n, c, L = a.shape
    return (torch.tensor(a.reshape(n * c, 1, L)),
            torch.tensor(np.tile(np.arange(c), n), dtype=torch.long))


def train_ae(model, x, ch, seed):
    """Обучение автоэнкодера с ранней остановкой по отложенной части кривых."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    cut = int(0.9 * len(x))
    tr, va = perm[:cut], perm[cut:]
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best, bad, state = np.inf, 0, None
    for _ep in range(AE_EPOCHS):
        model.train()
        for b in tr[torch.randperm(len(tr), generator=g)].split(BATCH):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x[b], ch[b])[0], x[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([float(nn.functional.mse_loss(model(x[b], ch[b])[0], x[b]))
                               for b in va.split(BATCH)]))
        if v < best - 1e-6:
            best, bad = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= AE_PATIENCE:
                break
    if state is not None:
        model.load_state_dict(state)
    return best


def codes_of(model, X, ok):
    """Коды всех каналов годных строк. -> [строк, каналов*code], негодные NaN."""
    out = np.full((len(X), len(CHANNELS) * model.e.code), np.nan, dtype=np.float32)
    if not ok.any():
        return out
    x, ch = flat(X, ok)
    model.eval()
    with torch.no_grad():
        z = torch.cat([model.e(x[b], ch[b]) for b in torch.arange(len(x)).split(BATCH)])
    out[ok] = z.numpy().reshape(int(ok.sum()), -1)
    return out


def recon_r2(model, X, ok):
    """R² восстановления по каждому каналу: без этого отрицательный результат ничего не даёт."""
    x, ch = flat(X, ok)
    model.eval()
    with torch.no_grad():
        y = torch.cat([model(x[b], ch[b])[0] for b in torch.arange(len(x)).split(BATCH)])
    a, b, c = x.numpy()[:, 0], y.numpy()[:, 0], ch.numpy()
    return [float(1 - np.sum((a[c == j] - b[c == j]) ** 2)
                  / np.sum((a[c == j] - a[c == j].mean()) ** 2)) for j in range(X.shape[1])]


def train_sup(model, X, y, tr, va, seed):
    """Обучение сети прямо на остаток; ранняя остановка по отложенному ГОДУ."""
    torch.manual_seed(seed)
    xt = torch.tensor(X[tr])
    yt = torch.tensor(y[tr], dtype=torch.float32)
    xv = torch.tensor(X[va])
    yv = torch.tensor(y[va], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    best, bad, state = np.inf, 0, None
    for _ep in range(SUP_EPOCHS):
        model.train()
        for b in torch.randperm(len(xt), generator=g).split(64):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(xv), yv))
        if v < best - 1e-5:
            best, bad = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= SUP_PATIENCE:
                break
    if state is not None:
        model.load_state_dict(state)
    return best


def predict_sup(model, X):
    model.eval()
    out = np.full(len(X), np.nan, dtype=np.float32)
    with torch.no_grad():
        for b in np.array_split(np.arange(len(X)), max(1, len(X) // BATCH)):
            out[b] = model(torch.tensor(X[b])).numpy()
    return out


# --------------------------------------------------------------------------- #
# Оценка
# --------------------------------------------------------------------------- #
def per_year_r(v, yr, era, y):
    out = []
    for g in sorted(set(yr[era])):
        m = (yr == g) & era & np.isfinite(v) & np.isfinite(y)
        if m.sum() < MIN_PER_YEAR or np.std(v[m]) < 1e-12:
            continue
        out.append(float(np.corrcoef(v[m], y[m])[0, 1]))
    return np.array(out)


def drop_district_era(y, tid, era_of):
    """Остаток минус средняя района В ЕГО ЭПОХЕ: иначе модель узнаёт район.

    Зачем. ``ост_norm`` нормирован по району на всей истории, но в пределах одной эпохи
    средняя района нулю не равна. Сеть, способная узнать район по его кривым, может
    подставлять эту средную и получать корреляцию внутри года ни из чего. Перестановка
    годов внутри района такую средную СОХРАНЯЕТ, поэтому и контроль на шуме становится
    положительным — в первом прогоне сеть на перемешанном остатке дала +0.050 и была
    положительна во все 12 лет проверки. После вычитания канал закрыт у обеих целей.
    """
    out = np.full_like(y, np.nan)
    for e in np.unique(era_of):
        for s in np.unique(tid[era_of == e]):
            k = np.flatnonzero((tid == s) & (era_of == e))
            if k.size >= MIN_PER_DISTRICT and np.isfinite(y[k]).any():
                out[k] = y[k] - np.nanmean(y[k])
    return out


def centre_in_year(y, yr):
    """Цель минус её средняя по году: модель упорядочивает районы, уровня года не знает."""
    out = np.full_like(y, np.nan)
    for g in sorted(set(yr)):
        m = (yr == g) & np.isfinite(y)
        if m.sum() >= MIN_PER_YEAR:
            out[m] = y[m] - y[m].mean()
    return out


def slope(v, y, rows):
    """Наклон цели на предсказание по отложенным строкам: масштаб поправки.

    Наклон зажат в [0, 1] и это не подгонка под проверку, а правило, объявленное заранее.
    Больше единицы честный наклон быть не может: это сжатие, а не растяжение. Меньше нуля
    означает «связи нет», и верное действие тогда — не поправлять вовсе, то есть нуль.
    Без зажима наклон, посчитанный по одному отложенному году, иногда вылетает (на пробе
    PLS дал ΔRMSE −2.4 ц/га из-за наклона, оценённого по горстке строк).
    """
    m = rows & np.isfinite(v) & np.isfinite(y)
    den = float(np.sum(v[m] ** 2))
    k = float(np.sum(v[m] * y[m]) / den) if den > 1e-12 else 0.0
    return min(max(k, 0.0), 1.0)


def ridge_fit(X, y, lam):
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    A = X[m].T @ X[m] + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X[m].T @ y[m])


def ridge_loo(X, y, yr, era, lams=LAMBDAS):
    """Гребень по годам эпохи обучения: год вон. -> (lam, r на отложенных годах)."""
    years = sorted(set(yr[era]))
    best = (lams[0], -9.0)
    for lam in lams:
        v = np.full(len(y), np.nan)
        for g in years:
            tr = era & (yr != g)
            w = ridge_fit(X, np.where(tr, y, np.nan), lam)
            m = era & (yr == g)
            v[m] = X[m] @ w
        r = per_year_r(v, yr, era, y)
        if len(r) and r.mean() > best[1]:
            best = (lam, float(r.mean()))
    return best


def pls_pick(X, y, yr, tr, va, comps=PLS_COMP):
    """Число компонент PLS по отложенному ГОДУ эпохи обучения.

    Полный перебор «год вон» здесь не по средствам: на шестичасовой сетке у PLS 6480
    столбцов, и одна подгонка идёт секунды. Отложенный год внутри эпохи обучения —
    то же по честности, дешевле по счёту.
    """
    best = (comps[0], -9.0)
    for nc in comps:
        p = PLSRegression(n_components=nc, scale=False).fit(X[tr], y[tr])
        v = np.full(len(y), np.nan)
        v[va] = np.asarray(p.predict(X[va])).reshape(-1)
        r = per_year_r(v, yr, va, y)
        if len(r) and r.mean() > best[1]:
            best = (nc, float(r.mean()))
    return best


def run_one(res, X, ok, yr, y, ysh, ost, sdv, sel, tst, lbl, code, seed, rows):
    """Три сжатия на одном разрешении и одном направлении переноса."""
    L = X.shape[2]
    selok, tstok = sel & ok, tst & ok
    print(f'\n--- {res}, {lbl}: обучение {selok.sum()} строк, проверка {tstok.sum()} ---',
          flush=True)

    # 1. Автоэнкодер: учится без меток и только на эпохе обучения
    t0 = time.time()
    ae = AE(len(CHANNELS), code, L, DEPTH[res])
    x, ch = flat(X, selok)
    val = train_ae(ae, x, ch, seed)
    del x, ch
    r2 = recon_r2(ae, X, tstok)
    print('    автоэнкодер: val MSE %.4f, R² восстановления на проверке %s, %.0f c'
          % (val, ' '.join('%s %.2f' % (c.replace('mean_', ''), v)
                           for c, v in zip(CHANNELS, r2)), time.time() - t0), flush=True)
    Z = codes_of(ae, X, ok)
    Zok = np.isfinite(Z).all(axis=1)
    del ae

    F = X.reshape(len(X), -1)
    last = max(sorted(set(yr[selok])))
    tr, va = selok & (yr != last), selok & (yr == last)
    for tgt, nm in ((y, 'данные'), (ysh, 'шум')):
        yc = centre_in_year(tgt, yr)
        fit = selok & np.isfinite(yc)
        # 1. гребень на замороженных кодах
        lam, r_in = ridge_loo(Z, yc, yr, fit & Zok)
        w = ridge_fit(Z, np.where(fit & Zok, yc, np.nan), lam)
        v_ae = np.where(Zok, Z @ w, np.nan)
        # 2. PLS на тех же кривых: линейное сжатие, знающее цель
        t1 = time.time()
        nc, p_in = pls_pick(F, np.nan_to_num(yc), yr, tr & np.isfinite(yc),
                            va & np.isfinite(yc))
        pls = PLSRegression(n_components=nc, scale=False).fit(F[fit], yc[fit])
        v_pls = np.asarray(pls.predict(F)).reshape(-1)
        del pls
        t2 = time.time()
        # 3. тот же энкодер, обученный прямо на остаток
        sup = Supervised(len(CHANNELS), code, L, DEPTH[res])
        train_sup(sup, X, np.nan_to_num(yc), tr & np.isfinite(yc), va & np.isfinite(yc), seed)
        v_sup = predict_sup(sup, X)
        del sup
        rv = per_year_r(v_sup, yr, va & np.isfinite(yc), yc)
        for v, how, inner in ((v_ae, 'автоэнкодер+гребень', r_in),
                              (v_pls, 'PLS', p_in),
                              (v_sup, 'сеть на остаток',
                               float(rv.mean()) if len(rv) else np.nan)):
            r = per_year_r(v, yr, tstok, yc)
            # в центнеры: предсказание центрируется внутри года (уровня года модель не
            # знает), масштаб берётся по отложенному году эпохи обучения — «в себя» он
            # завышен в три-четыре раза и портит RMSE даже при верном знаке
            vc = centre_in_year(v, yr)
            k = slope(vc, yc, va & np.isfinite(yc))
            m = tstok & np.isfinite(vc) & np.isfinite(ost) & np.isfinite(sdv)
            line = float(np.sqrt(np.mean(ost[m] ** 2)))
            mod = float(np.sqrt(np.mean((ost[m] - k * vc[m] * sdv[m]) ** 2)))
            print('    %-6s %-21s на отложенном годе r %+.4f; НА ПРОВЕРКЕ r %+.4f '
                  '(плюс в %d из %d), dRMSE %+.3f ц/га при линии %.2f'
                  % (nm, how, inner, r.mean() if len(r) else np.nan,
                     int((r > 0).sum()), len(r), line - mod, line), flush=True)
            rows.append({'разрешение': res, 'направление': lbl, 'что': nm, 'сжатие': how,
                         'затравка': seed,
                         'настройка': lam if how.startswith('авто') else
                         (nc if how == 'PLS' else ''),
                         'r_проверка': round(float(r.mean()), 4) if len(r) else np.nan,
                         'лет': len(r), 'знак_совпал': int((r > 0).sum()),
                         'dRMSE': round(line - mod, 4)})
        print('    (PLS %.0f c, сеть %.0f c)' % (t2 - t1, time.time() - t2), flush=True)
    return r2


def main():
    ap = argparse.ArgumentParser(description='Сжатие сырых рядов: разрешение и метки')
    ap.add_argument('--culture', required=True)
    ap.add_argument('--split', type=int, default=2012, help='последний год ранней эпохи')
    ap.add_argument('--res', default='both', choices=('both', '6h', 'daily'))
    ap.add_argument('--code', type=int, default=CODE)
    ap.add_argument('--seeds', default='0',
                    help='затравки через запятую: инициализация сети случайна')
    ap.add_argument('--limit', type=int, default=0,
                    help='взять только N районов: проба конвейера, не опыт')
    a = ap.parse_args()

    cu = P.culture_row(a.culture)
    _hdoy, task = P.harvest_doy(cu['culture_id'])
    src = os.path.join(config.REPORTS_DIR, 'peak_residual', task + '_early', 'residuals.csv')
    if not os.path.exists(src):
        raise SystemExit(f'нет {src}; сначала peak_residual_screen.py --early')
    rl = pd.read_csv(src, sep=';', decimal=',', encoding='utf-8-sig')
    if a.limit:
        keep = sorted(set(rl['territory_id']))[:a.limit]
        rl = rl[rl['territory_id'].isin(keep)].reset_index(drop=True)
    keys = rl[['territory_id', 'year']].reset_index(drop=True)
    tid, yr = keys['territory_id'].to_numpy(), keys['year'].to_numpy(int)
    early, late = yr <= a.split, yr > a.split
    era_of = np.where(early, 0, 1)
    y = drop_district_era(rl['ост_norm'].to_numpy(float), tid, era_of)

    rng = np.random.default_rng(5)
    ysh = y.copy()
    for s in np.unique(tid):
        k = np.flatnonzero(tid == s)
        ysh[k] = y[rng.permutation(k)]

    outdir = os.path.join(config.REPORTS_DIR, 'raw_ae', task)
    cdir = os.path.join(config.DATASETS_DIR, 'raw_ae')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    print(f'{cu["title"]}: строк {len(keys)}, районов {len(set(tid))}, '
          f'ранняя {early.sum()}, поздняя {late.sum()}')

    seeds = [int(v) for v in str(a.seeds).split(',') if v.strip()]
    ost = rl['ост'].to_numpy(float)
    sdv = rl['sd'].to_numpy(float)
    rows, recon = [], {}
    for res in (('6h', 'daily') if a.res == 'both' else (a.res,)):
        tag = f'{task}_{res}' + (f'_lim{a.limit}' if a.limit else '')
        A, cover = build_cube(keys, SUB[res], os.path.join(cdir, tag + '.npz'))
        ok = (cover >= MIN_COVER).all(axis=1)
        for s in np.unique(tid):
            for e in (0, 1):
                k = np.flatnonzero((tid == s) & (era_of == e))
                if k.size < MIN_PER_DISTRICT:
                    ok[k] = False
        print(f'{res}: годных строк {ok.sum()} из {len(ok)}; покрытие по каналам '
              + ' '.join('%s %.3f' % (c.replace('mean_', ''), cover[:, j].mean())
                         for j, c in enumerate(CHANNELS)))
        An = anomalies(A, tid, era_of)
        del A
        X = np.ascontiguousarray(np.where(np.isfinite(An), An, 0.0), dtype=np.float32)
        del An
        for sd_i in seeds:
            for sel, tst, lbl in ((early, late, 'ранняя -> поздняя'),
                                  (late, early, 'поздняя -> ранняя')):
                recon[(res, lbl, sd_i)] = run_one(
                    res, X, ok, yr, y, ysh, ost, sdv, sel, tst, lbl, a.code, sd_i, rows)
        del X

    t = pd.DataFrame(rows)
    print('\n=== итог ===')
    print(t.to_string(index=False))
    for val in ('r_проверка', 'dRMSE'):
        for nm in ('данные', 'шум'):
            piv = t[t['что'] == nm].pivot_table(
                index=['разрешение', 'сжатие'], columns='направление', values=val)
            print('\n%s, %s (затравки %s):\n%s'
                  % (val, nm, seeds, piv.round(4).to_string()))
    t.to_csv(os.path.join(outdir, 'folds.csv'), sep=';', decimal=',', index=False,
             encoding='utf-8-sig')
    pd.DataFrame([{'разрешение': k[0], 'направление': k[1], 'затравка': k[2],
                   **{c.replace('mean_', ''): round(v, 4) for c, v in zip(CHANNELS, r)}}
                  for k, r in recon.items()]).to_csv(
        os.path.join(outdir, 'recon.csv'), sep=';', decimal=',', index=False,
        encoding='utf-8-sig')
    print(f'\nотчёт -> {outdir}')


if __name__ == '__main__':
    main()
