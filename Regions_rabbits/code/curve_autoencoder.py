"""Автоэнкодер кривых-аномалий: сжать ход сезона без меток, потом предсказать остаток.

Замысел. Сводка по окну («весенний интеграл NDVI») выбрасывает форму кривой: один и тот же
интеграл даёт и район, отставший в развитии, и район, вышедший слабым. Перебор сводок уже
упёрся в потолок — внутригодовая корреляция лучшей комбинации 0.23 на озимой пшенице
(``peak_residual_screen.py``). Здесь кривая не сворачивается руками, а сжимается обучением:
автоэнкодер учит базис формы БЕЗ МЕТОК, затем его веса замораживаются, и по кодам персептрон
предсказывает остаток районной линии.

Чем это отличается от прежней попытки (``autoencoder.py``, проиграла эталону).

1. Раскладка данных другая. Там было 809–887 обучающих примеров «регион-год», здесь 6618
   «район-год» на 645 районах. Это единственный честный довод за повтор: автоэнкодеру
   достаётся 5 каналов × ~4800 обучающих строк ≈ 24 тысячи кривых на ~40 тысяч весов.
2. Каналы отобраны по ГЛАДКОСТИ, а не по полноте списка. Аномалия накопленного ряда — это
   интеграл суточной аномалии, то есть гладкая кривая без высокочастотного шума. Общий
   базис на пяти однотипных гладких каналах осмысленнее, чем на двенадцати разношёрстных,
   где свёртке приходится описывать и монотонную лестницу осадков, и зимние пики снега.
3. Норма района — по четырём предыдущим годам, а не по двум: шум нормы σ/2 вместо σ/1.4.
   Измерено отдельно, что вычитание нормы вообще удваивает внутригодовую корреляцию.
4. Цель центрируется ВНУТРИ года. Поправка применяется только для упорядочивания районов
   внутри года, значит и учить её надо на внутригодовых отклонениях. Если учить на остатке
   целиком, коэффициент вберёт более сильную связь «каким выдался год» и внутри года будет
   перехлёстывать вдвое — это измерено, и именно из-за этого прежняя линейная поправка в
   центнерах делала модель хуже.

Что обязательно докладывается рядом, иначе опыт ничего не доказывает.

* **Ошибка восстановления по каналам.** Отрицательный результат вида «персептрон не обошёл
  линию» слаб: всегда можно сказать, что плохо сжали. Если же ряд из кода восстанавливается
  с R² = 0.9, а урожайности в коде нет, вывод становится сильным.
* **PLS на тех же кривых.** Супервизорное линейное сжатие ищет направления, коррелирующие с
  целью, а автоэнкодер — направления максимального разброса. Если PLS выигрывает, дело не в
  нелинейности, а в том, что автоэнкодер тратит код не на то.
* **Контроль перестановкой.** Остаток перемешивается по годам внутри района, персептрон
  учится на тех же замороженных кодах. Это планка, с которой сравнивается результат.

Запуск:
    python code/curve_autoencoder.py --culture "Озимая пшеница" --code 8
    python code/curve_autoencoder.py --culture "Озимая пшеница" --folds 2   # проба

Выход: ``workspace/reports/curve_ae/<задание>/`` — ``folds.csv``, ``recon.csv``,
``predictions.csv``, графики; отчёт ``docs/Автоэнкодер_кривых_<культура>.docx``.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from torch import nn

import config
import db
import explore_models as E
import peak_regression as P
import peak_residual_screen as S
import plots

# канал: (ряд, как превращать в гладкую кривую). «накоп» — кумулятивная сумма по сетке.
CHANNELS = (
    ('ndvi', 'сырой'),             # сам ход съёмки: форма горки
    ('ndvi', 'накоп'),             # набранная биомасса к дню t
    ('mean_temp_acc', 'сырой'),    # накопленная температура Веги — уже монотонная
    ('mean_prec_acc', 'сырой'),    # накопленные осадки Веги — уже монотонные
    ('mean_snowc', 'накоп'),       # интеграл снежного покрова: «сколько снега пролежало»
)
GRID_FROM = 20          # с какого дня года кладётся сетка
GRID_STEP = 2           # шаг сетки в днях: 2 дня хватает для гладких кривых
NORM_YEARS = 4          # норма района по последним N прошлым годам
MIN_PREV = 2            # меньше прошлых лет — аномалию не считаем
EMB = 4                 # размер обучаемого признака канала
AE_EPOCHS = 40
AE_PATIENCE = 5
HEAD_EPOCHS = 300
HEAD_PATIENCE = 20
PLS_COMP = (2, 4, 8)


# --------------------------------------------------------------------------- #
# Кривые и аномалии
# --------------------------------------------------------------------------- #
def _grid(hdoy):
    return np.arange(GRID_FROM, int(hdoy) + 1, GRID_STEP)


def _curve(a, g):
    """Ряд (день, значение) на сетку g линейной интерполяцией. -> массив или None."""
    if a.size == 0:
        return None
    m = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
    m &= (a[:, 0] >= g[0] - 30) & (a[:, 0] <= g[-1] + 30)
    if m.sum() < 8:
        return None
    d, v = a[m, 0], a[m, 1]
    o = np.argsort(d)
    return np.interp(g, d[o], v[o])


def curves_table(rl, mask_parm, hdoy, cache):
    """Кривые всех каналов. -> (A [строк, каналов, точек], keys DataFrame)."""
    g = _grid(hdoy)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f'кривые: беру из {os.path.basename(cache)}, {z["A"].shape}')
        return z['A'], pd.DataFrame({'territory_id': z['tid'].astype(str), 'year': z['yr']})
    keys = rl[['territory_id', 'year']].drop_duplicates().sort_values(
        ['territory_id', 'year']).reset_index(drop=True)
    A = np.full((len(keys), len(CHANNELS), len(g)), np.nan)
    t0 = time.time()
    for i, (tid, grp) in enumerate(keys.groupby('territory_id'), 1):
        for j, (parm, kind) in enumerate(CHANNELS):
            p = mask_parm if parm == 'ndvi' else parm
            for ix, yr in zip(grp.index, grp['year']):
                a = np.asarray(db.get_time_series(tid, p, int(yr)), dtype=float)
                if a.size and parm == 'ndvi':
                    a = a[a[:, 1] > 0]
                c = _curve(a if a.size else np.empty((0, 2)), g)
                if c is None:
                    continue
                A[ix, j] = np.cumsum(c) * GRID_STEP if kind == 'накоп' else c
        if i % 150 == 0:
            print(f'    кривые: районов {i}, {time.time() - t0:.0f} c', flush=True)
    np.savez_compressed(cache, A=A, tid=keys['territory_id'].to_numpy(),
                        yr=keys['year'].to_numpy(), grid=g)
    print(f'кривые: {A.shape}, {time.time() - t0:.0f} c')
    return A, keys


def anomalies(A, tid, yr, window=NORM_YEARS):
    """Кривая года минус средняя кривая того же района по ПРЕДЫДУЩИМ годам.

    Норма только по прошлым годам: в работе в год Y других и нет. Первые годы каждого
    района остаются пустыми — это цена честной нормы, около пятой части строк.
    """
    out = np.full_like(A, np.nan)
    for t in np.unique(tid):
        k = np.flatnonzero(tid == t)
        k = k[np.argsort(yr[k])]
        for i in range(1, len(k)):
            p = A[k[max(0, i - window):i]]
            ok = np.isfinite(p).all(axis=2)              # кривая целиком или никак
            n = ok.sum(axis=0)
            mu = np.where(n[:, None] >= MIN_PREV,
                          np.nansum(np.where(ok[:, :, None], p, 0.0), axis=0)
                          / np.maximum(n, 1)[:, None], np.nan)
            out[k[i]] = A[k[i]] - mu
    return out


def yield_trend(rl, window=NORM_YEARS):
    """Наклон урожайности района по прошлым годам, в долях от уровня. -> массив.

    Единственный признак опыта, который не является формой кривой. Районная линия знает
    УРОВЕНЬ района, но не знает его тренда: район, который три года растёт, и район,
    который три года падает, для неё одинаковы. Берутся только прошлые годы.
    """
    out = np.full(len(rl), np.nan)
    v = rl['цель'].to_numpy(float)
    tid, yr = rl['territory_id'].to_numpy(), rl['year'].to_numpy(int)
    for t in np.unique(tid):
        k = np.flatnonzero(tid == t)
        k = k[np.argsort(yr[k])]
        for i in range(len(k)):
            p = k[max(0, i - window):i]
            y = v[p]
            m = np.isfinite(y)
            if m.sum() < 3 or np.mean(y[m]) <= 0:
                continue
            out[k[i]] = float(np.polyfit(np.arange(m.sum()), y[m], 1)[0] / np.mean(y[m]))
    return out


# --------------------------------------------------------------------------- #
# Автоэнкодер
# --------------------------------------------------------------------------- #
class SeriesAE(nn.Module):
    """Свёрточный автоэнкодер одной кривой; канал подаётся обучаемым признаком.

    Энкодер один на все каналы: тогда ему достаётся по кривой с каждого канала каждой
    строки, и примеров становится в C раз больше, чем строк. Декодер тоже знает канал,
    поэтому коду не нужно тратиться на хранение того, какой это канал.
    """

    def __init__(self, n_ch, code, length, emb=EMB):
        super().__init__()
        self.pad = (-length) % 8                      # длина должна делиться на 8
        self.length, self.inner = length, (length + self.pad) // 8
        self.emb = nn.Embedding(n_ch, emb)
        self.enc = nn.Sequential(
            nn.Conv1d(1 + emb, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU())
        self.to_code = nn.Linear(32 * self.inner, code)
        self.from_code = nn.Linear(code + emb, 32 * self.inner)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 5, 2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, 2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 7, 2, padding=3, output_padding=1))

    def encode(self, x, ch):
        e = self.emb(ch)
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad), mode='replicate')
        h = torch.cat([x, e[:, :, None].expand(-1, -1, x.shape[2])], dim=1)
        return self.to_code(self.enc(h).flatten(1))

    def forward(self, x, ch):
        c = self.encode(x, ch)
        h = self.from_code(torch.cat([c, self.emb(ch)], dim=1))
        y = self.dec(h.view(-1, 32, self.inner))
        return y[:, :, :self.length], c


def flat_curves(An, rows):
    """Строки -> (кривые [N,1,L], номера каналов [N]) для обучения энкодера."""
    A = An[rows]
    n, c, L = A.shape
    x = A.reshape(n * c, 1, L)
    ch = np.tile(np.arange(c), n)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(ch, dtype=torch.long)


def train_ae(model, x, ch, seed, quiet=True):
    """Обучение автоэнкодера с ранней остановкой по отложенной части кривых."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    cut = int(0.9 * len(x))
    tr, va = perm[:cut], perm[cut:]
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best, bad, state = np.inf, 0, None
    for ep in range(AE_EPOCHS):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        for b in order.split(256):
            opt.zero_grad()
            out, _ = model(x[b], ch[b])
            loss = nn.functional.mse_loss(out, x[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(x[va], ch[va])[0], x[va]))
        if v < best - 1e-6:
            best, bad = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= AE_PATIENCE:
                break
        if not quiet:
            print(f'      ае эпоха {ep} val {v:.5f}')
    if state is not None:
        model.load_state_dict(state)
    return best


def codes_of(model, An):
    """Коды всех каналов всех строк. -> [строк, каналов*code]; пустые строки -> NaN."""
    ok = np.isfinite(An).all(axis=(1, 2))
    n, c, _L = An.shape
    out = np.full((n, c * model.to_code.out_features), np.nan)
    if not ok.any():
        return out
    x, ch = flat_curves(An, ok)
    model.eval()
    with torch.no_grad():
        z = torch.cat([model.encode(x[b], ch[b]) for b in torch.arange(len(x)).split(512)])
    out[ok] = z.numpy().reshape(ok.sum(), -1)
    return out


def recon_r2(model, An, rows):
    """R² восстановления по каждому каналу: опыт без этого ничего не доказывает."""
    ok = rows & np.isfinite(An).all(axis=(1, 2))
    x, ch = flat_curves(An, ok)
    model.eval()
    with torch.no_grad():
        y = torch.cat([model(x[b], ch[b])[0] for b in torch.arange(len(x)).split(512)])
    x, y = x.numpy()[:, 0], y.numpy()[:, 0]
    ch = ch.numpy()
    return [float(1 - np.sum((x[ch == j] - y[ch == j]) ** 2)
                  / np.sum((x[ch == j] - x[ch == j].mean()) ** 2))
            for j in range(An.shape[1])]


def build_head(n_in, hidden=32, dropout=0.2):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 1))


def train_head(F, y, train, seed):
    """Персептрон по замороженным кодам. -> предсказание на всех строках.

    Цель центрирована внутри ОБУЧАЮЩИХ лет, поэтому на тестовых строках её нет. Условие
    «цель известна» отбирает строки для обучения, но не для предсказания: предсказывать
    можно всюду, где есть коды.
    """
    okf = np.isfinite(F).all(axis=1)
    ok = okf & np.isfinite(y)
    mu = F[train & ok].mean(axis=0)
    sg = F[train & ok].std(axis=0)
    sg = np.where(sg > 1e-9, sg, 1.0)
    Z = (F - mu) / sg
    g = torch.Generator().manual_seed(seed)
    idx = np.flatnonzero(train & ok)
    perm = torch.randperm(len(idx), generator=g).numpy()
    cut = int(0.85 * len(idx))
    tr, va = idx[perm[:cut]], idx[perm[cut:]]
    X = torch.tensor(np.nan_to_num(Z), dtype=torch.float32)
    T = torch.tensor(y, dtype=torch.float32)[:, None]
    torch.manual_seed(seed)
    net = build_head(F.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    best, bad, state = np.inf, 0, None
    for _ep in range(HEAD_EPOCHS):
        net.train()
        o = tr[torch.randperm(len(tr), generator=g).numpy()]
        for b in np.array_split(o, max(1, len(o) // 512)):
            opt.zero_grad()
            nn.functional.mse_loss(net(X[b]), T[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(X[va]), T[va]))
        if v < best - 1e-7:
            best, bad = v, 0
            state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= HEAD_PATIENCE:
                break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        p = net(X).numpy()[:, 0]
    return np.where(okf, p, np.nan)


# --------------------------------------------------------------------------- #
# Оценка
# --------------------------------------------------------------------------- #
def _centre(v, yr, rows):
    """Вычесть среднюю своего года по заданным строкам."""
    out = np.full(len(v), np.nan)
    for y in np.unique(yr[rows]):
        k = rows & (yr == y) & np.isfinite(v)
        if k.sum() >= 5:
            out[k] = v[k] - v[k].mean()
    return out


def _rho_in(p, y, yr, test):
    k = test & np.isfinite(p) & np.isfinite(y)
    if k.sum() < 50:
        return np.nan
    r = spearmanr(p[k], y[k]).statistic
    return float(r) if np.isfinite(r) else np.nan


def _gain(p, ynorm, yabs, sdv, yr, train, test):
    """Выигрыш по ошибке ВНУТРИ года, ц/га. Масштаб поправки калибруется по обучению.

    Выход модели — ОЦЕНКА, а не центнеры: персептрон, обученный на MSE, на незнакомом годе
    легко выдаёт верный порядок районов со вдвое завышенной амплитудой, и тогда RMSE
    портится при правильном ранге. Поэтому оценка умножается на один коэффициент,
    подобранный на обучающих годах — ровно так же, как это делалось для суммы сводок, иначе
    сравнение было бы не в равных условиях.
    """
    pc, yc = _centre(p, yr, train), _centre(ynorm, yr, train)
    m = train & np.isfinite(pc) & np.isfinite(yc)
    if m.sum() < 200 or np.std(pc[m]) < 1e-12:
        return np.nan
    b = float(np.polyfit(pc[m], yc[m], 1)[0])
    out = []
    for y in np.unique(yr[test]):
        k = test & (yr == y) & np.isfinite(p) & np.isfinite(yabs)
        if k.sum() < 20:
            continue
        corr = b * (p[k] - p[k].mean()) * sdv[k]
        e0, e1 = yabs[k], yabs[k] - corr
        out.append(float(np.sqrt(np.mean((e0 - e0.mean()) ** 2))
                         - np.sqrt(np.mean((e1 - e1.mean()) ** 2))))
    return float(np.mean(out)) if out else np.nan


def pls_pred(An, trend, ytr, train, test, n_comp):
    """Супервизорное линейное сжатие тех же кривых — контроль для автоэнкодера."""
    n = len(trend)
    F = np.hstack([An.reshape(n, -1), trend[:, None]])
    ok = np.isfinite(F).all(axis=1)
    m = train & ok & np.isfinite(ytr)
    if m.sum() < 200 or (test & ok).sum() < 50:
        return np.full(n, np.nan)
    mu, sg = F[m].mean(axis=0), F[m].std(axis=0)
    sg = np.where(sg > 1e-9, sg, 1.0)
    p = PLSRegression(n_components=n_comp, scale=False)
    p.fit((F[m] - mu) / sg, ytr[m])
    out = np.full(n, np.nan)
    out[ok] = p.predict((F[ok] - mu) / sg).ravel()
    return out


# --------------------------------------------------------------------------- #
# Графики и отчёт
# --------------------------------------------------------------------------- #
def _plots(outdir, folds, grid, An, ok):
    imgs = []
    cols = [c for c in folds.columns if c.startswith('r_')]

    def bars(lang):
        fig, ax = plt.subplots(figsize=(9.5, 5))
        k = np.arange(len(folds))
        w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(k + i * w - 0.4, folds[c], w, label=c.replace('r_', ''))
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(k)
        ax.set_xticklabels(folds['фолд'], rotation=45)
        ax.set_ylabel(plots.tr('Внутригодовая корреляция с остатком',
                               'Within-year correlation with the residual', lang))
        ax.set_title(plots.tr('Автоэнкодер, PLS и контроль по отложенным годам',
                              'Autoencoder, PLS and control by held-out year', lang))
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    imgs.append(('Что держится на отложенных годах',
                 plots.bilingual(outdir, 'ae_folds', bars)))

    def curves(lang):
        fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14, 3.2), sharex=True)
        rng = np.random.default_rng(1)
        pick = rng.choice(np.flatnonzero(ok), size=min(40, int(ok.sum())), replace=False)
        for j, ax in enumerate(axes):
            for i in pick:
                ax.plot(grid, An[i, j], lw=0.6, alpha=0.4, color='#1f77b4')
            ax.axhline(0, color='k', lw=1)
            ax.set_title(f'{CHANNELS[j][0]}|{CHANNELS[j][1]}', fontsize=8)
            ax.set_xlabel(plots.tr('день года', 'day of year', lang), fontsize=8)
        axes[0].set_ylabel(plots.tr('аномалия', 'anomaly', lang), fontsize=8)
        fig.suptitle(plots.tr('Входные кривые: отклонение от нормы района по прошлым годам',
                              'Input curves: deviation from the district past-years norm',
                              lang))
        fig.tight_layout()
        return fig

    imgs.append(('Входные кривые-аномалии', plots.bilingual(outdir, 'ae_curves', curves)))
    return imgs


def _docx(outdir, title, meta, folds, recon, imgs):
    from docx import Document
    from docx.shared import Inches
    docs = os.path.join(os.path.dirname(os.path.dirname(config.REPORTS_DIR)), 'docs')
    os.makedirs(docs, exist_ok=True)
    doc = Document()
    doc.add_heading(f'Автоэнкодер кривых-аномалий. {title}', 0)

    doc.add_heading('Замысел и чем он отличается от прежней попытки', level=1)
    doc.add_paragraph(
        'Сводка по окну выбрасывает форму кривой: один и тот же весенний интеграл NDVI даёт '
        'и район, отставший в развитии, и район, вышедший слабым. Перебор сводок упёрся в '
        'потолок, поэтому здесь кривая не сворачивается руками, а сжимается обучением: '
        'автоэнкодер учит базис формы без меток, веса замораживаются, и по кодам персептрон '
        'предсказывает остаток районной линии.')
    for s in (f'Раскладка другая: прежде было 809–887 примеров «регион-год», здесь '
              f'{meta["rows"]} «район-год» на {meta["districts"]} районах. Автоэнкодеру '
              f'достаётся {meta["curves"]} кривых на {meta["ae_params"]} весов — примеров '
              f'больше, чем весов;',
              'каналы отобраны по гладкости: аномалия накопленного ряда есть интеграл '
              'суточной аномалии, то есть кривая без высокочастотного шума. Общий базис на '
              'однотипных гладких каналах осмысленнее, чем на разношёрстных;',
              f'норма района — по {NORM_YEARS} предыдущим годам (шум нормы вдвое меньше, чем '
              'у нормы по двум), и только по прошлым: в работе в год Y других и нет;',
              'цель центрируется внутри года. Поправка применяется только для '
              'упорядочивания районов внутри года, значит и учить её надо на внутригодовых '
              'отклонениях — иначе коэффициент вберёт более сильную связь «каким выдался '
              'год» и внутри года будет перехлёстывать вдвое.'):
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph(
        f'Фолдов {meta["folds"]} из {meta["years"]} годов: норма по прошлым годам требует '
        f'как минимум {MIN_PREV} предыдущих лет, поэтому первые годы истории в опыте '
        f'участвуют только как норма. Строк с аномалией {meta["ok"]} из {meta["rows"]}.')
    doc.add_paragraph(
        f'Каналы: {", ".join(f"{a}|{b}" for a, b in CHANNELS)}. Сетка: дни '
        f'{GRID_FROM}–{meta["hdoy"]} с шагом {GRID_STEP}, {meta["length"]} точек. Код: '
        f'{meta["code"]} чисел на канал, то есть {meta["code"] * len(CHANNELS)} чисел на '
        f'строку плюс наклон урожайности района по прошлым годам. Фолды — отложенный год.')

    doc.add_heading('Что держится на отложенных годах', level=1)
    doc.add_paragraph(
        'Все столбцы r — внутригодовая корреляция предсказания с фактическим остатком '
        'районной линии на отложенном годе, то есть способность упорядочить районы между '
        'собой. «контроль» — тот же персептрон на тех же замороженных кодах, но остаток '
        'перемешан по годам внутри района; это планка, а не ноль. Столбцы ΔRMSE — выигрыш '
        f'по ошибке внутри года в {meta["unit"]} при ошибке линии {meta["rmse_line"]:.2f}.')
    E._table_from_df(doc, folds, maxrows=20)
    m = folds.drop(columns=['фолд']).mean()
    pls_best = max(PLS_COMP, key=lambda k: m[f'r_PLS{k}'])
    doc.add_paragraph(
        f'Среднее по {len(folds)} отложенным годам: автоэнкодер {m["r_ае"]:+.3f}, PLS '
        + ', '.join(f'{k} компонент {m[f"r_PLS{k}"]:+.3f}' for k in PLS_COMP)
        + f', контроль перестановкой {m["r_контроль"]:+.3f}. В {meta["unit"]}: автоэнкодер '
        f'{m["ΔRMSE_ае"]:+.3f}, PLS{pls_best} {m[f"ΔRMSE_PLS{pls_best}"]:+.3f}. Для '
        f'сравнения, та же метрика на том же остатке: объявленная тройка сводок кривой NDVI '
        f'даёт +0.230 и +0.078 {meta["unit"]}, честно отобранные три сводки — +0.177 и '
        f'+0.019.')
    doc.add_paragraph(
        'Число компонент PLS не подбиралось по отложенному году — это был бы отбор по '
        'ответу. Докладываются все варианты сразу, и устойчивость результата к этому '
        'выбору видна прямо в таблице.')
    doc.add_paragraph(
        'Сопоставление автоэнкодера с PLS и есть главное в этой таблице. PLS сжимает те же '
        'кривые, но супервизорно — ищет направления, коррелирующие с целью, а не '
        'направления максимального разброса. Если PLS не хуже, дело не в нелинейности: '
        'автоэнкодер тратит код не на то, что нужно.')

    doc.add_heading('Ошибка восстановления по каналам', level=1)
    doc.add_paragraph(
        'Без этой таблицы опыт ничего не доказывает. Вывод «персептрон не обошёл линию» сам '
        'по себе слаб — всегда можно возразить, что плохо сжали. Но если кривая из кода '
        'восстанавливается хорошо, а урожайности в коде нет, вывод становится сильным: вот '
        'представление, из которого кривая восстанавливается с такой-то точностью, и '
        'урожайности в нём не содержится.')
    E._table_from_df(doc, recon, maxrows=20)

    doc.add_heading('Графики', level=1)
    for name, png in imgs:
        doc.add_heading(name, level=2)
        if png:
            doc.add_picture(png, width=Inches(6.3))

    doc.add_heading('Как повторить', level=1)
    for cmd in ('python code/curve_autoencoder.py --culture "Озимая пшеница" --code 8',
                'python code/curve_autoencoder.py --culture "Озимая пшеница" --code 4',
                'python code/curve_autoencoder.py --culture "Яровая пшеница" --min-years 6'):
        doc.add_paragraph(cmd, style='Intense Quote')
    doc.add_paragraph(f'Каталог: {outdir}')

    out = os.path.join(docs, f'Автоэнкодер_кривых_{title.replace(" ", "_")}.docx')
    try:
        doc.save(out)
    except PermissionError:
        out = out.replace('.docx', '_new.docx')
        doc.save(out)
    print(f'Отчёт Word: {out}')
    return out


# --------------------------------------------------------------------------- #
def run(culture, target, min_years, form, code, seed, n_folds, rebuild):
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    cu = P.culture_row(culture)
    hdoy, task = P.harvest_doy(cu['culture_id'])
    title = str(cu['title'])
    outdir = os.path.join(config.REPORTS_DIR, 'curve_ae', task)
    os.makedirs(outdir, exist_ok=True)
    ccache = os.path.join(outdir, 'curves.npz')
    if rebuild and os.path.exists(ccache):
        os.remove(ccache)

    d, _diag = P.targets_table(cu['bdpmo_culture'])
    pcache = os.path.join(config.REPORTS_DIR, 'peak_regression', task, 'peaks.csv')
    os.makedirs(os.path.dirname(pcache), exist_ok=True)
    pk = P.peaks_table(sorted(d['territory_id'].unique()), sorted(d['year'].unique()),
                       cu['mask_culture'], hdoy, pcache)
    pk['territory_id'] = pk['territory_id'].astype(str)
    d = d.merge(pk, on=['territory_id', 'year'], how='inner')
    rl = S.resid_table(d, target, form, min_years)
    if rl.empty:
        raise SystemExit(f'ни один район не набрал {min_years} лет')
    rl['territory_id'] = rl['territory_id'].astype(str)
    mask_parm = str(d['маска'].mode().iloc[0])

    A, keys = curves_table(rl, mask_parm, hdoy, ccache)
    rl = rl.merge(keys.assign(_i=np.arange(len(keys))), on=['territory_id', 'year'],
                  how='inner').sort_values('_i').reset_index(drop=True)
    A = A[rl['_i'].to_numpy()]
    tid, yr = rl['territory_id'].to_numpy(), rl['year'].to_numpy(int)
    An = anomalies(A, tid, yr)
    # каждый канал в своих единицах, а энкодер общий; величины надо сделать сопоставимыми
    for j in range(An.shape[1]):
        sj = np.nanstd(An[:, j])
        An[:, j] /= sj if sj > 1e-12 else 1.0
    ok = np.isfinite(An).all(axis=(1, 2))
    trend = yield_trend(rl)
    # наклон требует трёх прошлых лет; где его нет, строку терять нельзя — иначе целый
    # ранний год выпадает из опыта целиком. Подставляем «тренд неизвестен» = 0.
    ts = np.nanstd(trend)
    trend = np.where(np.isfinite(trend),
                     (trend - np.nanmean(trend)) / (ts if ts > 1e-12 else 1.0), 0.0)
    ynorm, yabs, sdv = (rl['ост_norm'].to_numpy(float), rl['ост'].to_numpy(float),
                        rl['sd'].to_numpy(float))
    rmse_line = float(np.sqrt(np.mean(yabs ** 2)))
    grid = _grid(hdoy)
    print(f'строк {len(rl)}, с аномалией {ok.sum()} ({ok.mean():.0%}), районов '
          f'{rl["territory_id"].nunique()}, точек в кривой {len(grid)}, '
          f'RMSE линии {rmse_line:.2f}')

    years = [y for y in np.unique(yr) if ((yr == y) & ok).sum() >= 50]
    if n_folds:
        years = years[-n_folds:]
    rng = np.random.default_rng(seed)
    ysh = ynorm.copy()                       # контроль: годы внутри района перемешаны
    for t in np.unique(tid):
        k = np.flatnonzero(tid == t)
        ysh[k] = ynorm[rng.permutation(k)]

    frows, rrows, preds, npar = [], [], [], 0
    for y in years:
        t0 = time.time()
        train, test = (yr != y) & ok, (yr == y) & ok
        yc = _centre(ynorm, yr, train)       # цель — внутригодовое отклонение
        ysc = _centre(ysh, yr, train)
        torch.manual_seed(seed)
        model = SeriesAE(len(CHANNELS), code, len(grid))
        npar = sum(p.numel() for p in model.parameters())
        x, ch = flat_curves(An, train)
        val = train_ae(model, x, ch, seed)
        for par in model.parameters():        # веса замораживаются
            par.requires_grad_(False)
        F = np.hstack([codes_of(model, An), trend[:, None]])
        p_ae = train_head(F, yc, train, seed)
        p_sh = train_head(F, ysc, train, seed)
        # число компонент PLS НЕ выбирается по тесту: это был бы отбор по ответу.
        # Докладываются все варианты, читатель видит и устойчивость к этому выбору.
        row = {'фолд': str(y), 'строк_обуч': int(train.sum()),
               'строк_тест': int(test.sum()),
               'r_ае': round(_rho_in(p_ae, ynorm, yr, test), 4),
               'r_контроль': round(_rho_in(p_sh, ynorm, yr, test), 4),
               'ΔRMSE_ае': round(_gain(p_ae, ynorm, yabs, sdv, yr, train, test), 4)}
        for nc in PLS_COMP:
            p = pls_pred(An, trend, yc, train, test, nc)
            row[f'r_PLS{nc}'] = round(_rho_in(p, ynorm, yr, test), 4)
            row[f'ΔRMSE_PLS{nc}'] = round(_gain(p, ynorm, yabs, sdv, yr, train, test), 4)
            if nc == PLS_COMP[-1]:
                p_pls = p
        row['ае_val_mse'] = round(val, 5)
        frows.append(row)
        rr = recon_r2(model, An, test)
        rrows.append(dict(zip([f'{a}|{b}' for a, b in CHANNELS],
                              [round(v, 4) for v in rr]), фолд=str(y)))
        preds.append(pd.DataFrame({'territory_id': tid[test], 'year': yr[test],
                                   'ост': yabs[test], 'sd': sdv[test],
                                   'п_ае': p_ae[test], 'п_pls': p_pls[test]}))
        print(f'  {y}: ае {row["r_ае"]:+.3f}  pls '
              + '/'.join(f'{row[f"r_PLS{k}"]:+.3f}' for k in PLS_COMP)
              + f'  контроль {row["r_контроль"]:+.3f}  восстановление {np.mean(rr):.3f}  '
              f'{time.time() - t0:.0f} c', flush=True)

    folds = pd.DataFrame(frows)
    recon = pd.DataFrame(rrows)[['фолд'] + [f'{a}|{b}' for a, b in CHANNELS]]
    print('\n=== по фолдам')
    print(folds.to_string(index=False))
    print('\n=== среднее')
    print(folds.drop(columns=['фолд']).mean().round(4).to_string())
    print('\n=== R2 восстановления по каналам')
    print(recon.to_string(index=False))

    folds.to_csv(os.path.join(outdir, 'folds.csv'), sep=';', decimal=',', index=False,
                 encoding='utf-8-sig')
    recon.to_csv(os.path.join(outdir, 'recon.csv'), sep=';', decimal=',', index=False,
                 encoding='utf-8-sig')
    pd.concat(preds, ignore_index=True).round(4).to_csv(
        os.path.join(outdir, 'predictions.csv'), sep=';', decimal=',', index=False,
        encoding='utf-8-sig')
    meta = {'unit': P.UNITS[target], 'rows': len(rl), 'code': code,
            'folds': len(folds), 'years': int(np.unique(yr).size), 'ok': int(ok.sum()),
            'districts': int(rl['territory_id'].nunique()),
            'curves': int(ok.sum()) * len(CHANNELS), 'ae_params': npar,
            'length': len(grid), 'hdoy': int(hdoy), 'rmse_line': rmse_line}
    imgs = _plots(outdir, folds, grid, An, ok)
    _docx(outdir, title, meta, folds, recon, imgs)
    print(f'Таблицы: {outdir}')
    return folds


def main():
    ap = argparse.ArgumentParser(description='Автоэнкодер кривых-аномалий и персептрон')
    ap.add_argument('--culture', required=True)
    ap.add_argument('--target', default='убранная', choices=P.TARGETS)
    ap.add_argument('--min-years', type=int, default=7)
    ap.add_argument('--form', default='линейная', choices=P.FORMS)
    ap.add_argument('--code', type=int, default=8, help='чисел в коде на один канал')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--folds', type=int, default=0, help='взять только последние N лет')
    ap.add_argument('--rebuild', action='store_true', help='пересчитать кэш кривых')
    a = ap.parse_args()
    run(a.culture, a.target, a.min_years, a.form, a.code, a.seed, a.folds, a.rebuild)


if __name__ == '__main__':
    main()
