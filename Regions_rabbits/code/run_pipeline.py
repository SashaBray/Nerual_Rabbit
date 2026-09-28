"""Полный прогон конвейера на чистую (хронологический порядок годов):
удалить датасеты+модели(+отчёты) -> пересобрать офлайн -> почистить -> разведка(со свипом L)
-> обучить нейросети -> форкаст заблаговременности (all). Отчёт собирается отдельно (full_report.py).

Запуск:  python code/run_pipeline.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

import config

PY = sys.executable


def tasks():
    import pandas as pd
    df = pd.read_csv(config.DATASET_TASKS_FILE, dtype=str, encoding='utf-8-sig')
    return [t for t in df['task_id'] if str(t).endswith('_all')]


def sh(args, log):
    line = '>>> ' + ' '.join(args)
    print(line, flush=True)
    rc = subprocess.run([PY, '-X', 'utf8'] + args, cwd=os.path.dirname(__file__)).returncode
    log.append((args[1] if len(args) > 1 else args[0], rc))
    if rc != 0:
        print(f'    !! код возврата {rc}', flush=True)
    return rc


def batch_for(n):
    if n >= 5000:
        return 1000
    if n >= 2500:
        return 512
    if n >= 1200:
        return 256
    return 64


def n_samples(task):
    p = os.path.join(config.dataset_dir(task), 'meta.json')
    return int(json.load(open(p, encoding='utf-8'))['n_samples']) if os.path.exists(p) else 0


def main():
    t0 = time.time()
    TASKS = tasks()
    print(f'Задач: {len(TASKS)} -> {TASKS}', flush=True)
    log = []

    print('\n[1/6] Удаление датасетов, моделей и старых отчётов…', flush=True)
    for d in (config.DATASETS_DIR, config.MODELS_DIR):
        if os.path.isdir(d):
            for x in os.listdir(d):
                shutil.rmtree(os.path.join(d, x), ignore_errors=True)
    for sub in ('experiments', 'nn', 'forecast', 'summary'):
        shutil.rmtree(os.path.join(config.REPORTS_DIR, sub), ignore_errors=True)

    print('\n[2/6] Пересборка датасетов (офлайн, хронологический порядок)…', flush=True)
    for t in TASKS:
        sh(['build_dataset.py', t], log)

    print('\n[3/6] Чистка датасетов (битые/выбросы)…', flush=True)
    sh(['clean_datasets.py', '--apply'], log)

    print('\n[4/6] Разведка (классические модели, свип длины L)…', flush=True)
    for t in TASKS:
        sh(['explore_models.py', t], log)

    print('\n[5/6] Обучение нейросетей (mse24)…', flush=True)
    for t in TASKS:
        b = batch_for(n_samples(t))
        sh(['train_nn.py', t, '--arch', 'mse24', '--lr', '0.001',
            '--batch', str(b), '--epochs', '500', '--patience', '60'], log)

    print('\n[6/6] Эксперимент заблаговременности (все модели)…', flush=True)
    sh(['forecast_leadtime.py', 'all'], log)

    dt = (time.time() - t0) / 60.0
    fails = [a for a, rc in log if rc != 0]
    print(f'\n==== PIPELINE DONE за {dt:.0f} мин. Шагов: {len(log)}, '
          f'с ошибками: {len(fails)} {fails if fails else ""} ====', flush=True)


if __name__ == '__main__':
    main()
