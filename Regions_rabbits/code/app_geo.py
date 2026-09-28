"""Справочник регионов и районов для выпадающих списков GUI (из таблицы territories)."""
import db


def _to_int(v):
    try:
        s = str(v).strip()
        if s in ('', 'nan', 'None'):
            return None
        return int(float(s))
    except (ValueError, TypeError):
        return None


def list_regions():
    """[(id_region, имя_региона), ...] по алфавиту."""
    df = db.load('territories')
    if df.empty:
        return []
    out = {}
    for _, r in df.iterrows():
        rid = _to_int(r.get('id_region'))
        if rid is None:
            continue
        out.setdefault(rid, str(r.get('region') or rid))
    return sorted(out.items(), key=lambda x: x[1])


def list_districts(id_region):
    """[(id_district, имя_района), ...] для региона, по алфавиту."""
    df = db.load('territories')
    if df.empty:
        return []
    out = {}
    for _, r in df.iterrows():
        if _to_int(r.get('id_region')) != int(id_region):
            continue
        did = _to_int(r.get('id_district'))
        if did is None:
            continue
        out.setdefault(did, str(r.get('district') or did))
    return sorted(out.items(), key=lambda x: x[1])


def region_name(id_region):
    for rid, name in list_regions():
        if rid == int(id_region):
            return name
    return str(id_region)
