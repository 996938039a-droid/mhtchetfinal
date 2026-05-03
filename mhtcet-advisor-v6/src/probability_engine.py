"""
probability_engine.py — Core probability and prediction logic.

Key design:
  - Stage I only for cutoff history (primary merit list, not Stage II)
  - Per-round independent history: R2 predicted from R2 history, not R1-drop
  - Partial data: if some years missing for a round, use what's available
  - Non-linear trend adjustment: scales differently at high vs low percentiles
  - Seat matrix: zero-seat quota filter + seat-count probability weight
"""

import math, re
import numpy as np
import pandas as pd
from typing import Optional

YEAR_WEIGHTS     = [0.40, 0.30, 0.20, 0.10]
SIGMOID_K        = 0.8
_SEAT_WEIGHT_MIN = 0.85
_SEAT_WEIGHT_MAX = 1.15
_SEAT_LOG_MIN    = math.log(10)
_SEAT_LOG_MAX    = math.log(576)

_CAT_TO_SM_BASE = {
    'OPEN':'OPEN','SC':'SC','ST':'ST','VJ':'VJ/DT',
    'NT1':'NTB','NT2':'NTC','NT3':'NTD',
    'OBC':'OBC','SEBC':'SEBC',
}


# ─── Seat helpers ─────────────────────────────────────────────────────────────

def _category_seat_key(code: str, si: dict) -> float:
    if not si: return -1
    code = code.upper()
    if code == 'TFWS':                                return si.get('tfws_seats', 0)
    if code.startswith('EWS'):                        return si.get('ews_seats', 0)
    if code == 'ORPHAN':                              return si.get('orphan_seats', 0)
    if 'PWD' in code:                                 return si.get('pwd_total', -1)
    if code.startswith('DEF'):                        return si.get('def_total', -1)
    gender = 'G' if code.startswith('G') else ('L' if code.startswith('L') else None)
    if not gender: return -1
    rest = code[1:]
    if   rest.endswith('S'): level, cat = 'SL',  rest[:-1]
    elif rest.endswith('H'): level, cat = 'HU',  rest[:-1]
    elif rest.endswith('O'): level, cat = 'OHU', rest[:-1]
    else: return -1
    sm_base = _CAT_TO_SM_BASE.get(cat)
    if not sm_base: return -1
    col = f"{level.lower()}_{sm_base.lower().replace('/dt','')}_{'g' if gender=='G' else 'l'}"
    val = si.get(col, si.get(col.replace('_vj_','_vj/dt_'), -1))
    return val if val is not None else -1


def filter_categories_by_seats(categories: list, si: dict) -> list:
    if not si: return categories
    kept = [c for c in categories if _category_seat_key(c, si) != 0]
    return kept if kept else categories


def seat_count_weight(cap_seats: float) -> float:
    if cap_seats <= 0: return 1.0
    norm = (math.log(max(cap_seats,10)) - _SEAT_LOG_MIN) / (_SEAT_LOG_MAX - _SEAT_LOG_MIN)
    return _SEAT_WEIGHT_MIN + norm * (_SEAT_WEIGHT_MAX - _SEAT_WEIGHT_MIN)


# ─── Non-linear trend adjustment ──────────────────────────────────────────────

def apply_trend_adjustment(base_cutoff: float, trend_adj: float) -> float:
    """
    Apply trend adjustment non-linearly based on where the cutoff sits.

    The idea: a +1.5 percentile shift means different things at different levels.
    At 98+ percentile, the scale is very compressed — a +1.5 shift is enormous.
    At 60 percentile, +1.5 barely matters.

    We scale the adjustment using a compression factor:
      - Above 95: scale = 0.4  (very compressed, adjustment has less effect)
      - 85-95:    scale = 0.7
      - 70-85:    scale = 0.9
      - Below 70: scale = 1.0  (full adjustment applies)

    This reflects reality: when cutoffs are already at 98%, a "harder exam"
    can't push them much further. When cutoffs are at 65%, there's more room.
    """
    if trend_adj == 0:
        return base_cutoff

    if base_cutoff >= 95:
        scale = 0.4
    elif base_cutoff >= 85:
        scale = 0.7
    elif base_cutoff >= 70:
        scale = 0.9
    else:
        scale = 1.0

    adjusted = base_cutoff + (trend_adj * scale)
    return round(max(0.0, min(adjusted, 100.0)), 4)


# ─── Core math ────────────────────────────────────────────────────────────────

def sigmoid(x: float, k: float = SIGMOID_K) -> float:
    return 100.0 / (1.0 + np.exp(-k * x))


def weighted_average(values: list) -> float:
    """Recency-weighted average. Last element in list = most recent = 40% weight."""
    n = len(values)
    w = YEAR_WEIGHTS[:n][::-1]
    w = [x / sum(w) for x in w]
    return sum(v * wt for v, wt in zip(values, w))


def predict_cutoff_for_round(
    subset: pd.DataFrame,
    target_round: int,
    trend_adjustment: float = 0.0,
) -> Optional[float]:
    """
    Predict closing percentile for target_round using that round's own history.

    subset: rows already filtered to one college + branch + category,
            with columns [year, cap_round, percentile].
            MUST already be filtered to Stage I rows only (caller's responsibility).

    Partial data handling:
      - Uses whatever years are available for that round (no minimum required)
      - Fallback chain if round has NO data: R3→R2→R1
      - Missing years within a round are simply absent from weighted avg
        (weights renormalise automatically)

    Returns None only if there is absolutely no data at all.
    """
    rounds_in_data = sorted(subset['cap_round'].unique())
    if not rounds_in_data:
        return None

    # Determine which round's history to use
    if target_round in rounds_in_data:
        use_round = target_round
    elif target_round == 3 and 2 in rounds_in_data:
        use_round = 2
    elif 1 in rounds_in_data:
        use_round = 1
    else:
        use_round = rounds_in_data[0]

    rows = subset[subset['cap_round'] == use_round].sort_values('year')
    if rows.empty:
        return None

    historical = rows['percentile'].tolist()
    base = weighted_average(historical)
    predicted = apply_trend_adjustment(base, trend_adjustment)
    return predicted


def get_round_data_status(cutoff_df: pd.DataFrame) -> dict:
    if cutoff_df.empty:
        return {1: False, 2: False, 3: False, 'years': []}
    rounds = set(cutoff_df['cap_round'].unique())
    years  = sorted(cutoff_df['year'].unique())
    return {1: 1 in rounds, 2: 2 in rounds, 3: 3 in rounds, 'years': years}


def compute_probability(
    student_pct: float, predicted_cutoff: float,
    historical: list, cap_seats: float = 60.0,
) -> float:
    gap = student_pct - predicted_cutoff
    k   = max(0.3, SIGMOID_K - (np.std(historical) * 0.05 if len(historical) >= 2 else 0))
    raw = sigmoid(gap, k)
    return round(float(max(0.0, min(raw * seat_count_weight(cap_seats), 100.0))), 1)


def classify(probability: float) -> dict:
    if probability < 10: return {"label":"Reach",   "color":"#EF4444","emoji":"🎯","order":0}
    if probability < 30: return {"label":"Dream",   "color":"#F97316","emoji":"⭐","order":1}
    if probability < 70: return {"label":"Target",  "color":"#3B82F6","emoji":"✅","order":2}
    if probability < 90: return {"label":"Safe",    "color":"#22C55E","emoji":"🛡️","order":3}
    return                      {"label":"Assured", "color":"#6B7280","emoji":"🔒","order":4}


def _detect_trend(historical: list) -> str:
    if len(historical) < 2: return "stable"
    delta = historical[-1] - historical[0]
    if delta > 1.5:  return "rising"
    if delta < -1.5: return "falling"
    return "stable"


def _norm_str(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[,.'\"()]", '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


# ─── Category helpers ─────────────────────────────────────────────────────────

def build_category_columns(base_category: str, gender: str, seat_level: str) -> list:
    g = "L" if gender == "female" else "G"
    cat_map = {"OPEN":"OPEN","SC":"SC","ST":"ST","VJ":"VJ","NT1":"NT1",
               "NT2":"NT2","NT3":"NT3","OBC":"OBC","SEBC":"SEBC","EWS":"EWS"}
    if base_category == "EWS":
        return [f"EWS{seat_level}"]
    return [f"{g}{cat_map.get(base_category, base_category)}{seat_level}"]


def get_relevant_categories(
    base_category: str, gender: str,
    home_university: str, college_university: str,
    special_quotas: list,
) -> list:
    is_home = bool(home_university and college_university and
                   home_university.lower() == college_university.lower())
    cols  = build_category_columns(base_category, gender, "S")
    cols += build_category_columns(base_category, gender, "H" if is_home else "O")
    for sq in special_quotas:
        if sq == "TFWS":    cols.append("TFWS")
        elif sq == "DEF":   cols += ["DEFOPEN", f"DEF{base_category[:2].upper()}"]
        elif sq == "PWD":   cols.append("PWDOPEN")
        elif sq == "ORPHAN": cols.append("ORPHAN")
    return list(dict.fromkeys(cols))


# ─── Single college-branch analysis ──────────────────────────────────────────

def analyse_college_branch(
    cutoff_df: pd.DataFrame,
    college_name: str,
    course_name: str,
    eligible_categories: list,
    student_percentile: float,
    target_round: int,
    trend_adjustment: float,
    seat_info: Optional[dict] = None,
) -> Optional[dict]:
    sub = cutoff_df[
        (cutoff_df['college_name'] == college_name) &
        (cutoff_df['course_name']  == course_name)
    ]
    if sub.empty: return None

    # ── KEY FIX: Use Stage I only ──────────────────────────────────────────
    # Stage I = primary merit list. Stage II = secondary pass, lower cutoff.
    # Mixing them caused artificially low predictions (min was picking Stage II).
    if 'stage' in sub.columns:
        stage1 = sub[sub['stage'].astype(str).str.strip().str.upper() == 'I']
        if not stage1.empty:
            sub = stage1
        # If stage column exists but no 'I' rows, use all (data quality issue)

    valid_cats = filter_categories_by_seats(eligible_categories, seat_info or {})

    best_result = None
    best_prob   = -1

    for cat in valid_cats:
        cat_rows = sub[sub['category'] == cat]
        if cat_rows.empty: continue

        # Best closing percentile per year+round (min within Stage I = most conservative)
        yearly = (
            cat_rows.groupby(['year','cap_round'])
            .agg(percentile=('percentile','min'))
            .reset_index()
        )

        predicted = predict_cutoff_for_round(yearly, target_round, trend_adjustment)
        if predicted is None: continue

        # Historical for THIS round (for volatility)
        this_round = yearly[yearly['cap_round'] == target_round]
        historical = (this_round if not this_round.empty else yearly).sort_values('year')['percentile'].tolist()

        cap_seats = (seat_info or {}).get('cap_seats', 60)
        prob      = compute_probability(student_percentile, predicted, historical, cap_seats)
        cl        = classify(prob)

        if prob > best_prob:
            best_prob = prob
            si = seat_info or {}
            best_result = {
                "college_name":       college_name,
                "course_name":        course_name,
                "best_category":      cat,
                "predicted_cutoff":   predicted,
                "historical_cutoffs": historical,
                "rounds_available":   sorted(yearly['cap_round'].unique().tolist()),
                "probability":        prob,
                "classification":     cl,
                "gap":                round(student_percentile - predicted, 2),
                "trend":              _detect_trend(historical),
                "data_years":         len(historical),
                "cap_seats":    si.get('cap_seats', 0),
                "sl_seats":     si.get('sl_total', 0),
                "hu_seats":     si.get('hu_total', 0),
                "ohu_seats":    si.get('ohu_total', 0),
                "has_hu_seats": si.get('has_hu', False),
                "tfws_seats":   si.get('tfws_seats', 0),
                "ews_seats":    si.get('ews_seats', 0),
                "pwd_seats":    si.get('pwd_total', 0),
                "def_seats":    si.get('def_total', 0),
                "has_seat_data": bool(si),
            }
    return best_result


# ─── Bulk predictions ─────────────────────────────────────────────────────────

def generate_all_predictions(
    cutoff_df, seat_matrix_df, student_percentile, base_category,
    gender, home_university, special_quotas, preferred_branches,
    college_type_filter, target_round, trend_adjustment, branch_priority,
    university_map, seat_lookup=None,
) -> pd.DataFrame:
    if cutoff_df.empty: return pd.DataFrame()

    if seat_lookup is None and not seat_matrix_df.empty:
        from src.data_loader import build_seat_lookup
        seat_lookup = build_seat_lookup(seat_matrix_df)
    seat_lookup = seat_lookup or {}

    if college_type_filter:
        def mt(status):
            if not status: return False
            s = str(status).lower()
            return any(ct.lower() in s for ct in college_type_filter)
        fdf = cutoff_df[cutoff_df['status'].apply(mt)]
    else:
        fdf = cutoff_df

    if fdf.empty: return pd.DataFrame()
    if preferred_branches:
        fdf = fdf[fdf['course_name'].isin(preferred_branches)]
    if fdf.empty: return pd.DataFrame()

    combos  = fdf[['college_name','course_name','status']].drop_duplicates()
    results = []

    for _, row in combos.iterrows():
        college = row['college_name']
        branch  = row['course_name']
        cu      = university_map.get(college, "")
        cats    = get_relevant_categories(base_category, gender, home_university, cu, special_quotas)
        si      = seat_lookup.get((_norm_str(college), _norm_str(branch)), {})
        res     = analyse_college_branch(fdf, college, branch, cats,
                                         student_percentile, target_round,
                                         trend_adjustment, si)
        if res:
            res['status']             = row['status']
            res['college_university'] = cu
            res['is_home_university'] = bool(
                home_university and cu and home_university.lower() == cu.lower())
            results.append(res)

    if not results: return pd.DataFrame()

    df = pd.DataFrame(results)
    if preferred_branches:
        br = {b: i for i, b in enumerate(preferred_branches)}
        df['branch_rank'] = df['course_name'].map(br).fillna(len(preferred_branches))
    else:
        df['branch_rank'] = 0

    df['classification_order'] = df['classification'].apply(lambda x: x['order'])

    if branch_priority:
        df = df.sort_values(['branch_rank','classification_order','probability'],
                            ascending=[True,True,False])
    else:
        df = df.sort_values(['classification_order','branch_rank','probability'],
                            ascending=[True,True,False])
    return df.reset_index(drop=True)


def generate_preference_list(predictions_df: pd.DataFrame, max_list: int = 10) -> pd.DataFrame:
    if predictions_df.empty: return pd.DataFrame()
    df     = predictions_df.copy()
    labels = df['classification'].apply(lambda x: x['label'] if isinstance(x,dict) else x)
    dream  = df[labels.isin(['Dream','Reach'])].head(max(1, int(max_list*0.20)))
    target = df[labels == 'Target'].head(max(1, int(max_list*0.50)))
    safe   = df[labels.isin(['Safe','Assured'])].head(max(3, int(max_list*0.30)))
    pref   = (pd.concat([dream,target,safe])
               .drop_duplicates(subset=['college_name','course_name'])
               .head(max_list).reset_index(drop=True))
    pref.index = pref.index + 1
    return pref


def float_freeze_advice(current_college, current_branch, current_probability,
                         predictions_df, next_round) -> dict:
    if predictions_df.empty:
        return {"advice":"FREEZE","reason":"No data available."}
    cc = classify(current_probability)
    better = predictions_df[
        (predictions_df['probability'] > current_probability + 5) &
        ~((predictions_df['college_name']==current_college) &
          (predictions_df['course_name']==current_branch))
    ]
    scb = predictions_df[
        (predictions_df['college_name']==current_college) &
        (predictions_df['course_name']!=current_branch) &
        (predictions_df['probability']>40)
    ]
    if cc['label'] in ['Assured','Safe'] and better.empty:
        return {"advice":"FREEZE",
                "reason":f"Your allocation at {current_college} ({current_branch}) is strong. No significantly better options likely.",
                "better_options":0}
    elif not scb.empty and better.empty:
        return {"advice":"SLIDE",
                "reason":f"Stay at {current_college} but try for a better branch. {len(scb)} better branch(es) available.",
                "better_options":len(scb),
                "slide_options":scb[['course_name','probability']].head(3).to_dict('records')}
    elif not better.empty:
        top = better.head(3)[['college_name','course_name','probability']].to_dict('records')
        return {"advice":"FLOAT",
                "reason":f"{len(better)} better option(s) may open in Round {next_round}. You keep your current seat while trying.",
                "better_options":len(better),"top_options":top}
    return {"advice":"FREEZE",
            "reason":"Current allocation is reasonable. Risk outweighs potential gains.",
            "better_options":0}
