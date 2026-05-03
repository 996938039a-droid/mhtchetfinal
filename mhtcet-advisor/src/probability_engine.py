"""
probability_engine.py — Core probability and prediction logic.
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

SPECIAL_QUOTA_CODES = {'TFWS','EWS','EWSS','EWSH','EWSO','ORPHAN','PWDOPEN','DEFOPEN'}

_CAT_TO_SM_BASE = {
    'OPEN':'OPEN','SC':'SC','ST':'ST','VJ':'VJ/DT',
    'NT1':'NTB','NT2':'NTC','NT3':'NTD',
    'OBC':'OBC','SEBC':'SEBC',
}

COLLEGE_TYPE_RANK = {
    'government autonomous': 1,
    'university department': 2,
    'university managed autonomous': 2,
    'government': 3,
    'government-aided autonomous': 4,
    'government-aided': 5,
    'un-aided autonomous': 6,
    'un-aided': 7,
}


def _college_tier(status: str) -> int:
    if not status: return 9
    s = status.lower()
    for k, v in COLLEGE_TYPE_RANK.items():
        if k in s: return v
    return 8


def _category_seat_key(code: str, si: dict) -> float:
    if not si: return -1
    code = code.upper()
    if code == 'TFWS':           return si.get('tfws_seats', 0)
    if code.startswith('EWS'):   return si.get('ews_seats', 0)
    if code == 'ORPHAN':         return si.get('orphan_seats', 0)
    if 'PWD' in code:            return si.get('pwd_total', -1)
    if code.startswith('DEF'):   return si.get('def_total', -1)
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


def apply_trend_adjustment(base_cutoff: float, trend_adj: float) -> float:
    if trend_adj == 0: return base_cutoff
    if   base_cutoff >= 95: scale = 0.4
    elif base_cutoff >= 85: scale = 0.7
    elif base_cutoff >= 70: scale = 0.9
    else:                   scale = 1.0
    return round(max(0.0, min(base_cutoff + trend_adj * scale, 100.0)), 4)


def sigmoid(x: float, k: float = SIGMOID_K) -> float:
    return 100.0 / (1.0 + np.exp(-k * x))


def weighted_average(values: list) -> float:
    n = len(values)
    w = YEAR_WEIGHTS[:n][::-1]
    w = [x / sum(w) for x in w]
    return sum(v * wt for v, wt in zip(values, w))


def predict_cutoff_for_round(subset: pd.DataFrame, target_round: int,
                              trend_adjustment: float = 0.0) -> Optional[float]:
    rounds_in_data = sorted(subset['cap_round'].unique())
    if not rounds_in_data: return None
    if   target_round in rounds_in_data:            use_round = target_round
    elif target_round == 3 and 2 in rounds_in_data: use_round = 2
    elif 1 in rounds_in_data:                       use_round = 1
    else:                                           use_round = rounds_in_data[0]
    rows = subset[subset['cap_round'] == use_round].sort_values('year')
    if rows.empty: return None
    base = weighted_average(rows['percentile'].tolist())
    return apply_trend_adjustment(base, trend_adjustment)


def get_round_data_status(cutoff_df: pd.DataFrame) -> dict:
    if cutoff_df.empty: return {1:False,2:False,3:False,'years':[]}
    rounds = set(cutoff_df['cap_round'].unique())
    return {1:1 in rounds, 2:2 in rounds, 3:3 in rounds,
            'years': sorted(cutoff_df['year'].unique())}


def compute_probability(student_pct, predicted_cutoff, historical, cap_seats=60.0):
    gap = student_pct - predicted_cutoff
    k   = max(0.3, SIGMOID_K - (np.std(historical)*0.05 if len(historical)>=2 else 0))
    raw = sigmoid(gap, k)
    return round(float(max(0.0, min(raw * seat_count_weight(cap_seats), 100.0))), 1)


def classify(probability: float) -> dict:
    if probability < 10: return {"label":"Reach",   "color":"#EF4444","emoji":"🎯","order":0}
    if probability < 30: return {"label":"Dream",   "color":"#F97316","emoji":"⭐","order":1}
    if probability < 70: return {"label":"Target",  "color":"#2563EB","emoji":"✅","order":2}
    if probability < 90: return {"label":"Safe",    "color":"#16A34A","emoji":"🛡️","order":3}
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


def build_category_columns(base_category, gender, seat_level):
    g = "L" if gender == "female" else "G"
    cat_map = {"OPEN":"OPEN","SC":"SC","ST":"ST","VJ":"VJ","NT1":"NT1",
               "NT2":"NT2","NT3":"NT3","OBC":"OBC","SEBC":"SEBC","EWS":"EWS"}
    if base_category == "EWS":
        return [f"EWS{seat_level}"]
    return [f"{g}{cat_map.get(base_category, base_category)}{seat_level}"]


def get_relevant_categories(base_category, gender, home_university,
                             college_university, special_quotas):
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


def analyse_college_branch(cutoff_df, college_name, course_name,
                            eligible_categories, student_percentile,
                            target_round, trend_adjustment, seat_info=None):
    sub = cutoff_df[
        (cutoff_df['college_name'] == college_name) &
        (cutoff_df['course_name']  == course_name)
    ]
    if sub.empty: return None

    # Stage I only
    if 'stage' in sub.columns:
        s1 = sub[sub['stage'].astype(str).str.strip().str.upper() == 'I']
        if not s1.empty: sub = s1

    valid_cats = filter_categories_by_seats(eligible_categories, seat_info or {})
    best_result, best_prob = None, -1

    for cat in valid_cats:
        cat_rows = sub[sub['category'] == cat]
        if cat_rows.empty: continue
        yearly = (cat_rows.groupby(['year','cap_round'])
                  .agg(percentile=('percentile','min')).reset_index())
        predicted = predict_cutoff_for_round(yearly, target_round, trend_adjustment)
        if predicted is None: continue
        this_round = yearly[yearly['cap_round'] == target_round]
        historical = (this_round if not this_round.empty else yearly).sort_values('year')['percentile'].tolist()
        cap_seats  = (seat_info or {}).get('cap_seats', 60)
        prob       = compute_probability(student_percentile, predicted, historical, cap_seats)
        cl         = classify(prob)
        is_special = cat.upper() in SPECIAL_QUOTA_CODES or cat.upper().startswith('EWS')

        if prob > best_prob:
            best_prob = prob
            si = seat_info or {}
            best_result = {
                "college_name": college_name, "course_name": course_name,
                "best_category": cat, "is_special_quota": is_special,
                "predicted_cutoff": predicted, "historical_cutoffs": historical,
                "rounds_available": sorted(yearly['cap_round'].unique().tolist()),
                "probability": prob, "classification": cl,
                "gap": round(student_percentile - predicted, 2),
                "trend": _detect_trend(historical), "data_years": len(historical),
                "cap_seats": si.get('cap_seats',0), "sl_seats": si.get('sl_total',0),
                "hu_seats": si.get('hu_total',0),   "ohu_seats": si.get('ohu_total',0),
                "has_hu_seats": si.get('has_hu',False),
                "tfws_seats": si.get('tfws_seats',0), "ews_seats": si.get('ews_seats',0),
                "pwd_seats": si.get('pwd_total',0),   "def_seats": si.get('def_total',0),
                "has_seat_data": bool(si),
            }
    return best_result


def generate_all_predictions(cutoff_df, seat_matrix_df, student_percentile,
                              base_category, gender, home_university, special_quotas,
                              preferred_branches, college_type_filter, target_round,
                              trend_adjustment, branch_priority, university_map,
                              seat_lookup=None):
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
        cu      = university_map.get(college,"")
        cats    = get_relevant_categories(base_category, gender, home_university, cu, special_quotas)
        si      = seat_lookup.get((_norm_str(college), _norm_str(branch)), {})
        res     = analyse_college_branch(fdf, college, branch, cats,
                                          student_percentile, target_round,
                                          trend_adjustment, si)
        if res:
            res['status']             = row['status']
            res['college_tier']       = _college_tier(row['status'])
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


def generate_preference_list(predictions_df, max_list=10):
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


# ─── Rebuilt Float/Freeze logic ───────────────────────────────────────────────

def float_freeze_advice(
    current_college: str,
    current_branch: str,
    current_probability: float,
    current_status: str,
    current_round: int,
    predictions_df: pd.DataFrame,
    cutoff_df: pd.DataFrame,
    student_percentile: float,
    preferred_branches: list,
    trend_adjustment: float,
    seat_lookup: dict,
    university_map: dict,
    base_category: str,
    gender: str,
    home_university: str,
    special_quotas: list,
) -> dict:
    """
    Rebuilt Float/Freeze/Slide logic with proper criteria:

    FLOAT criteria (in Maharashtra CAP, floating is SAFE — you keep current seat):
      - There exist options in next round with meaningfully higher probability (>10pts gap)
      - OR current allocation is Target/Dream/Reach (probability <70%) AND better exists
      - OR better options match student's preferred branches

    SLIDE criteria:
      - Student is at a good college (tier 1-3) but not in preferred branch
      - Same college has preferred branch with reasonable probability (>35%)
      - No significantly better college available

    FREEZE criteria:
      - Current is Assured/Safe AND no meaningfully better options in next round
      - OR current is in preferred branch at a top college

    Key insight: In Maharashtra CAP, FLOAT has ZERO downside — you keep your
    current seat while trying for better. The only real risk is SLIDE (you
    might lose your current branch). So the bar for recommending FLOAT is low.
    """
    if predictions_df.empty:
        return {"advice":"FREEZE","reason":"No data to compare against.","factors":[]}

    next_round = current_round + 1
    current_cl = classify(current_probability)
    current_tier = _college_tier(current_status)

    # Recompute probabilities for NEXT round for all options
    # This is the key fix — we can't use current round probabilities to advise on next round
    next_round_preds = []
    for _, row in predictions_df.iterrows():
        cu   = university_map.get(row['college_name'],"")
        elig = get_relevant_categories(base_category, gender, home_university, cu, special_quotas)
        si_k = (_norm_str(row['college_name']), _norm_str(row['course_name']))
        si   = seat_lookup.get(si_k, {})
        res  = analyse_college_branch(
            cutoff_df, row['college_name'], row['course_name'],
            elig, student_percentile, next_round, trend_adjustment, si
        )
        if res:
            res['college_tier']      = _college_tier(row.get('status',''))
            res['is_preferred']      = row['course_name'] in (preferred_branches or [])
            res['is_home_university']= row.get('is_home_university', False)
            next_round_preds.append(res)

    if not next_round_preds:
        # No data for next round — safe to freeze
        return {
            "advice": "FREEZE",
            "reason": f"No data available for Round {next_round}. Your current seat is secure.",
            "factors": ["No next-round data available"]
        }

    nr_df = pd.DataFrame(next_round_preds)

    # Filter out current allocation
    others = nr_df[~((nr_df['college_name']==current_college) &
                      (nr_df['course_name']==current_branch))]

    # ── Factor 1: Better probability options in next round ─────────────────
    # Meaningful improvement = >10 percentile points higher probability
    meaningfully_better = others[others['probability'] > current_probability + 10]

    # ── Factor 2: Better tier college ─────────────────────────────────────
    better_tier = others[
        (others['college_tier'] < current_tier) &
        (others['probability'] >= 40)
    ]

    # ── Factor 3: Preferred branch match ───────────────────────────────────
    current_is_preferred = current_branch in (preferred_branches or [])
    better_preferred = others[
        (others['is_preferred'] == True) &
        (others['probability'] >= 35) &
        (others['college_name'] != current_college)
    ] if preferred_branches else pd.DataFrame()

    # ── Factor 4: Same college, better branch (SLIDE candidate) ────────────
    same_college_better_branch = nr_df[
        (nr_df['college_name'] == current_college) &
        (nr_df['course_name']  != current_branch) &
        (nr_df['probability']  >= 35)
    ]
    preferred_branch_same_college = same_college_better_branch[
        same_college_better_branch['is_preferred'] == True
    ] if preferred_branches else pd.DataFrame()

    # ── Decision logic ─────────────────────────────────────────────────────

    factors = []

    # SLIDE: at good college, preferred branch available here, no better college out there
    if (current_tier <= 4 and
        not preferred_branch_same_college.empty and
        meaningfully_better.empty and
        better_tier.empty):
        top_slide = preferred_branch_same_college.nlargest(3, 'probability')
        factors.append(f"Your college ({current_college[:30]}) is strong")
        factors.append(f"Your preferred branch is available here with ≥35% probability")
        factors.append("No significantly better colleges found for next round")
        return {
            "advice": "SLIDE",
            "reason": (f"You're at a good college. Your preferred branch "
                       f"**{top_slide.iloc[0]['course_name']}** may open up here in Round {next_round} "
                       f"({top_slide.iloc[0]['probability']:.0f}% probability). "
                       f"Sliding is the better move than floating to a different college."),
            "factors": factors,
            "slide_options": top_slide[['course_name','probability']].to_dict('records'),
        }

    # FLOAT: safe in Maharashtra (you keep current seat) — recommend if any meaningful gain
    float_reasons = []
    float_options = []

    if not meaningfully_better.empty:
        float_reasons.append(f"{len(meaningfully_better)} options have >10% higher probability in Round {next_round}")
        top = meaningfully_better.nlargest(3, 'probability')
        float_options.extend(top[['college_name','course_name','probability','college_tier']].to_dict('records'))

    if not better_tier.empty and better_tier.iloc[0]['college_tier'] < current_tier:
        float_reasons.append(f"Better ranked colleges are reachable in Round {next_round}")
        top_t = better_tier.nlargest(2,'probability')
        for r in top_t.to_dict('records'):
            if r not in float_options:
                float_options.append(r)

    if not better_preferred.empty and not current_is_preferred:
        float_reasons.append(f"Your preferred branch is available at other colleges in Round {next_round}")
        top_p = better_preferred.nlargest(2,'probability')
        for r in top_p.to_dict('records'):
            if r not in float_options:
                float_options.append(r)

    if float_reasons:
        # Deduplicate float_options
        seen, uniq = set(), []
        for o in float_options:
            k = (o['college_name'], o['course_name'])
            if k not in seen:
                seen.add(k)
                uniq.append(o)
        uniq.sort(key=lambda x: -x['probability'])

        important_note = ("Remember: in Maharashtra CAP, **floating is risk-free** — "
                          "you keep your current seat as a backup.")
        return {
            "advice": "FLOAT",
            "reason": f"{' | '.join(float_reasons)}. {important_note}",
            "factors": float_reasons,
            "top_options": uniq[:4],
        }

    # FREEZE
    if current_cl['label'] in ['Assured','Safe']:
        factors.append(f"Current probability is {current_probability:.0f}% ({current_cl['label']})")
        factors.append("No meaningfully better options found for next round")
        if current_is_preferred:
            factors.append("You're already in your preferred branch")
        reason = (f"Your allocation is strong at {current_probability:.0f}% probability. "
                  f"No options in Round {next_round} offer a significant improvement. Lock it in.")
    else:
        factors.append(f"Current probability is {current_probability:.0f}% ({current_cl['label']})")
        factors.append("No better options available even in next round")
        reason = (f"Your current allocation has {current_probability:.0f}% probability. "
                  f"While it's not ideal, no better realistic options appear in Round {next_round}. "
                  f"Consider accepting this and using ACAP for better options.")

    return {"advice":"FREEZE", "reason": reason, "factors": factors}
