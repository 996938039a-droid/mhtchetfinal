"""
app.py — MHT-CET College Preference Advisor v7
"""
import os, sys, re as _re
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import (
    load_all_cutoffs, load_seat_matrix, load_config,
    get_available_branches, build_seat_lookup,
)
from src.probability_engine import (
    get_round_data_status, generate_all_predictions, generate_preference_list,
    float_freeze_advice, classify, analyse_college_branch, get_relevant_categories,
    SPECIAL_QUOTA_CODES,
)
from src.export import generate_pdf

st.set_page_config(page_title="MHT-CET Advisor", page_icon="🎓",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
*,*::before,*::after{box-sizing:border-box}
.stApp{background:var(--bg)!important}

/* ── CSS variables — light default ── */
:root, [data-theme="light"] {
  --bg:#F7F9FC; --surface:#fff; --surface2:#F0F4F8;
  --border:#E2E8F0; --text:#1A202C; --text2:#4A5568; --text3:#718096;
  --accent:#2563EB; --accent2:#1E3A5F;
}
/* Dark override via Streamlit's own dark class */
[data-testid="stAppViewContainer"][class*="dark"] { --bg:#0F1117; }
@media(prefers-color-scheme:dark){
  :root{--bg:#0F1117;--surface:#1E2130;--surface2:#262B3D;
        --border:#2D3748;--text:#F0F4F8;--text2:#CBD5E0;--text3:#A0AEC0;
        --accent:#60A5FA;--accent2:#93C5FD}
}

/* ── Header — full width ── */
.page-header{
  width:100%;padding:20px 0 8px 0;
  display:flex;flex-direction:column;gap:4px;
}
.page-title{font-size:1.4rem;font-weight:800;color:var(--text);letter-spacing:-.02em;margin:0}
.page-sub{font-size:0.8rem;color:var(--text3);margin:0}
.chip-row{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.chip{background:var(--surface);border:1px solid var(--border);border-radius:20px;
      padding:3px 10px;font-size:0.75rem;color:var(--text2)}
.thin-hr{border:none;border-top:1px solid var(--border);margin:10px 0}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:var(--accent2)!important}
[data-testid="stSidebar"] *{color:#E2E8F0!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{color:#fff!important}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15)!important}

/* ── Form section label ── */
.fsec{font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
      color:var(--text3);margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--border)}

/* ── Stat cards ── */
.scard{background:var(--surface);border-radius:10px;padding:12px 16px;
       border:1px solid var(--border);border-top:3px solid #ccc;text-align:center}
.scard-val{font-size:1.7rem;font-weight:800;margin:0}
.scard-lbl{font-size:.7rem;color:var(--text3);margin:2px 0 0}

/* ── Preference row ── */
.prow-name{font-size:.95rem;font-weight:700;color:var(--text);margin:0}
.prow-sub{font-size:.8rem;color:var(--text2);margin:2px 0 0}
.prow-meta{font-size:.72rem;color:var(--text3);margin:3px 0 0}
.prow-num{font-size:1.3rem;font-weight:900;color:var(--accent2);text-align:center}

/* ── Badge ── */
.badge{display:inline-block;padding:3px 9px;border-radius:20px;
       font-size:.7rem;font-weight:700;color:#fff!important}
.quota-badge{display:inline-block;padding:2px 7px;border-radius:4px;
             font-size:.65rem;font-weight:700;background:#7C3AED;color:#fff!important}

/* ── Advice boxes ── */
.adv{border-radius:10px;padding:15px 18px;margin:10px 0}
.adv-freeze{background:#ECFDF5;border:1px solid #86EFAC}
.adv-float{background:#EFF6FF;border:1px solid #93C5FD}
.adv-slide{background:#FFF7ED;border:1px solid #FCA5A5}
.adv h3{margin:0 0 5px;font-size:.95rem;color:#1A202C!important}
.adv p{margin:0;font-size:.85rem;color:#374151!important}
.factor-list{margin:8px 0 0;padding:0;list-style:none}
.factor-list li{font-size:.78rem;color:#374151!important;padding:2px 0}
.factor-list li::before{content:"→ ";color:#6B7280}

/* ── Info box ── */
.ibox{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
      padding:10px 14px;font-size:.82rem;color:#1E40AF!important;margin:8px 0}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:var(--surface2);border-radius:8px;padding:3px;gap:2px}
.stTabs [data-baseweb="tab"]{border-radius:6px;padding:5px 13px;font-size:.8rem;
                              font-weight:600;color:var(--text2)!important}
.stTabs [aria-selected="true"]{background:var(--accent2)!important;color:#fff!important}

/* ── College profile table ── */
.cp-header{font-size:.72rem;font-weight:700;color:var(--text3);text-transform:uppercase;
           letter-spacing:.06em;padding:6px 8px;background:var(--surface2)}
.cp-cell{font-size:.8rem;color:var(--text);padding:6px 8px;border-bottom:1px solid var(--border)}

#MainMenu,footer,header{visibility:hidden}
.stButton>button[kind="primary"]{background:var(--accent2)!important;border:none!important;
                                  border-radius:8px!important;font-weight:700!important}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BC = {"Reach":"#EF4444","Dream":"#F97316","Target":"#2563EB","Safe":"#16A34A","Assured":"#6B7280"}
BE = {"Reach":"🎯","Dream":"⭐","Target":"✅","Safe":"🛡️","Assured":"🔒"}
TI = {"rising":"📈","falling":"📉","stable":"→"}

def bdg(label, extra_style=""):
    c = BC.get(label,"#999")
    return f"<span class='badge' style='background:{c};{extra_style}'>{BE.get(label,'')} {label}</span>"

def get_label(cl):
    return cl['label'] if isinstance(cl,dict) else str(cl)

def norm_key(s):
    s = str(s).lower().strip()
    s = _re.sub(r"[,.'\"()]",'',s)
    s = _re.sub(r'\s+',' ',s)
    return s

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    c = load_all_cutoffs(); sm = load_seat_matrix()
    cfg = load_config(); sl = build_seat_lookup(sm)
    return c, sm, cfg, sl

@st.cache_data(show_spinner=False)
def build_umap(_cdf, _cfg):
    if _cdf.empty: return {}
    kw = _cfg.get("university_college_keywords",{})
    out = {}
    for college in _cdf['college_name'].unique():
        cl = college.lower()
        out[college] = next((u for u,ks in kw.items() if any(k.lower() in cl for k in ks)),"")
    return out

with st.spinner(""):
    cutoff_df, seat_matrix_df, config, seat_lookup = get_data()

university_map = build_umap(cutoff_df, config)
categories_map = config.get("categories",{})
district_univ  = config.get("district_university_map",{})
special_q_map  = config.get("special_quotas",{})
all_branches   = get_available_branches(cutoff_df)
all_colleges   = sorted(cutoff_df['college_name'].unique()) if not cutoff_df.empty else []

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("predictions",pd.DataFrame()), ("preference_list",pd.DataFrame()),
    ("student_profile",{}), ("results_ready",False),
    ("trend_preset",0.0), ("sidebar_open",False),
    ("f_class",["Dream","Target","Safe","Assured"]),
    ("f_branch",[]), ("f_minp",0),
]:
    if k not in st.session_state: st.session_state[k] = v

predictions     = st.session_state.predictions
pref_list       = st.session_state.preference_list
student_profile = st.session_state.student_profile
results_ready   = st.session_state.results_ready

# ── Sidebar ───────────────────────────────────────────────────────────────────
# Always render sidebar when results_ready OR sidebar_open toggled
if results_ready or st.session_state.sidebar_open:
    with st.sidebar:
        st.markdown("### 🎓 Edit Profile")
        st.caption("Change any value and click Update")
        st.markdown("---")

        def _idx(lst, val, default=0):
            try: return list(lst).index(val)
            except ValueError: return default

        sb_pct  = st.number_input("Percentile", 0.0, 100.0,
                    float(student_profile.get("percentile",85.0)),
                    step=0.01, format="%.2f", key="sb_pct")
        sb_cat  = st.selectbox("Category", list(categories_map.keys()),
                    index=_idx(list(categories_map.keys()),
                               student_profile.get("raw_category",list(categories_map.keys())[0])),
                    format_func=lambda k: categories_map[k], key="sb_cat")
        sb_gen  = st.radio("Gender", ["male","female"],
                    index=_idx(["male","female"], student_profile.get("gender","male")),
                    format_func=str.capitalize, horizontal=True, key="sb_gen")
        sb_dist = st.selectbox("District", sorted(district_univ.keys()),
                    index=_idx(sorted(district_univ.keys()),
                               student_profile.get("district",sorted(district_univ.keys())[0])),
                    key="sb_dist")
        sb_hu   = district_univ.get(sb_dist,"")
        sb_sq   = st.multiselect("Special Quota", list(special_q_map.keys()),
                    default=student_profile.get("special_quotas",[]),
                    format_func=lambda k: special_q_map[k], key="sb_sq")
        sb_br   = st.multiselect("Preferred Branches", all_branches,
                    default=student_profile.get("branches",[]), key="sb_br")
        sb_bpri = st.radio("Sort By", ["Branch First","College First"],
                    index=0 if student_profile.get("branch_priority",True) else 1,
                    key="sb_bpri") == "Branch First"
        sb_near = st.toggle("📍 Prioritise Nearby",
                    student_profile.get("prioritise_nearby",False), key="sb_near")
        sb_types = st.multiselect("College Types",
                    ["Government","Government Autonomous","Government-Aided",
                     "Government-Aided Autonomous","Un-Aided","Un-Aided Autonomous",
                     "University Department","University Managed"],
                    default=student_profile.get("college_types",
                        ["Government","Government Autonomous",
                         "Government-Aided","Government-Aided Autonomous"]),
                    key="sb_types")
        sb_rnd  = st.selectbox("CAP Round", [1,2,3],
                    index=student_profile.get("cap_round",1)-1,
                    format_func=lambda x: f"Round {x}", key="sb_rnd")

        st.markdown("**Trend Adjustment**")
        pc1,pc2,pc3 = st.columns(3)
        if pc1.button("📉−2", use_container_width=True, key="sb_e"):
            st.session_state.trend_preset = -2.0
        if pc2.button("→0",  use_container_width=True, key="sb_n"):
            st.session_state.trend_preset = 0.0
        if pc3.button("📈+2", use_container_width=True, key="sb_h"):
            st.session_state.trend_preset = 2.0
        sb_tadj = st.slider("Fine-tune",-5.0,5.0,
                    float(st.session_state.trend_preset),0.5,format="%.1f",key="sb_tadj")
        sb_maxp = st.slider("Max Preferences",5,20,
                    student_profile.get("max_pref",10), key="sb_maxp")
        st.markdown("---")
        rerun_btn = st.button("🔄 Update Results", type="primary", use_container_width=True, key="rerun")

        rds = get_round_data_status(cutoff_df)
        st.markdown("**Data**")
        for rn in [1,2,3]:
            st.caption(f"{'✅' if rds.get(rn) else '⚠️'} Round {rn}")
        yrs = rds.get('years',[])
        if yrs: st.caption(f"{', '.join(str(y) for y in yrs)}")
else:
    rerun_btn = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <p class='page-title'>🎓 MHT-CET College Preference Advisor</p>
  <p class='page-sub'>Maharashtra Engineering Admissions · CAP Round Analysis · 2022–2024 Data</p>
</div>
""", unsafe_allow_html=True)

if not cutoff_df.empty:
    rds = get_round_data_status(cutoff_df)
    yrs = rds.get('years',[])
    st.markdown(
        f"<div class='chip-row'>"
        f"<span class='chip'>📅 {', '.join(str(y) for y in yrs)}</span>"
        f"<span class='chip'>🔄 Rounds {', '.join(str(r) for r in [1,2,3] if rds.get(r))}</span>"
        f"<span class='chip'>🏛️ {cutoff_df['college_name'].nunique()} colleges</span>"
        f"<span class='chip'>📚 {cutoff_df['course_name'].nunique()} branches</span>"
        f"</div>", unsafe_allow_html=True
    )
else:
    st.error("No data found. Place cutoff Excel files in `data/cutoffs/`.")
    st.stop()

st.markdown("<div class='thin-hr'></div>", unsafe_allow_html=True)

# ── Handle sidebar update ─────────────────────────────────────────────────────
if rerun_btn and results_ready:
    with st.spinner("Updating…"):
        preds = generate_all_predictions(
            cutoff_df=cutoff_df, seat_matrix_df=seat_matrix_df,
            student_percentile=sb_pct, base_category=sb_cat,
            gender=sb_gen, home_university=sb_hu,
            special_quotas=sb_sq, preferred_branches=sb_br,
            college_type_filter=sb_types, target_round=sb_rnd,
            trend_adjustment=sb_tadj, branch_priority=sb_bpri,
            university_map=university_map, seat_lookup=seat_lookup,
        )
        if sb_near and not preds.empty:
            preds['nearby_boost'] = preds['is_home_university'].apply(lambda x:0 if x else 1)
            sc = (['branch_rank','nearby_boost','classification_order','probability']
                  if sb_bpri else ['nearby_boost','classification_order','branch_rank','probability'])
            preds = preds.sort_values(sc, ascending=[True,True,True,False]).reset_index(drop=True)
        pl = generate_preference_list(preds, max_list=sb_maxp)
    st.session_state.predictions     = preds
    st.session_state.preference_list = pl
    st.session_state.student_profile.update({
        "percentile":sb_pct, "raw_category":sb_cat,
        "category":f"{sb_cat} — {categories_map.get(sb_cat,'')}",
        "gender":sb_gen, "district":sb_dist, "home_university":sb_hu,
        "special_quotas":sb_sq, "branches":sb_br,
        "branch_priority":sb_bpri, "prioritise_nearby":sb_near,
        "college_types":sb_types, "cap_round":sb_rnd,
        "trend_adj":sb_tadj, "max_pref":sb_maxp,
    })
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════
if not results_ready:
    st.markdown("### Enter Your Details")
    st.markdown("<div class='fsec'>Your Profile</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([2,2,2,2])
    with c1:
        percentile = st.number_input("MHT-CET Percentile",0.0,100.0,85.0,step=0.01,format="%.2f")
    with c2:
        category = st.selectbox("Category", list(categories_map.keys()),
                                  format_func=lambda k: categories_map[k])
    with c3:
        gender = st.radio("Gender",["male","female"],format_func=str.capitalize,horizontal=True)
    with c4:
        district = st.selectbox("Home District", sorted(district_univ.keys()))
        home_university = district_univ.get(district,"")
        if home_university: st.caption(f"🏛️ {home_university}")

    c5,c6 = st.columns(2)
    with c5:
        special_quotas = st.multiselect("Special Quota (if eligible)",
                                         list(special_q_map.keys()),
                                         format_func=lambda k: special_q_map[k])
    with c6:
        preferred_branches = st.multiselect("Preferred Branches (priority order)", all_branches)

    st.markdown("<div class='fsec'>Filters & Settings</div>", unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)
    with d1:
        branch_priority = st.radio("Sort By",["Branch First","College First"]) == "Branch First"
        prioritise_nearby = st.toggle("📍 Prioritise Colleges Near Me", False)
    with d2:
        college_types = st.multiselect("College Types",
            ["Government","Government Autonomous","Government-Aided",
             "Government-Aided Autonomous","Un-Aided","Un-Aided Autonomous",
             "University Department","University Managed"],
            default=["Government","Government Autonomous",
                     "Government-Aided","Government-Aided Autonomous"])
    with d3:
        cap_round = st.selectbox("Target CAP Round",[1,2,3],format_func=lambda x: f"CAP Round {x}")
        max_pref  = st.slider("Max Preferences",5,20,10)

    st.markdown("<div class='fsec'>Cutoff Trend Adjustment for 2025</div>", unsafe_allow_html=True)
    st.caption("Scales non-linearly — larger effect at lower percentiles, compressed at 95+")
    ta1,ta2,ta3,ta4 = st.columns([1,1,1,3])
    with ta1:
        if st.button("📉 Easier (−2)"): st.session_state.trend_preset = -2.0
    with ta2:
        if st.button("→ Same (0)"):     st.session_state.trend_preset = 0.0
    with ta3:
        if st.button("📈 Harder (+2)"): st.session_state.trend_preset = 2.0
    with ta4:
        trend_adj = st.slider("Fine-tune",-5.0,5.0,
                               float(st.session_state.trend_preset),0.5,
                               format="%.1f", label_visibility="collapsed")

    st.markdown("")
    run_btn = st.button("🔍 Generate Recommendations", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Analysing data across all years and rounds…"):
            preds = generate_all_predictions(
                cutoff_df=cutoff_df, seat_matrix_df=seat_matrix_df,
                student_percentile=percentile, base_category=category,
                gender=gender, home_university=home_university,
                special_quotas=special_quotas, preferred_branches=preferred_branches,
                college_type_filter=college_types, target_round=cap_round,
                trend_adjustment=trend_adj, branch_priority=branch_priority,
                university_map=university_map, seat_lookup=seat_lookup,
            )
            if prioritise_nearby and not preds.empty:
                preds['nearby_boost'] = preds['is_home_university'].apply(lambda x:0 if x else 1)
                sc = (['branch_rank','nearby_boost','classification_order','probability']
                      if branch_priority else
                      ['nearby_boost','classification_order','branch_rank','probability'])
                preds = preds.sort_values(sc,ascending=[True,True,True,False]).reset_index(drop=True)
            pl = generate_preference_list(preds, max_list=max_pref)
        st.session_state.predictions     = preds
        st.session_state.preference_list = pl
        st.session_state.results_ready   = True
        st.session_state.student_profile = {
            "percentile":percentile, "raw_category":category,
            "category":f"{category} — {categories_map.get(category,'')}",
            "gender":gender, "district":district, "home_university":home_university,
            "special_quotas":special_quotas, "branches":preferred_branches,
            "branch_priority":branch_priority, "prioritise_nearby":prioritise_nearby,
            "college_types":college_types, "cap_round":cap_round,
            "trend_adj":trend_adj, "max_pref":max_pref,
        }
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS VIEW
# ═══════════════════════════════════════════════════════════════════════════════
else:
    predictions     = st.session_state.predictions
    pref_list       = st.session_state.preference_list
    student_profile = st.session_state.student_profile
    sp = student_profile

    # Profile bar + Edit button
    cp, ce = st.columns([9,1])
    with cp:
        st.markdown(
            f"<div class='chip-row'>"
            f"<span class='chip'>🎯 {sp.get('percentile','—')} pct</span>"
            f"<span class='chip'>{sp.get('category','—')}</span>"
            f"<span class='chip'>{str(sp.get('gender','')).capitalize()}</span>"
            f"<span class='chip'>📍 {sp.get('district','—')}</span>"
            f"<span class='chip'>Round {sp.get('cap_round','—')}</span>"
            f"<span class='chip'>Trend {sp.get('trend_adj',0):+.1f} pts</span>"
            f"</div>", unsafe_allow_html=True
        )
    with ce:
        # Toggle sidebar open/close
        sidebar_label = "✕ Close" if st.session_state.sidebar_open else "✏️ Edit"
        if st.button(sidebar_label, key="edit_btn"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()

    # Stat cards
    if not predictions.empty:
        ls = predictions['classification'].apply(get_label)
        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        for col,key,color in [
            (sc1,"Reach","#EF4444"),(sc2,"Dream","#F97316"),
            (sc3,"Target","#2563EB"),(sc4,"Safe","#16A34A"),(sc5,"Assured","#6B7280")
        ]:
            col.markdown(
                f"<div class='scard' style='border-top-color:{color}'>"
                f"<div class='scard-val' style='color:{color}'>{(ls==key).sum()}</div>"
                f"<div class='scard-lbl'>{BE.get(key,'')} {key}</div></div>",
                unsafe_allow_html=True
            )
        st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📋 Preference List","🔍 All Options","📊 Round Analysis",
        "⚖️ Float / Freeze","🏛️ College Profile","📖 ACAP Guide","📤 Export",
    ])

    # ═══ TAB 1 — PREFERENCE LIST ════════════════════════════════════════
    with tab1:
        if pref_list.empty:
            st.markdown('<div class="ibox">No matches. Adjust your filters.</div>',
                        unsafe_allow_html=True)
        else:
            # Legend
            lc = st.columns(5)
            for col,(lbl,color) in zip(lc,BC.items()):
                col.markdown(f"<div style='text-align:center;padding:4px 0'>"
                             f"<span class='badge' style='background:{color}'>"
                             f"{BE[lbl]} {lbl}</span></div>", unsafe_allow_html=True)
            st.markdown("<div class='thin-hr'></div>", unsafe_allow_html=True)

            for idx, row in pref_list.iterrows():
                lbl   = get_label(row['classification'])
                color = BC.get(lbl,"#999")
                prob  = row.get('probability',0)
                cut   = row.get('predicted_cutoff',0)
                gap   = row.get('gap',0)
                trend = TI.get(row.get('trend','stable'),"")
                is_hu = row.get('is_home_university',False)
                is_sq = row.get('is_special_quota', False)
                gc    = "#16A34A" if gap>=0 else "#DC2626"
                sign  = "+" if gap>=0 else ""

                # Seat info
                cap_s = int(row.get('cap_seats',0))
                hu_s  = int(row.get('hu_seats',0))
                oh_s  = int(row.get('ohu_seats',0))
                sl_s  = int(row.get('sl_seats',0))
                if row.get('has_seat_data') and cap_s>0:
                    if hu_s>0 and oh_s>0:
                        seat_s = f"HU:{hu_s} · OHU:{oh_s}"
                    elif sl_s>0:
                        seat_s = f"SL:{sl_s} seats"
                    else:
                        seat_s = f"{cap_s} seats"
                    tfws=int(row.get('tfws_seats',0)); ews=int(row.get('ews_seats',0))
                    if tfws: seat_s += f" · TFWS:{tfws}"
                    if ews:  seat_s += f" · EWS:{ews}"
                else:
                    seat_s = "—"

                # Category display with quota badge
                cat_code = row.get('best_category','')
                if is_sq:
                    cat_display = (f"{cat_code} <span class='quota-badge'>QUOTA</span>")
                else:
                    cat_display = cat_code

                hu_chip = (" <span style='background:#DBEAFE;color:#1E40AF;border-radius:4px;"
                           "padding:1px 5px;font-size:.62rem;font-weight:700'>🏠 HU</span>") if is_hu else ""

                cn,ci,cs,cb = st.columns([0.5,5,2.8,1.5])
                with cn:
                    st.markdown(f"<div style='text-align:center;padding-top:8px'>"
                                f"<span class='prow-num'>{idx}</span></div>",
                                unsafe_allow_html=True)
                with ci:
                    st.markdown(
                        f"<p class='prow-name'>{row['college_name']}{hu_chip}</p>"
                        f"<p class='prow-sub'>📚 {row['course_name']}</p>"
                        f"<p class='prow-meta'>🏷️ {cat_display} · {row.get('status','')} · {seat_s}</p>",
                        unsafe_allow_html=True
                    )
                with cs:
                    st.markdown(
                        f"<div style='padding-top:6px;text-align:right'>"
                        f"<p style='margin:0;font-size:.82rem;color:var(--text2)'>"
                        f"Cutoff: <strong>{cut:.2f}</strong> {trend}</p>"
                        f"<p style='margin:3px 0 0;font-size:.82rem;color:var(--text2)'>"
                        f"Gap: <strong style='color:{gc}'>{sign}{gap:.2f}</strong>"
                        f" · Prob: <strong>{prob:.0f}%</strong></p></div>",
                        unsafe_allow_html=True
                    )
                with cb:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:10px'>"
                        f"<span class='badge' style='background:{color};padding:5px 12px'>"
                        f"{BE.get(lbl,'')} {lbl}</span></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("<div class='thin-hr'></div>", unsafe_allow_html=True)

            # Bar chart
            bx = list(range(1,len(pref_list)+1))
            by = [r.get('probability',0) for _,r in pref_list.iterrows()]
            bc_list = [BC.get(get_label(r['classification']),"#999") for _,r in pref_list.iterrows()]
            hover = [f"#{i} {r['college_name']}<br>{r['course_name']}<br>{r.get('probability',0):.0f}%"
                     for i,r in pref_list.iterrows()]
            fig = go.Figure(go.Bar(
                x=bx, y=by, marker_color=bc_list,
                text=[f"{p:.0f}%" for p in by], textposition='outside',
                textfont=dict(size=10),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            ))
            fig.add_hline(y=70,line_dash="dash",line_color="#16A34A",line_width=1.5,
                          annotation_text="Safe(70%)",annotation_font_size=9)
            fig.add_hline(y=30,line_dash="dash",line_color="#F97316",line_width=1.5,
                          annotation_text="Dream(30%)",annotation_font_size=9)
            fig.update_layout(
                height=250,margin=dict(t=20,b=20,l=30,r=20),
                xaxis=dict(title="Pref #"),yaxis=dict(range=[0,118],title="Prob (%)"),
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="ibox">💡 Float is risk-free in Maharashtra CAP — '
                        'you keep your current seat while trying for Dream options in later rounds.</div>',
                        unsafe_allow_html=True)

    # ═══ TAB 2 — ALL OPTIONS ════════════════════════════════════════════
    with tab2:
        if predictions.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>',unsafe_allow_html=True)
        else:
            fc1,fc2,fc3 = st.columns(3)
            with fc1:
                fc = st.multiselect("Classification",
                                     ["Reach","Dream","Target","Safe","Assured"],
                                     default=st.session_state.f_class, key="t2_class")
                st.session_state.f_class = fc
            with fc2:
                fb = st.multiselect("Branch",
                                     sorted(predictions['course_name'].unique()),
                                     default=[b for b in st.session_state.f_branch
                                              if b in predictions['course_name'].unique()],
                                     key="t2_branch")
                st.session_state.f_branch = fb
            with fc3:
                fm = st.slider("Min Probability (%)",0,100,st.session_state.f_minp,key="t2_minp")
                st.session_state.f_minp = fm

            disp = predictions.copy()
            ls = disp['classification'].apply(get_label)
            if fc: disp = disp[ls.isin(fc)]
            if fb: disp = disp[disp['course_name'].isin(fb)]
            disp = disp[disp['probability'] >= fm]
            st.caption(f"**{len(disp)}** of {len(predictions)} options")

            rows = []
            for _,r in disp.iterrows():
                lbl = get_label(r['classification'])
                hu_s=int(r.get('hu_seats',0)); oh_s=int(r.get('ohu_seats',0)); sl_s=int(r.get('sl_seats',0))
                ss = (f"HU:{hu_s}/OHU:{oh_s}" if hu_s>0 and oh_s>0
                      else f"SL:{sl_s}" if sl_s>0
                      else str(int(r.get('cap_seats',0))) if r.get('has_seat_data') else "—")
                is_sq = r.get('is_special_quota',False)
                cat_d = f"{r.get('best_category','')} ★" if is_sq else r.get('best_category','')
                rows.append({
                    "College":r['college_name'],"Branch":r['course_name'],
                    "Type":r.get('status',''),"Cat":cat_d,
                    "Cutoff":round(r.get('predicted_cutoff',0),2),
                    "Prob":f"{r.get('probability',0):.0f}%",
                    "Class":f"{BE.get(lbl,'')} {lbl}","Seats":ss,
                    "TFWS":int(r.get('tfws_seats',0)) or "—",
                    "Trend":TI.get(r.get('trend',''),""),
                    "HU":"✅" if r.get('is_home_university') else "",
                    "Yrs":r.get('data_years',0),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=450)
            if not disp.empty:
                fh = px.histogram(disp,x='probability',nbins=20,color_discrete_sequence=["#2563EB"])
                fh.add_vline(x=70,line_dash="dash",line_color="#16A34A",annotation_text="Safe",annotation_font_size=9)
                fh.add_vline(x=30,line_dash="dash",line_color="#F97316",annotation_text="Dream",annotation_font_size=9)
                fh.update_layout(height=200,margin=dict(t=10,b=20,l=10,r=10),
                                  xaxis_title="Probability (%)",yaxis_title="Count",
                                  plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fh, use_container_width=True)

    # ═══ TAB 3 — ROUND ANALYSIS ═════════════════════════════════════════
    with tab3:
        if predictions.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>',unsafe_allow_html=True)
        else:
            r1c,r2c = st.columns(2)
            with r1c:
                ra_col = st.selectbox("College",sorted(predictions['college_name'].unique()),key="ra_col")
            with r2c:
                ra_br  = st.selectbox("Branch",
                    sorted(predictions[predictions['college_name']==ra_col]['course_name'].unique()),
                    key="ra_br")

            round_probs = []
            for rnum in [1,2,3]:
                cu   = university_map.get(ra_col,"")
                elig = get_relevant_categories(
                    sp.get("raw_category","OPEN"),sp.get("gender","male"),
                    sp.get("home_university",""),cu,sp.get("special_quotas",[]))
                si   = seat_lookup.get((norm_key(ra_col),norm_key(ra_br)),{})
                res  = analyse_college_branch(cutoff_df,ra_col,ra_br,elig,
                           sp.get("percentile",85.0),rnum,sp.get("trend_adj",0.0),si)
                if res:
                    round_probs.append({"Round":f"Round {rnum}","r_num":rnum,
                        "Probability":res['probability'],
                        "Predicted Cutoff":res['predicted_cutoff'],
                        "Classification":res['classification']['label']})

            if round_probs:
                rdf = pd.DataFrame(round_probs).sort_values('r_num')
                rc  = [BC.get(r['Classification'],"#2563EB") for r in round_probs]
                fig_r = go.Figure(go.Bar(
                    x=rdf['Round'],y=rdf['Probability'],marker_color=rc,width=0.4,
                    text=[f"{p:.0f}%" for p in rdf['Probability']],textposition='outside',
                    textfont=dict(size=13),
                ))
                fig_r.add_hline(y=70,line_dash="dash",line_color="#16A34A",line_width=1.5,
                                annotation_text="Safe(70%)",annotation_font_size=9)
                fig_r.add_hline(y=30,line_dash="dash",line_color="#F97316",line_width=1.5,
                                annotation_text="Dream(30%)",annotation_font_size=9)
                fig_r.update_layout(
                    title=f"{ra_br} · {ra_col[:55]}",
                    yaxis=dict(range=[0,118],title="Probability (%)"),
                    height=320,margin=dict(t=50,b=20,l=40,r=20),
                    plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_r, use_container_width=True)

                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=rdf['Round'],y=rdf['Predicted Cutoff'],
                    mode='lines+markers+text',line=dict(color="#2563EB",width=2.5),
                    marker=dict(size=9),text=[f"{c:.2f}" for c in rdf['Predicted Cutoff']],
                    textposition="top center",textfont=dict(size=10),
                ))
                fig_c.add_hline(y=sp.get("percentile",85.0),line_dash="dot",line_color="#16A34A",
                                line_width=1.5,
                                annotation_text=f"Your pct: {sp.get('percentile',85.0):.2f}",
                                annotation_font_size=9)
                fig_c.update_layout(
                    title="Cutoff per Round",yaxis=dict(title="Closing Percentile"),
                    height=220,margin=dict(t=35,b=20,l=40,r=20),
                    plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_c, use_container_width=True)

                show = rdf[['Round','Probability','Predicted Cutoff','Classification']].copy()
                show['Probability'] = show['Probability'].apply(lambda x:f"{x:.1f}%")
                show['Predicted Cutoff'] = show['Predicted Cutoff'].apply(lambda x:f"{x:.4f}")
                st.dataframe(show.drop(columns=['r_num'],errors='ignore'),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("Not enough data for this combination.")

            # Heatmap
            st.markdown("---")
            st.caption("**Heatmap — hover over cells for full college and branch name**")
            hdata, seen = [], set()
            for _,row in predictions.head(12).iterrows():
                cu   = university_map.get(row['college_name'],"")
                elig = get_relevant_categories(
                    sp.get("raw_category","OPEN"),sp.get("gender","male"),
                    sp.get("home_university",""),cu,sp.get("special_quotas",[]))
                okey = f"{row['college_name']} ||| {row['course_name']}"
                for rnum in [1,2,3]:
                    if (okey,rnum) in seen: continue
                    seen.add((okey,rnum))
                    si  = seat_lookup.get((norm_key(row['college_name']),norm_key(row['course_name'])),{})
                    res = analyse_college_branch(cutoff_df,row['college_name'],row['course_name'],
                              elig,sp.get("percentile",85.0),rnum,sp.get("trend_adj",0.0),si)
                    if res:
                        hdata.append({"Option":okey,"College":row['college_name'],
                                       "Branch":row['course_name'],
                                       "Round":f"R{rnum}","Probability":res['probability']})

            if hdata:
                hdf    = pd.DataFrame(hdata)
                hdf_g  = hdf.groupby(['Option','Round'],as_index=False)['Probability'].mean()
                if not hdf_g.empty and hdf_g['Option'].nunique()>0:
                    try:
                        hpivot = hdf_g.pivot(index='Option',columns='Round',values='Probability')
                        # Build short display labels for y-axis
                        short = []
                        for opt in hpivot.index:
                            p  = opt.split(" ||| ")
                            cs = (p[0][:25]+"…") if len(p[0])>25 else p[0]
                            bs = (p[1][:18]+"…") if len(p)>1 and len(p[1])>18 else (p[1] if len(p)>1 else "")
                            short.append(f"{cs} / {bs}")

                        # Build customdata with FULL names for hover
                        # customdata shape: (n_rows, n_cols, 2) — college, branch per cell
                        full_names = []
                        for opt in hpivot.index:
                            p = opt.split(" ||| ")
                            full_names.append(f"{p[0]}<br>{p[1] if len(p)>1 else ''}")

                        hpivot.index = short
                        fig_hm = go.Figure(go.Heatmap(
                            z=hpivot.values,
                            x=list(hpivot.columns),
                            y=list(hpivot.index),
                            colorscale="RdYlGn",zmin=0,zmax=100,
                            text=[[f"{v:.0f}%" if not pd.isna(v) else "—"
                                   for v in row] for row in hpivot.values],
                            texttemplate="%{text}",
                            # customdata carries full college+branch name per row
                            customdata=[[full_names[i]]*len(hpivot.columns)
                                        for i in range(len(hpivot))],
                            hovertemplate=(
                                "<b>%{customdata}</b><br>"
                                "Round: %{x}<br>"
                                "Probability: %{z:.1f}%<extra></extra>"
                            ),
                            colorbar=dict(title="Prob %"),
                        ))
                        n_rows = len(hpivot)
                        fig_hm.update_layout(
                            height=max(300, n_rows*42),
                            margin=dict(t=10,b=10,l=10,r=60),
                            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)
                    except Exception:
                        st.info("Not enough data to render heatmap.")

    # ═══ TAB 4 — FLOAT / FREEZE ═════════════════════════════════════════
    with tab4:
        st.markdown("Enter your current allocation to get a personalised recommendation.")
        if predictions.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>',unsafe_allow_html=True)
        else:
            ff1,ff2 = st.columns(2)
            with ff1:
                ff_col = st.selectbox("Allocated College",
                                       sorted(predictions['college_name'].unique()),key="ff_col")
            with ff2:
                ff_brs = predictions[predictions['college_name']==ff_col]['course_name'].unique()
                ff_br  = st.selectbox("Allocated Branch",sorted(ff_brs),key="ff_br")
            ff_rnd = st.selectbox("Round of Allocation",[1,2],
                                   format_func=lambda x:f"Round {x}",key="ff_rnd")

            cu   = university_map.get(ff_col,"")
            elig = get_relevant_categories(
                sp.get("raw_category","OPEN"),sp.get("gender","male"),
                sp.get("home_university",""),cu,sp.get("special_quotas",[]))
            si   = seat_lookup.get((norm_key(ff_col),norm_key(ff_br)),{})
            cur  = analyse_college_branch(cutoff_df,ff_col,ff_br,elig,
                       sp.get("percentile",85.0),ff_rnd,sp.get("trend_adj",0.0),si)

            if cur:
                cur_lbl = get_label(cur['classification'])
                cur_col = BC.get(cur_lbl,"#999")
                cur_status = predictions[predictions['college_name']==ff_col]['status'].iloc[0] \
                             if not predictions[predictions['college_name']==ff_col].empty else ""

                st.markdown(
                    f"<div style='background:var(--surface);border:1px solid var(--border);"
                    f"border-left:4px solid {cur_col};border-radius:10px;"
                    f"padding:14px 18px;margin:10px 0'>"
                    f"<p style='font-size:.72rem;color:var(--text3);margin:0'>CURRENT ALLOCATION</p>"
                    f"<p style='font-weight:800;font-size:1rem;color:var(--text);margin:4px 0'>{ff_col}</p>"
                    f"<p style='font-size:.88rem;color:var(--text2);margin:0'>{ff_br}</p>"
                    f"<div style='margin-top:8px;display:flex;gap:12px;align-items:center'>"
                    f"<span class='badge' style='background:{cur_col}'>"
                    f"{cur['classification'].get('emoji','')} {cur_lbl}</span>"
                    f"<span style='font-weight:700'>{cur['probability']:.0f}% probability</span>"
                    f"<span style='color:var(--text3);font-size:.82rem'>Cutoff: {cur['predicted_cutoff']:.2f}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

                with st.spinner("Analysing next round options…"):
                    adv = float_freeze_advice(
                        current_college=ff_col, current_branch=ff_br,
                        current_probability=cur['probability'],
                        current_status=cur_status, current_round=ff_rnd,
                        predictions_df=predictions, cutoff_df=cutoff_df,
                        student_percentile=sp.get("percentile",85.0),
                        preferred_branches=sp.get("branches",[]),
                        trend_adjustment=sp.get("trend_adj",0.0),
                        seat_lookup=seat_lookup, university_map=university_map,
                        base_category=sp.get("raw_category","OPEN"),
                        gender=sp.get("gender","male"),
                        home_university=sp.get("home_university",""),
                        special_quotas=sp.get("special_quotas",[]),
                    )

                a   = adv['advice']
                cls = {"FREEZE":"adv-freeze","FLOAT":"adv-float","SLIDE":"adv-slide"}[a]
                ico = {"FREEZE":"🔒","FLOAT":"🌊","SLIDE":"↔️"}[a]
                factors_html = "".join(f"<li>{f}</li>" for f in adv.get('factors',[]))
                st.markdown(
                    f"<div class='adv {cls}'>"
                    f"<h3>{ico} Recommendation: {a}</h3>"
                    f"<p>{adv['reason']}</p>"
                    f"{'<ul class=\"factor-list\">'+factors_html+'</ul>' if factors_html else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if a=="FLOAT" and adv.get('top_options'):
                    st.markdown("**Better options in next round:**")
                    for opt in adv['top_options']:
                        lbl2 = classify(opt['probability'])['label']
                        st.markdown(
                            f"- **{opt['college_name']}** / {opt['course_name']} — "
                            f"{opt['probability']:.0f}% {bdg(lbl2)}",
                            unsafe_allow_html=True)
                if a=="SLIDE" and adv.get('slide_options'):
                    st.markdown("**Better branches at same college:**")
                    for opt in adv['slide_options']:
                        st.markdown(f"- **{opt['course_name']}** — {opt['probability']:.0f}%")
            else:
                st.warning("No data for this allocation.")

            st.markdown("---")
            st.markdown("""
| | What happens | When to use |
|--|--|--|
| 🔒 **Freeze** | Accept current seat, exit process | Strong seat, or no better options |
| 🌊 **Float** | Hold current seat, try for better — **zero risk in Maharashtra CAP** | Anytime better options exist |
| ↔️ **Slide** | Same college, try for better branch — **risks current branch** | Happy with college, want better branch |
""")

    # ═══ TAB 5 — COLLEGE PROFILE ════════════════════════════════════════
    with tab5:
        st.markdown("### College Profile")
        st.caption("Full historical cutoff data for every category, every round, every year")

        cp_col_sel = st.selectbox("Select College", all_colleges, key="cp_college")

        if not cutoff_df.empty and cp_col_sel:
            cdf = cutoff_df[cutoff_df['college_name'] == cp_col_sel].copy()

            # Stage I only
            if 'stage' in cdf.columns:
                s1 = cdf[cdf['stage'].astype(str).str.strip().str.upper() == 'I']
                if not s1.empty: cdf = s1

            branches_avail = sorted(cdf['course_name'].unique())
            cp_br_sel = st.selectbox("Select Branch", branches_avail, key="cp_branch")

            bdf = cdf[cdf['course_name'] == cp_br_sel]

            if bdf.empty:
                st.info("No data for this branch.")
            else:
                years_avail  = sorted(bdf['year'].unique())
                rounds_avail = sorted(bdf['cap_round'].unique())
                cats_avail   = sorted(bdf['category'].unique())

                # Summary chips
                si_key = (norm_key(cp_col_sel), norm_key(cp_br_sel))
                si_info = seat_lookup.get(si_key, {})
                cap_s   = int(si_info.get('cap_seats',0))
                st.markdown(
                    f"<div class='chip-row'>"
                    f"<span class='chip'>🏛️ {cdf['status'].iloc[0] if len(cdf)>0 else '—'}</span>"
                    f"<span class='chip'>📅 {', '.join(str(y) for y in years_avail)}</span>"
                    f"<span class='chip'>🔄 Rounds {', '.join(str(r) for r in rounds_avail)}</span>"
                    f"{'<span class=\"chip\">🪑 '+str(cap_s)+' seats</span>' if cap_s>0 else ''}"
                    f"</div>", unsafe_allow_html=True
                )

                # Category filter
                cp_cats = st.multiselect(
                    "Filter categories (leave blank for all)",
                    cats_avail, key="cp_cats"
                )
                show_cats = cp_cats if cp_cats else cats_avail

                # Build pivot: rows = category, cols = (year, round)
                pivot_rows = []
                for cat in show_cats:
                    cat_data = bdf[bdf['category'] == cat]
                    row_d = {"Category": cat}
                    for year in years_avail:
                        for rnd in rounds_avail:
                            cell = cat_data[
                                (cat_data['year']==year) &
                                (cat_data['cap_round']==rnd)
                            ]
                            col_name = f"{year} R{rnd}"
                            if not cell.empty:
                                row_d[col_name] = round(cell['percentile'].min(), 2)
                            else:
                                row_d[col_name] = None
                    pivot_rows.append(row_d)

                if pivot_rows:
                    pivot_df = pd.DataFrame(pivot_rows).set_index("Category")
                    # Colour-code the dataframe
                    st.dataframe(
                        pivot_df.style.background_gradient(
                            cmap='RdYlGn_r', axis=None,
                            subset=[c for c in pivot_df.columns]
                        ).format("{:.2f}", na_rep="—"),
                        use_container_width=True,
                        height=min(600, 60 + len(pivot_df)*36),
                    )
                    st.caption("Values show closing percentile (lower = easier to get). "
                               "Green = lower cutoff. Red = higher cutoff.")

                    # Trend chart for selected categories
                    st.markdown("**Closing Percentile Trend — Round 1 across Years**")
                    fig_trend = go.Figure()
                    for cat in show_cats[:8]:  # max 8 lines for readability
                        cat_data = bdf[(bdf['category']==cat) & (bdf['cap_round']==1)]
                        if cat_data.empty: continue
                        yr_data = cat_data.groupby('year')['percentile'].min().reset_index()
                        if len(yr_data) < 1: continue
                        fig_trend.add_trace(go.Scatter(
                            x=yr_data['year'].astype(str),
                            y=yr_data['percentile'],
                            mode='lines+markers+text',
                            name=cat,
                            text=[f"{v:.1f}" for v in yr_data['percentile']],
                            textposition="top center",
                            textfont=dict(size=9),
                            line=dict(width=2),
                            marker=dict(size=7),
                        ))
                    fig_trend.update_layout(
                        height=320, margin=dict(t=20,b=20,l=40,r=20),
                        xaxis_title="Year", yaxis_title="Closing Percentile",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # Seat info if available
                    if si_info:
                        st.markdown("**Seat Breakdown (2024 Seat Matrix)**")
                        seat_cols = st.columns(4)
                        seat_cols[0].metric("Total CAP Seats", int(si_info.get('cap_seats',0)))
                        seat_cols[1].metric("HU Seats", int(si_info.get('hu_total',0)))
                        seat_cols[2].metric("OHU Seats", int(si_info.get('ohu_total',0)))
                        seat_cols[3].metric("TFWS Seats", int(si_info.get('tfws_seats',0)))

    # ═══ TAB 6 — ACAP ═══════════════════════════════════════════════════
    with tab6:
        st.markdown("### ACAP — Autonomous College Admission Process")
        st.markdown("""
After CAP Round 3, colleges with **Autonomous** status conduct their own counselling for remaining seats.

**Who can participate:** Students not satisfied with CAP Round 3, or those who got no seat.

**How it works:**
1. Autonomous colleges list their vacant seats after CAP Round 3
2. Each college announces its own schedule on their website (October–November)
3. You attend direct counselling — bring all original documents
4. Merit = your MHT-CET percentile — no new exam needed
5. Same Maharashtra reservation rules apply

| Tip | Detail |
|-----|--------|
| 📅 Monitor college websites | No central schedule — each college posts independently |
| 🎯 Realistic cutoffs | ACAP cutoffs ≈ CAP Round 3 or slightly lower |
| ⚡ Act fast | Windows are 2–4 days per college |
| 🔄 Hold your CAP seat | Attend ACAP while holding CAP allocation — only surrender if ACAP gives better |

| Event | Timing |
|-------|--------|
| CAP Round 1 | Early August |
| CAP Round 2 | Mid August |
| CAP Round 3 | Early September |
| ACAP | September–October |
| Classes begin | November |

> Verify dates at [fe2025.mahacet.org](https://fe2025.mahacet.org)

| College | City | Strengths |
|---------|------|-----------|
| COEP Technological University | Pune | CS, Mech, E&TC |
| VJTI | Mumbai | CS, IT, Electronics |
| SGGS Institute of Engineering | Nanded | CS, Mech |
| Walchand College of Engineering | Sangli | CS, Mech, Civil |
| KJ Somaiya | Mumbai | CS, IT |
| PICT | Pune | CS, IT |
| Cummins College | Pune | Women's — CS, IT |
""")

    # ═══ TAB 7 — EXPORT ═════════════════════════════════════════════════
    with tab7:
        if pref_list.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>',unsafe_allow_html=True)
        else:
            prev_rows = []
            for idx,row in pref_list.iterrows():
                lbl  = get_label(row['classification'])
                hu_s = int(row.get('hu_seats',0)); oh_s=int(row.get('ohu_seats',0))
                sl_s = int(row.get('sl_seats',0))
                ss   = (f"HU:{hu_s}/OHU:{oh_s}" if hu_s>0 and oh_s>0
                        else f"SL:{sl_s}" if sl_s>0
                        else str(int(row.get('cap_seats',0))) if row.get('has_seat_data') else "—")
                is_sq = row.get('is_special_quota',False)
                prev_rows.append({
                    "#":idx,"College":row['college_name'],"Branch":row['course_name'],
                    "Cat":f"{row.get('best_category','')} [QUOTA]" if is_sq else row.get('best_category',''),
                    "Cutoff":f"{row.get('predicted_cutoff',0):.2f}",
                    "Prob":f"{row.get('probability',0):.0f}%",
                    "Class":f"{BE.get(lbl,'')} {lbl}","Seats":ss,
                    "TFWS":int(row.get('tfws_seats',0)) or "—",
                    "EWS":int(row.get('ews_seats',0)) or "—",
                })
            st.dataframe(pd.DataFrame(prev_rows),use_container_width=True,hide_index=True)
            st.markdown("---")
            ex1,ex2 = st.columns(2)
            with ex1:
                st.markdown("**PDF Export**")
                if st.button("Generate PDF",type="primary",use_container_width=True):
                    with st.spinner("Generating…"):
                        try:
                            pdf = generate_pdf(student_profile, pref_list)
                            st.download_button("⬇️ Download PDF",data=pdf,
                                file_name=f"MHTCET_R{student_profile.get('cap_round',1)}.pdf",
                                mime="application/pdf",use_container_width=True)
                        except Exception as e:
                            st.error(f"PDF failed: {e}")
            with ex2:
                st.markdown("**CSV Export**")
                if not predictions.empty:
                    csv_df = predictions.copy()
                    csv_df['class'] = csv_df['classification'].apply(get_label)
                    csv_df = csv_df.drop(columns=['classification'],errors='ignore')
                    st.download_button("⬇️ Download CSV",data=csv_df.to_csv(index=False),
                        file_name="MHTCET_AllOptions.csv",mime="text/csv",use_container_width=True)
            st.markdown('<div class="ibox">⚠️ Historical data 2022–2024 for guidance only. '
                        'Verify at <a href="https://fe2025.mahacet.org">fe2025.mahacet.org</a>.</div>',
                        unsafe_allow_html=True)
