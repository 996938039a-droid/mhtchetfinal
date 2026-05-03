"""
app.py — MHT-CET College Preference Advisor
Complete rewrite addressing all reported issues.
"""
import os, sys
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
    get_round_data_status,
    generate_all_predictions, generate_preference_list,
    float_freeze_advice, classify,
    analyse_college_branch, get_relevant_categories,
)
from src.export import generate_pdf

st.set_page_config(
    page_title="MHT-CET Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme-adaptive CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

/* Light theme */
[data-theme="light"], .stApp {
    --bg:         #F7F9FC;
    --surface:    #FFFFFF;
    --surface2:   #F0F4F8;
    --border:     #E2E8F0;
    --text:       #1A202C;
    --text2:      #4A5568;
    --text3:      #718096;
    --accent:     #2563EB;
    --accent2:    #1E3A5F;
    --reach:      #EF4444;
    --dream:      #F97316;
    --target:     #2563EB;
    --safe:       #16A34A;
    --assured:    #6B7280;
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
    .stApp {
        --bg:       #0F1117;
        --surface:  #1E2130;
        --surface2: #262B3D;
        --border:   #2D3748;
        --text:     #F0F4F8;
        --text2:    #CBD5E0;
        --text3:    #A0AEC0;
        --accent:   #60A5FA;
        --accent2:  #93C5FD;
    }
}

.stApp { background-color: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--accent2) !important;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* ── Input form card ── */
.form-card {
    background: var(--surface);
    border-radius: 16px;
    padding: 28px 32px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.form-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text3);
    margin: 20px 0 10px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

/* ── Page header ── */
.page-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0;
}
.page-subtitle {
    font-size: 0.82rem;
    color: var(--text3);
    margin: 4px 0 0 0;
}

/* ── Stat chips ── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
.chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--text2);
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

/* ── Summary stat cards ── */
.scard {
    background: var(--surface);
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid var(--border);
    border-top: 3px solid var(--border);
    text-align: center;
}
.scard-val { font-size: 1.8rem; font-weight: 800; color: var(--text); margin: 0; }
.scard-lbl { font-size: 0.72rem; color: var(--text3); margin: 2px 0 0 0; }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: 0.02em;
}

/* ── Preference row card ── */
.prow {
    background: var(--surface);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 12px 16px;
    margin-bottom: 6px;
    display: grid;
    grid-template-columns: 36px 1fr auto auto;
    gap: 12px;
    align-items: center;
}
.prow-num  { font-size: 1.3rem; font-weight: 900; color: var(--accent2); text-align: center; }
.prow-name { font-size: 0.95rem; font-weight: 700; color: var(--text); margin: 0; }
.prow-sub  { font-size: 0.78rem; color: var(--text2); margin: 2px 0 0 0; }
.prow-meta { font-size: 0.72rem; color: var(--text3); margin: 3px 0 0 0; }
.prow-stat { text-align: right; }
.prow-cut  { font-size: 0.78rem; color: var(--text2); }
.prow-prob { font-size: 1.1rem; font-weight: 800; color: var(--text); }

/* ── Advice boxes ── */
.adv { border-radius: 10px; padding: 16px 20px; margin: 10px 0; }
.adv-freeze { background: #ECFDF5; border: 1px solid #86EFAC; }
.adv-float  { background: #EFF6FF; border: 1px solid #93C5FD; }
.adv-slide  { background: #FFF7ED; border: 1px solid #FCA5A5; }
.adv h3 { margin: 0 0 6px 0; font-size: 1rem; color: #1A202C !important; }
.adv p  { margin: 0; font-size: 0.88rem; color: #374151 !important; }

/* ── Info box ── */
.ibox {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #1E40AF !important;
    margin: 8px 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2);
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text2) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent2) !important;
    color: #fff !important;
}

/* ── Divider ── */
.thin-hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Generate button ── */
.stButton > button[kind="primary"] {
    background: var(--accent2) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BADGE_COLORS = {"Reach":"#EF4444","Dream":"#F97316","Target":"#2563EB","Safe":"#16A34A","Assured":"#6B7280"}
BADGE_EMOJI  = {"Reach":"🎯","Dream":"⭐","Target":"✅","Safe":"🛡️","Assured":"🔒"}
TREND_ICON   = {"rising":"📈","falling":"📉","stable":"→"}

def badge(label):
    c = BADGE_COLORS.get(label,"#999")
    return f"<span class='badge' style='background:{c}'>{BADGE_EMOJI.get(label,'')} {label}</span>"

def get_label(cl):
    return cl['label'] if isinstance(cl,dict) else str(cl)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    cutoffs  = load_all_cutoffs()
    sm       = load_seat_matrix()
    cfg      = load_config()
    sl       = build_seat_lookup(sm)
    return cutoffs, sm, cfg, sl

@st.cache_data(show_spinner=False)
def build_umap(_cdf, _cfg):
    if _cdf.empty: return {}
    kw  = _cfg.get("university_college_keywords", {})
    out = {}
    for college in _cdf['college_name'].unique():
        cl = college.lower()
        out[college] = next((u for u, ks in kw.items() if any(k.lower() in cl for k in ks)), "")
    return out

with st.spinner(""):
    cutoff_df, seat_matrix_df, config, seat_lookup = get_data()

university_map  = build_umap(cutoff_df, config)
categories_map  = config.get("categories", {})
district_univ   = config.get("district_university_map", {})
special_q_map   = config.get("special_quotas", {})
all_branches    = get_available_branches(cutoff_df)

# ── Session state init ────────────────────────────────────────────────────────
DEFAULTS = {
    "predictions":    pd.DataFrame(),
    "preference_list": pd.DataFrame(),
    "student_profile": {},
    "results_ready":  False,
    "trend_preset":   0.0,
    # Tab2 filter state (persists across reruns so filters don't jump tabs)
    "f_class":  ["Dream","Target","Safe","Assured"],
    "f_branch": [],
    "f_minp":   0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

predictions     = st.session_state.predictions
pref_list       = st.session_state.preference_list
student_profile = st.session_state.student_profile
results_ready   = st.session_state.results_ready

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — shown after results are ready, with an unhide button
# ═══════════════════════════════════════════════════════════════════════════════
if results_ready:
    with st.sidebar:
        st.markdown("### 🎓 MHT-CET Advisor")
        st.caption("Edit your profile and regenerate")
        st.markdown("---")

        # Mirror inputs in sidebar for re-run
        sb_pct = st.number_input("Percentile", 0.0, 100.0,
                                  float(student_profile.get("percentile", 85.0)),
                                  step=0.01, format="%.2f", key="sb_pct")
        sb_cat = st.selectbox("Category", list(categories_map.keys()),
                               index=list(categories_map.keys()).index(
                                   student_profile.get("raw_category",
                                   list(categories_map.keys())[0])),
                               format_func=lambda k: categories_map[k], key="sb_cat")
        sb_gen = st.radio("Gender", ["male","female"],
                           index=["male","female"].index(student_profile.get("gender","male")),
                           format_func=str.capitalize, horizontal=True, key="sb_gen")
        sb_dist = st.selectbox("District", sorted(district_univ.keys()),
                                index=sorted(district_univ.keys()).index(
                                    student_profile.get("district",
                                    sorted(district_univ.keys())[0])),
                                key="sb_dist")
        sb_hu   = district_univ.get(sb_dist, "")
        sb_sq   = st.multiselect("Special Quota", list(special_q_map.keys()),
                                  default=student_profile.get("special_quotas",[]),
                                  format_func=lambda k: special_q_map[k], key="sb_sq")
        sb_br   = st.multiselect("Preferred Branches",
                                  all_branches,
                                  default=student_profile.get("branches",[]),
                                  key="sb_br")
        sb_bpri = st.radio("Sort By", ["Branch First","College First"],
                            index=0 if student_profile.get("branch_priority", True) else 1,
                            key="sb_bpri") == "Branch First"
        sb_near = st.toggle("📍 Prioritise Nearby", student_profile.get("prioritise_nearby", False), key="sb_near")
        sb_types = st.multiselect("College Types",
            ["Government","Government Autonomous","Government-Aided",
             "Government-Aided Autonomous","Un-Aided","Un-Aided Autonomous",
             "University Department","University Managed"],
            default=student_profile.get("college_types",
                ["Government","Government Autonomous","Government-Aided","Government-Aided Autonomous"]),
            key="sb_types")
        sb_rnd  = st.selectbox("CAP Round", [1,2,3],
                                index=student_profile.get("cap_round",1)-1,
                                format_func=lambda x: f"Round {x}", key="sb_rnd")

        # Trend adjustment
        st.markdown("**Cutoff Trend Adjustment**")
        pc1, pc2, pc3 = st.columns(3)
        if pc1.button("📉−2", use_container_width=True, key="sb_e"):
            st.session_state.trend_preset = -2.0
        if pc2.button("→ 0", use_container_width=True, key="sb_n"):
            st.session_state.trend_preset = 0.0
        if pc3.button("📈+2", use_container_width=True, key="sb_h"):
            st.session_state.trend_preset = 2.0
        sb_tadj = st.slider("Fine-tune", -5.0, 5.0,
                             float(st.session_state.trend_preset), 0.5,
                             format="%.1f", key="sb_tadj")
        sb_maxp = st.slider("Max Preferences", 5, 20,
                             student_profile.get("max_pref", 10), key="sb_maxp")

        st.markdown("---")
        rerun_btn = st.button("🔄 Update Results", type="primary", use_container_width=True)

        # Data status
        rds = get_round_data_status(cutoff_df)
        st.markdown("---")
        st.markdown("**Data Coverage**")
        for rn in [1,2,3]:
            st.caption(f"{'✅' if rds.get(rn) else '⚠️'} Round {rn}")
        yrs = rds.get('years',[])
        if yrs:
            st.caption(f"Years: {', '.join(str(y) for y in yrs)}")

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='page-title'>🎓 MHT-CET College Preference Advisor</p>
<p class='page-subtitle'>Maharashtra Engineering Admissions · CAP Round Analysis · 2022–2024 Data</p>
""", unsafe_allow_html=True)

if not cutoff_df.empty:
    rds  = get_round_data_status(cutoff_df)
    yrs  = rds.get('years',[])
    st.markdown(
        f"<div class='chip-row'>"
        f"<span class='chip'>📅 {', '.join(str(y) for y in yrs)}</span>"
        f"<span class='chip'>🔄 Rounds {', '.join(str(r) for r in [1,2,3] if rds.get(r))}</span>"
        f"<span class='chip'>🏛️ {cutoff_df['college_name'].nunique()} colleges</span>"
        f"<span class='chip'>📚 {cutoff_df['course_name'].nunique()} branches</span>"
        f"</div>",
        unsafe_allow_html=True
    )
else:
    st.error("No data found. Place cutoff Excel files in `data/cutoffs/`.")
    st.stop()

st.markdown("<div class='thin-hr'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FORM — shown as main page until results generated
# ═══════════════════════════════════════════════════════════════════════════════
if not results_ready:
    st.markdown("### Enter Your Details")

    with st.container():
        # Row 1: Profile
        st.markdown("<div class='form-section-title'>Your Profile</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2,2,2,2])
        with c1:
            percentile = st.number_input("MHT-CET Percentile", 0.0, 100.0, 85.0,
                                          step=0.01, format="%.2f")
        with c2:
            category = st.selectbox("Category", list(categories_map.keys()),
                                     format_func=lambda k: categories_map[k])
        with c3:
            gender = st.radio("Gender", ["male","female"],
                               format_func=str.capitalize, horizontal=True)
        with c4:
            district = st.selectbox("Home District", sorted(district_univ.keys()))
            home_university = district_univ.get(district, "")
            if home_university:
                st.caption(f"🏛️ {home_university}")

        c5, c6 = st.columns([3,3])
        with c5:
            special_quotas = st.multiselect("Special Quota (if eligible)",
                                             list(special_q_map.keys()),
                                             format_func=lambda k: special_q_map[k])
        with c6:
            preferred_branches = st.multiselect("Preferred Branches (priority order)",
                                                  all_branches)

        # Row 2: Preferences
        st.markdown("<div class='form-section-title'>Filters & Settings</div>",
                    unsafe_allow_html=True)
        d1, d2, d3 = st.columns([2,2,2])
        with d1:
            branch_priority = st.radio("Sort Results By",
                                        ["Branch First","College First"]) == "Branch First"
            prioritise_nearby = st.toggle("📍 Prioritise Colleges Near Me", False)
        with d2:
            college_types = st.multiselect("College Types",
                ["Government","Government Autonomous","Government-Aided",
                 "Government-Aided Autonomous","Un-Aided","Un-Aided Autonomous",
                 "University Department","University Managed"],
                default=["Government","Government Autonomous",
                         "Government-Aided","Government-Aided Autonomous"])
        with d3:
            cap_round = st.selectbox("Target CAP Round", [1,2,3],
                                      format_func=lambda x: f"CAP Round {x}")
            max_pref  = st.slider("Max Preferences", 5, 20, 10)

        # Trend adjustment
        st.markdown("<div class='form-section-title'>Cutoff Trend Adjustment for 2025</div>",
                    unsafe_allow_html=True)
        st.caption("Adjust if you expect this year's cutoffs to be higher or lower than historical average. "
                   "The adjustment scales automatically — high percentile ranges compress more than low ones.")
        ta1, ta2, ta3, ta4 = st.columns([1,1,1,3])
        with ta1:
            if st.button("📉 Easier (−2)"):
                st.session_state.trend_preset = -2.0
        with ta2:
            if st.button("→ Same (0)"):
                st.session_state.trend_preset = 0.0
        with ta3:
            if st.button("📈 Harder (+2)"):
                st.session_state.trend_preset = 2.0
        with ta4:
            trend_adj = st.slider("Fine-tune", -5.0, 5.0,
                                   float(st.session_state.trend_preset), 0.5,
                                   format="%.1f",
                                   label_visibility="collapsed")
        if trend_adj > 0:
            st.caption(f"⬆️ Cutoffs will be scaled up — effect is larger at lower percentile ranges, smaller at 95+")
        elif trend_adj < 0:
            st.caption(f"⬇️ Cutoffs will be scaled down — same non-linear scaling applies")

        st.markdown("")
        run_btn = st.button("🔍 Generate Recommendations", type="primary",
                             use_container_width=True)

    # Handle run
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
                preds['nearby_boost'] = preds['is_home_university'].apply(lambda x: 0 if x else 1)
                sc = (['branch_rank','nearby_boost','classification_order','probability']
                      if branch_priority else
                      ['nearby_boost','classification_order','branch_rank','probability'])
                preds = preds.sort_values(sc, ascending=[True,True,True,False]).reset_index(drop=True)
            pl = generate_preference_list(preds, max_list=max_pref)

        st.session_state.predictions     = preds
        st.session_state.preference_list = pl
        st.session_state.results_ready   = True
        st.session_state.student_profile = {
            "percentile": percentile,
            "raw_category": category,
            "category": f"{category} — {categories_map.get(category,'')}",
            "gender": gender, "district": district,
            "home_university": home_university,
            "special_quotas": special_quotas,
            "branches": preferred_branches,
            "branch_priority": branch_priority,
            "prioritise_nearby": prioritise_nearby,
            "college_types": college_types,
            "cap_round": cap_round,
            "trend_adj": trend_adj,
            "max_pref": max_pref,
        }
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS VIEW — shown after generate is clicked
# ═══════════════════════════════════════════════════════════════════════════════
else:
    # Handle sidebar re-run
    if "rerun_btn" in dir() and rerun_btn:
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
                preds['nearby_boost'] = preds['is_home_university'].apply(lambda x: 0 if x else 1)
                sc = (['branch_rank','nearby_boost','classification_order','probability']
                      if sb_bpri else
                      ['nearby_boost','classification_order','branch_rank','probability'])
                preds = preds.sort_values(sc, ascending=[True,True,True,False]).reset_index(drop=True)
            pl = generate_preference_list(preds, max_list=sb_maxp)
        st.session_state.predictions     = preds
        st.session_state.preference_list = pl
        st.session_state.student_profile.update({
            "percentile": sb_pct, "raw_category": sb_cat,
            "category": f"{sb_cat} — {categories_map.get(sb_cat,'')}",
            "gender": sb_gen, "district": sb_dist, "home_university": sb_hu,
            "special_quotas": sb_sq, "branches": sb_br,
            "branch_priority": sb_bpri, "prioritise_nearby": sb_near,
            "college_types": sb_types, "cap_round": sb_rnd,
            "trend_adj": sb_tadj, "max_pref": sb_maxp,
        })
        st.rerun()

    predictions     = st.session_state.predictions
    pref_list       = st.session_state.preference_list
    student_profile = st.session_state.student_profile

    # Profile summary row
    sp = student_profile
    col_prof, col_edit = st.columns([8,1])
    with col_prof:
        st.markdown(
            f"<div class='chip-row'>"
            f"<span class='chip'>🎯 {sp.get('percentile','—')} pct</span>"
            f"<span class='chip'>{sp.get('category','—')}</span>"
            f"<span class='chip'>{str(sp.get('gender','')).capitalize()}</span>"
            f"<span class='chip'>📍 {sp.get('district','—')}</span>"
            f"<span class='chip'>Round {sp.get('cap_round','—')}</span>"
            f"<span class='chip'>Trend {sp.get('trend_adj',0):+.1f} pts</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_edit:
        if st.button("✏️ Edit", help="Edit your profile in the sidebar"):
            pass  # sidebar auto-shows when results_ready

    # Summary stat cards
    if not predictions.empty:
        labels_s = predictions['classification'].apply(get_label)
        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        for col, key, color in [
            (sc1,"Reach","#EF4444"),(sc2,"Dream","#F97316"),
            (sc3,"Target","#2563EB"),(sc4,"Safe","#16A34A"),(sc5,"Assured","#6B7280")
        ]:
            col.markdown(
                f"<div class='scard' style='border-top-color:{color}'>"
                f"<div class='scard-val' style='color:{color}'>{(labels_s==key).sum()}</div>"
                f"<div class='scard-lbl'>{BADGE_EMOJI.get(key,'')} {key}</div></div>",
                unsafe_allow_html=True
            )
        st.markdown("")

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📋 Preference List","🔍 All Options",
        "📊 Round Analysis","⚖️ Float / Freeze",
        "🏛️ ACAP Guide","📤 Export",
    ])

    # ═══ TAB 1 — PREFERENCE LIST ════════════════════════════════════════════
    with tab1:
        if pref_list.empty:
            st.markdown('<div class="ibox">No matching options found. Try adjusting your filters.</div>',
                        unsafe_allow_html=True)
        else:
            # Legend
            lcols = st.columns(5)
            for lc, (lbl, color) in zip(lcols, BADGE_COLORS.items()):
                lc.markdown(
                    f"<div style='text-align:center;padding:4px 0'>"
                    f"<span class='badge' style='background:{color}'>"
                    f"{BADGE_EMOJI[lbl]} {lbl}</span></div>",
                    unsafe_allow_html=True
                )
            st.markdown("<div class='thin-hr'></div>", unsafe_allow_html=True)

            for idx, row in pref_list.iterrows():
                lbl    = get_label(row['classification'])
                color  = BADGE_COLORS.get(lbl,"#999")
                prob   = row.get('probability', 0)
                cutoff = row.get('predicted_cutoff', 0)
                gap    = row.get('gap', 0)
                trend  = TREND_ICON.get(row.get('trend','stable'), "")
                is_hu  = row.get('is_home_university', False)
                gap_color = "#16A34A" if gap >= 0 else "#DC2626"
                sign   = "+" if gap >= 0 else ""

                # Seat info
                cap_s = int(row.get('cap_seats',0))
                hu_s  = int(row.get('hu_seats',0))
                oh_s  = int(row.get('ohu_seats',0))
                sl_s  = int(row.get('sl_seats',0))
                if row.get('has_seat_data') and cap_s > 0:
                    if hu_s > 0 and oh_s > 0:
                        seat_s = f"HU: {hu_s} &nbsp;·&nbsp; OHU: {oh_s}"
                    elif sl_s > 0:
                        seat_s = f"SL: {sl_s} seats"
                    else:
                        seat_s = f"{cap_s} seats"
                    tfws = int(row.get('tfws_seats',0))
                    ews  = int(row.get('ews_seats',0))
                    if tfws: seat_s += f" &nbsp;·&nbsp; TFWS: {tfws}"
                    if ews:  seat_s += f" &nbsp;·&nbsp; EWS: {ews}"
                else:
                    seat_s = "—"

                hu_chip = (" <span style='background:#DBEAFE;color:#1E40AF;border-radius:4px;"
                           "padding:1px 6px;font-size:0.65rem;font-weight:700'>🏠 HU</span>") if is_hu else ""

                cn, ci, cs, cb = st.columns([0.5, 5, 2.8, 1.5])
                with cn:
                    st.markdown(f"<div style='text-align:center;padding-top:8px'>"
                                f"<span class='prow-num'>{idx}</span></div>",
                                unsafe_allow_html=True)
                with ci:
                    st.markdown(
                        f"<p class='prow-name'>{row['college_name']}{hu_chip}</p>"
                        f"<p class='prow-sub'>📚 {row['course_name']}</p>"
                        f"<p class='prow-meta'>🏷️ {row.get('best_category','')} &nbsp;·&nbsp; "
                        f"{row.get('status','')} &nbsp;·&nbsp; {seat_s}</p>",
                        unsafe_allow_html=True
                    )
                with cs:
                    st.markdown(
                        f"<div style='padding-top:6px;text-align:right'>"
                        f"<p class='prow-cut'>Cutoff: <strong>{cutoff:.2f}</strong> {trend}</p>"
                        f"<p class='prow-cut'>Gap: <strong style='color:{gap_color}'>{sign}{gap:.2f}</strong>"
                        f" &nbsp; Prob: <strong>{prob:.0f}%</strong></p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with cb:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:10px'>"
                        f"<span class='badge' style='background:{color};padding:5px 12px'>"
                        f"{BADGE_EMOJI.get(lbl,'')} {lbl}</span></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("<div class='thin-hr'></div>", unsafe_allow_html=True)

            # Probability chart
            st.markdown("**Probability Overview**")
            bx = list(range(1, len(pref_list)+1))
            by = [r.get('probability',0) for _,r in pref_list.iterrows()]
            bc = [BADGE_COLORS.get(get_label(r['classification']),"#999")
                  for _,r in pref_list.iterrows()]
            bnames = [f"#{i} {r['college_name'][:22]}" for i,r in pref_list.iterrows()]
            fig = go.Figure(go.Bar(
                x=bx, y=by, marker_color=bc,
                text=[f"{p:.0f}%" for p in by], textposition='outside',
                textfont=dict(size=10),
                customdata=bnames,
                hovertemplate="<b>%{customdata}</b><br>%{y:.0f}%<extra></extra>",
            ))
            fig.add_hline(y=70, line_dash="dash", line_color="#16A34A", line_width=1.5,
                          annotation_text="Safe (70%)", annotation_font_size=10)
            fig.add_hline(y=30, line_dash="dash", line_color="#F97316", line_width=1.5,
                          annotation_text="Dream (30%)", annotation_font_size=10)
            fig.update_layout(
                height=260, margin=dict(t=20,b=20,l=30,r=20),
                xaxis=dict(title="Pref #", tickfont=dict(size=10)),
                yaxis=dict(range=[0,118], title="Probability (%)"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="ibox">💡 Dream/Reach at top — cutoffs drop in later rounds so you may get them. '
                        'Target forms the bulk. Safe/Assured protect you at the bottom.</div>',
                        unsafe_allow_html=True)

    # ═══ TAB 2 — ALL OPTIONS ════════════════════════════════════════════════
    with tab2:
        if predictions.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>', unsafe_allow_html=True)
        else:
            # Filters stored in session_state so they don't cause tab jumps
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_class = st.multiselect(
                    "Classification",
                    ["Reach","Dream","Target","Safe","Assured"],
                    default=st.session_state.f_class,
                    key="t2_class",
                )
                st.session_state.f_class = f_class
            with fc2:
                f_branch = st.multiselect(
                    "Branch",
                    sorted(predictions['course_name'].unique()),
                    default=[b for b in st.session_state.f_branch
                             if b in predictions['course_name'].unique()],
                    key="t2_branch",
                )
                st.session_state.f_branch = f_branch
            with fc3:
                f_minp = st.slider("Min Probability (%)", 0, 100,
                                    st.session_state.f_minp, key="t2_minp")
                st.session_state.f_minp = f_minp

            disp = predictions.copy()
            ls   = disp['classification'].apply(get_label)
            if f_class:  disp = disp[ls.isin(f_class)]
            if f_branch: disp = disp[disp['course_name'].isin(f_branch)]
            disp = disp[disp['probability'] >= f_minp]

            st.caption(f"Showing **{len(disp)}** of {len(predictions)} options")

            rows = []
            for _, r in disp.iterrows():
                lbl  = get_label(r['classification'])
                hu_s = int(r.get('hu_seats',0))
                oh_s = int(r.get('ohu_seats',0))
                sl_s = int(r.get('sl_seats',0))
                seat_str = (f"HU:{hu_s}/OHU:{oh_s}" if (hu_s>0 and oh_s>0)
                            else f"SL:{sl_s}" if sl_s>0
                            else f"{int(r.get('cap_seats',0))}" if r.get('has_seat_data') else "—")
                rows.append({
                    "College": r['college_name'],
                    "Branch":  r['course_name'],
                    "Type":    r.get('status',''),
                    "Cat":     r.get('best_category',''),
                    "Cutoff":  round(r.get('predicted_cutoff',0),2),
                    "Prob":    f"{r.get('probability',0):.0f}%",
                    "Class":   f"{BADGE_EMOJI.get(lbl,'')} {lbl}",
                    "Seats":   seat_str,
                    "TFWS":    int(r.get('tfws_seats',0)) or "—",
                    "Trend":   TREND_ICON.get(r.get('trend',''),""),
                    "HU":      "✅" if r.get('is_home_university') else "",
                    "Yrs":     r.get('data_years',0),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=460)

            # Distribution chart
            if not disp.empty:
                fig_h = px.histogram(disp, x='probability', nbins=20,
                                      color_discrete_sequence=["#2563EB"])
                fig_h.add_vline(x=70, line_dash="dash", line_color="#16A34A",
                                annotation_text="Safe", annotation_font_size=10)
                fig_h.add_vline(x=30, line_dash="dash", line_color="#F97316",
                                annotation_text="Dream", annotation_font_size=10)
                fig_h.update_layout(
                    height=220, margin=dict(t=10,b=20,l=10,r=10),
                    xaxis_title="Probability (%)", yaxis_title="Count",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_h, use_container_width=True)

    # ═══ TAB 3 — ROUND ANALYSIS ═════════════════════════════════════════════
    with tab3:
        if predictions.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>', unsafe_allow_html=True)
        else:
            r1c, r2c = st.columns(2)
            with r1c:
                ra_col = st.selectbox("College",
                                       sorted(predictions['college_name'].unique()), key="ra_col")
            with r2c:
                ra_br  = st.selectbox("Branch",
                    sorted(predictions[predictions['college_name']==ra_col]['course_name'].unique()),
                    key="ra_br")

            sp = student_profile
            round_probs = []
            for rnum in [1,2,3]:
                cu   = university_map.get(ra_col,"")
                elig = get_relevant_categories(
                    sp.get("raw_category","OPEN"), sp.get("gender","male"),
                    sp.get("home_university",""), cu,
                    sp.get("special_quotas",[])
                )
                si_key = (
                    __import__('re').sub(r"[,.'\"()]",'',ra_col.lower().strip()),
                    __import__('re').sub(r"[,.'\"()]",'',ra_br.lower().strip()),
                )
                si  = seat_lookup.get(si_key, {})
                res = analyse_college_branch(
                    cutoff_df, ra_col, ra_br, elig,
                    sp.get("percentile",85.0), rnum,
                    sp.get("trend_adj",0.0), si
                )
                if res:
                    round_probs.append({
                        "Round": f"Round {rnum}", "r_num": rnum,
                        "Probability": res['probability'],
                        "Predicted Cutoff": res['predicted_cutoff'],
                        "Classification": res['classification']['label'],
                    })

            if round_probs:
                rdf = pd.DataFrame(round_probs).sort_values('r_num')
                r_colors = [BADGE_COLORS.get(r['Classification'],"#2563EB") for r in round_probs]

                fig_r = go.Figure(go.Bar(
                    x=rdf['Round'], y=rdf['Probability'],
                    marker_color=r_colors, width=0.4,
                    text=[f"{p:.0f}%" for p in rdf['Probability']],
                    textposition='outside', textfont=dict(size=13),
                ))
                fig_r.add_hline(y=70, line_dash="dash", line_color="#16A34A", line_width=1.5,
                                annotation_text="Safe (70%)", annotation_font_size=10)
                fig_r.add_hline(y=30, line_dash="dash", line_color="#F97316", line_width=1.5,
                                annotation_text="Dream (30%)", annotation_font_size=10)
                fig_r.update_layout(
                    title=f"{ra_br} · {ra_col[:50]}",
                    yaxis=dict(range=[0,118], title="Probability (%)"),
                    height=340, margin=dict(t=50,b=30,l=40,r=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_r, use_container_width=True)

                # Cutoff line
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=rdf['Round'], y=rdf['Predicted Cutoff'],
                    mode='lines+markers+text',
                    line=dict(color="#2563EB", width=2.5),
                    marker=dict(size=9, color="#1E3A5F"),
                    text=[f"{c:.2f}" for c in rdf['Predicted Cutoff']],
                    textposition="top center", textfont=dict(size=10),
                ))
                fig_c.add_hline(
                    y=sp.get("percentile",85.0),
                    line_dash="dot", line_color="#16A34A", line_width=1.5,
                    annotation_text=f"Your pct: {sp.get('percentile',85.0):.2f}",
                    annotation_font_size=10,
                )
                fig_c.update_layout(
                    title="Predicted Cutoff per Round",
                    yaxis=dict(title="Closing Percentile"),
                    height=240, margin=dict(t=40,b=20,l=40,r=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_c, use_container_width=True)

                show = rdf[['Round','Probability','Predicted Cutoff','Classification']].copy()
                show['Probability'] = show['Probability'].apply(lambda x: f"{x:.1f}%")
                show['Predicted Cutoff'] = show['Predicted Cutoff'].apply(lambda x: f"{x:.4f}")
                st.dataframe(show.drop(columns=['r_num'], errors='ignore'),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("Not enough data for this combination.")

            # Heatmap
            st.markdown("---")
            st.markdown("**Probability Heatmap — Top Options Across Rounds**")
            st.caption("Deeper green = higher probability")

            hdata, seen = [], set()
            for _, row in predictions.head(12).iterrows():
                cu   = university_map.get(row['college_name'],"")
                elig = get_relevant_categories(
                    sp.get("raw_category","OPEN"), sp.get("gender","male"),
                    sp.get("home_university",""), cu, sp.get("special_quotas",[])
                )
                okey = f"{row['college_name']} ||| {row['course_name']}"
                for rnum in [1,2,3]:
                    dk = (okey, rnum)
                    if dk in seen: continue
                    seen.add(dk)
                    import re as _re
                    si_k = (
                        _re.sub(r"[,.'\"()]",'',row['college_name'].lower().strip()),
                        _re.sub(r"[,.'\"()]",'',row['course_name'].lower().strip()),
                    )
                    si  = seat_lookup.get(si_k,{})
                    res = analyse_college_branch(
                        cutoff_df, row['college_name'], row['course_name'],
                        elig, sp.get("percentile",85.0), rnum,
                        sp.get("trend_adj",0.0), si
                    )
                    if res:
                        hdata.append({"Option":okey,"Round":f"R{rnum}","Probability":res['probability']})

            if hdata:
                hdf    = pd.DataFrame(hdata)
                hdf    = hdf.groupby(['Option','Round'], as_index=False)['Probability'].mean()
                # Only pivot if we have data
                if not hdf.empty and hdf['Option'].nunique() > 0:
                    try:
                        hpivot = hdf.pivot(index='Option', columns='Round', values='Probability')
                        short  = []
                        for opt in hpivot.index:
                            p = opt.split(" ||| ")
                            cs = (p[0][:28]+"…") if len(p[0])>28 else p[0]
                            bs = (p[1][:20]+"…") if len(p)>1 and len(p[1])>20 else (p[1] if len(p)>1 else "")
                            short.append(f"{cs} / {bs}")
                        hpivot.index = short
                        fig_hm = px.imshow(
                            hpivot, color_continuous_scale="RdYlGn",
                            zmin=0, zmax=100, aspect="auto", text_auto=".0f",
                        )
                        fig_hm.update_traces(textfont=dict(size=11))
                        n_rows = len(hpivot)
                        fig_hm.update_layout(
                            height=max(300, n_rows * 40),
                            margin=dict(t=10,b=10,l=10,r=60),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)
                    except Exception:
                        st.info("Not enough data to render heatmap.")

    # ═══ TAB 4 — FLOAT / FREEZE ═════════════════════════════════════════════
    with tab4:
        st.markdown("Enter your current CAP allocation to get a Float / Freeze / Slide recommendation.")
        if predictions.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>', unsafe_allow_html=True)
        else:
            ff1, ff2 = st.columns(2)
            with ff1:
                ff_col = st.selectbox("Allocated College",
                                       sorted(predictions['college_name'].unique()), key="ff_col")
            with ff2:
                ff_brs = predictions[predictions['college_name']==ff_col]['course_name'].unique()
                ff_br  = st.selectbox("Allocated Branch", sorted(ff_brs), key="ff_br")
            ff_rnd = st.selectbox("Round of Allocation", [1,2],
                                   format_func=lambda x: f"Round {x}", key="ff_rnd")

            sp     = student_profile
            cu     = university_map.get(ff_col,"")
            elig   = get_relevant_categories(
                sp.get("raw_category","OPEN"), sp.get("gender","male"),
                sp.get("home_university",""), cu, sp.get("special_quotas",[])
            )
            import re as _re2
            si_k = (
                _re2.sub(r"[,.'\"()]",'',ff_col.lower().strip()),
                _re2.sub(r"[,.'\"()]",'',ff_br.lower().strip()),
            )
            si  = seat_lookup.get(si_k,{})
            cur = analyse_college_branch(
                cutoff_df, ff_col, ff_br, elig,
                sp.get("percentile",85.0), ff_rnd,
                sp.get("trend_adj",0.0), si
            )
            if cur:
                cur_lbl = get_label(cur['classification'])
                col_c = BADGE_COLORS.get(cur_lbl,"#999")
                st.markdown(
                    f"<div style='background:var(--surface);border:1px solid var(--border);"
                    f"border-left:4px solid {col_c};border-radius:10px;"
                    f"padding:14px 18px;margin:10px 0'>"
                    f"<p style='font-size:0.72rem;color:var(--text3);margin:0'>CURRENT ALLOCATION</p>"
                    f"<p style='font-weight:800;font-size:1rem;color:var(--text);margin:4px 0'>{ff_col}</p>"
                    f"<p style='font-size:0.88rem;color:var(--text2);margin:0'>{ff_br}</p>"
                    f"<div style='margin-top:8px;display:flex;gap:12px;align-items:center'>"
                    f"<span class='badge' style='background:{col_c}'>{cur['classification'].get('emoji','')} {cur_lbl}</span>"
                    f"<span style='font-weight:700'>{cur['probability']:.0f}% probability</span>"
                    f"<span style='color:var(--text3);font-size:0.82rem'>Cutoff: {cur['predicted_cutoff']:.2f}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )
                adv = float_freeze_advice(ff_col, ff_br, cur['probability'],
                                           predictions, ff_rnd+1)
                a   = adv['advice']
                cls = {"FREEZE":"adv-freeze","FLOAT":"adv-float","SLIDE":"adv-slide"}[a]
                ico = {"FREEZE":"🔒","FLOAT":"🌊","SLIDE":"↔️"}[a]
                st.markdown(
                    f"<div class='adv {cls}'>"
                    f"<h3>{ico} Recommendation: {a}</h3>"
                    f"<p>{adv['reason']}</p></div>",
                    unsafe_allow_html=True
                )
                if a == "FLOAT" and adv.get('top_options'):
                    st.markdown("**Better options that may open next round:**")
                    for opt in adv['top_options']:
                        lbl2 = classify(opt['probability'])['label']
                        st.markdown(f"- **{opt['college_name']}** / {opt['course_name']} — "
                                    f"{opt['probability']:.0f}% {badge(lbl2)}",
                                    unsafe_allow_html=True)
                if a == "SLIDE" and adv.get('slide_options'):
                    st.markdown("**Better branches at same college:**")
                    for opt in adv['slide_options']:
                        st.markdown(f"- **{opt['course_name']}** — {opt['probability']:.0f}%")
            else:
                st.warning("No data for this combination.")

            st.markdown("---")
            st.markdown("""
| Decision | Meaning | Best when |
|----------|---------|-----------|
| 🔒 **Freeze** | Accept current seat, exit process | Seat is strong or you can't risk losing it |
| 🌊 **Float** | Hold current seat, try for better | Current seat okay, better options likely |
| ↔️ **Slide** | Same college, different branch | Happy with college, want better branch |
""")

    # ═══ TAB 5 — ACAP ═══════════════════════════════════════════════════════
    with tab5:
        st.markdown("### ACAP — Autonomous College Admission Process")
        st.markdown("""
After CAP Round 3, colleges with **Autonomous** status conduct their own counselling for remaining seats.

**Who can participate:** Students not satisfied with CAP Round 3, or those who got no seat.

**How it works:**
1. After CAP Round 3, autonomous colleges list their vacant seats
2. Each college announces its own schedule on their website (October–November)
3. You attend direct counselling with original documents
4. Merit = your MHT-CET percentile — no new exam
5. Same Maharashtra reservation rules apply

**Strategy tips:**

| Tip | Detail |
|-----|--------|
| 📅 Monitor college websites | No central schedule — each college posts independently |
| 🎯 Realistic expectations | ACAP cutoffs ≈ CAP Round 3 or slightly lower |
| ⚡ Act fast | Windows are 2–4 days per college |
| 🔄 Hold your CAP seat | Attend ACAP while holding CAP seat — only surrender if ACAP gives better |

**Typical calendar:**

| Event | Timing |
|-------|--------|
| CAP Round 1 | Early August |
| CAP Round 2 | Mid August |
| CAP Round 3 | Early September |
| ACAP | September–October |
| Classes begin | November |

> Verify at [fe2025.mahacet.org](https://fe2025.mahacet.org)

**Top autonomous colleges:**

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

    # ═══ TAB 6 — EXPORT ═════════════════════════════════════════════════════
    with tab6:
        if pref_list.empty:
            st.markdown('<div class="ibox">Generate recommendations first.</div>', unsafe_allow_html=True)
        else:
            prev_rows = []
            for idx, row in pref_list.iterrows():
                lbl  = get_label(row['classification'])
                hu_s = int(row.get('hu_seats',0)); oh_s = int(row.get('ohu_seats',0))
                sl_s = int(row.get('sl_seats',0))
                ss   = (f"HU:{hu_s}/OHU:{oh_s}" if hu_s>0 and oh_s>0
                        else f"SL:{sl_s}" if sl_s>0
                        else f"{int(row.get('cap_seats',0))}" if row.get('has_seat_data') else "—")
                prev_rows.append({
                    "#": idx, "College": row['college_name'], "Branch": row['course_name'],
                    "Cat": row.get('best_category',''),
                    "Cutoff": f"{row.get('predicted_cutoff',0):.2f}",
                    "Prob": f"{row.get('probability',0):.0f}%",
                    "Class": f"{BADGE_EMOJI.get(lbl,'')} {lbl}",
                    "Seats": ss,
                    "TFWS": int(row.get('tfws_seats',0)) or "—",
                    "EWS":  int(row.get('ews_seats',0))  or "—",
                })
            st.dataframe(pd.DataFrame(prev_rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            ex1, ex2 = st.columns(2)
            with ex1:
                st.markdown("**PDF Export**")
                if st.button("Generate PDF", type="primary", use_container_width=True):
                    with st.spinner("Generating…"):
                        try:
                            pdf = generate_pdf(student_profile, pref_list)
                            st.download_button("⬇️ Download PDF", data=pdf,
                                file_name=f"MHTCET_R{student_profile.get('cap_round',1)}.pdf",
                                mime="application/pdf", use_container_width=True)
                        except Exception as e:
                            st.error(f"PDF failed: {e}")
            with ex2:
                st.markdown("**CSV Export**")
                if not predictions.empty:
                    csv_df = predictions.copy()
                    csv_df['class'] = csv_df['classification'].apply(get_label)
                    csv_df = csv_df.drop(columns=['classification'], errors='ignore')
                    st.download_button("⬇️ Download CSV", data=csv_df.to_csv(index=False),
                        file_name="MHTCET_AllOptions.csv", mime="text/csv",
                        use_container_width=True)

            st.markdown('<div class="ibox">⚠️ Historical data 2022–2024 for guidance only. '
                        'Verify at <a href="https://fe2025.mahacet.org">fe2025.mahacet.org</a> '
                        'before finalising.</div>', unsafe_allow_html=True)
