"""
data_loader.py — Loads and normalises cutoff + seat matrix Excel files.

Cutoff files: wide format, one row per college-branch.
  Category columns: "GOPENS Merit" / "GOPENS Percentile" etc.
  Melted into long tidy table: college | branch | category | merit | percentile | year | round

Seat matrix: one row per college-branch with all seat breakdown columns.
  Exposed as a lookup dict keyed on normalised (college_name, course_name)
  so probability_engine can query it in O(1).
"""

import os
import re
import glob
import math
import pandas as pd
import streamlit as st
import yaml

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


@st.cache_data(show_spinner=False)
def load_config():
    with open(os.path.join(CONFIG_DIR, "categories.yaml")) as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_all_cutoffs() -> pd.DataFrame:
    pattern = os.path.join(DATA_DIR, "cutoffs", "*.xlsx")
    files   = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    frames = []
    for fpath in files:
        fname = os.path.basename(fpath)
        year, cap_round = _parse_filename(fname)
        if year is None:
            continue
        try:
            df = pd.read_excel(fpath, sheet_name=0)
            df = _normalise_cutoff(df, year, cap_round)
            frames.append(df)
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_seat_matrix() -> pd.DataFrame:
    """Load latest seat matrix; returns cleaned DataFrame."""
    pattern = os.path.join(DATA_DIR, "seat_matrix", "*.xlsx")
    files   = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_excel(files[0], sheet_name=0)
        return _normalise_seat_matrix(df)
    except Exception as e:
        st.warning(f"Could not load seat matrix: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_seat_lookup(_seat_matrix_df: pd.DataFrame) -> dict:
    """
    Build a fast O(1) lookup dict from the seat matrix.

    Key:   (normalised_college_name, normalised_course_name)
    Value: dict with all seat breakdown fields needed by probability_engine:

        cap_seats    — total CAP seats (used for probability weighting)
        sl_total     — State Level seats
        hu_total     — Home University seats (NaN if not applicable)
        ohu_total    — Other Than HU seats (NaN if not applicable)
        tfws_seats   — TFWS quota seats (0 = none available)
        ews_seats    — EWS quota seats
        pwd_total    — PWD quota seats
        def_total    — Defence quota seats
        orphan_seats — Orphan quota seats
        has_hu       — bool: True if HU seats exist for this branch
        has_ohu      — bool: True if OHU seats exist for this branch

        # Per-category seat counts (for quota eligibility filtering)
        sl_open_g, sl_open_l, sl_sc_g, sl_sc_l,
        sl_st_g, sl_st_l, sl_vj_g, sl_vj_l,
        sl_nt1_g, sl_nt1_l, sl_nt2_g, sl_nt2_l,
        sl_nt3_g, sl_nt3_l, sl_obc_g, sl_obc_l,
        sl_sebc_g, sl_sebc_l,
        hu_open_g, hu_open_l, hu_sc_g, hu_sc_l,
        hu_obc_g, hu_obc_l,
        ohu_open_g, ohu_open_l, ohu_sc_g, ohu_sc_l,
        ohu_obc_g, ohu_obc_l
    """
    if _seat_matrix_df.empty:
        return {}

    lookup = {}
    for _, row in _seat_matrix_df.iterrows():
        college = row.get('college_name', '')
        course  = row.get('course_name', '')
        if not college or not course:
            continue

        key = (_norm_str(college), _norm_str(course))

        def g(col, default=0):
            v = row.get(col, default)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return default
            try:
                return float(v)
            except Exception:
                return default

        hu_total  = g('hu_total',  0)
        ohu_total = g('ohu_total', 0)

        lookup[key] = {
            # Totals
            'cap_seats':    g('cap_seats', 60),
            'sl_total':     g('sl_total',  0),
            'hu_total':     hu_total,
            'ohu_total':    ohu_total,
            'has_hu':       hu_total  > 0,
            'has_ohu':      ohu_total > 0,
            # Special quotas
            'tfws_seats':   g('tfws_seats',   0),
            'ews_seats':    g('ews_seats',     0),
            'pwd_total':    g('pwd_total',     0),
            'def_total':    g('def_total',     0),
            'orphan_seats': g('orphan_seats',  0),
            # SL per-category
            'sl_open_g':  g('SL_OPEN_G'),  'sl_open_l':  g('SL_OPEN_L'),
            'sl_sc_g':    g('SL_SC_G'),    'sl_sc_l':    g('SL_SC_L'),
            'sl_st_g':    g('SL_ST_G'),    'sl_st_l':    g('SL_ST_L'),
            'sl_vj_g':    g('SL_VJ/DT_G'), 'sl_vj_l':   g('SL_VJ/DT_L'),
            'sl_nt1_g':   g('SL_NTB_G'),   'sl_nt1_l':  g('SL_NTB_L'),
            'sl_nt2_g':   g('SL_NTC_G'),   'sl_nt2_l':  g('SL_NTC_L'),
            'sl_nt3_g':   g('SL_NTD_G'),   'sl_nt3_l':  g('SL_NTD_L'),
            'sl_obc_g':   g('SL_OBC_G'),   'sl_obc_l':  g('SL_OBC_L'),
            'sl_sebc_g':  g('SL_SEBC_G'),  'sl_sebc_l': g('SL_SEBC_L'),
            # HU per-category
            'hu_open_g':  g('HU_OPEN_G'),  'hu_open_l':  g('HU_OPEN_L'),
            'hu_sc_g':    g('HU_SC_G'),    'hu_sc_l':    g('HU_SC_L'),
            'hu_obc_g':   g('HU_OBC_G'),   'hu_obc_l':   g('HU_OBC_L'),
            # OHU per-category
            'ohu_open_g': g('OHU_OPEN_G'), 'ohu_open_l': g('OHU_OPEN_L'),
            'ohu_sc_g':   g('OHU_SC_G'),   'ohu_sc_l':   g('OHU_SC_L'),
            'ohu_obc_g':  g('OHU_OBC_G'),  'ohu_obc_l':  g('OHU_OBC_L'),
        }

    return lookup


def _norm_str(s: str) -> str:
    """Normalise a string for fuzzy matching (lower, strip punctuation/spaces)."""
    s = str(s).lower().strip()
    s = re.sub(r"[,.'\"()]", '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def _parse_filename(fname: str):
    m = re.search(r'(20\d{2}).*?CAP(\d)', fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _normalise_cutoff(df: pd.DataFrame, year: int, cap_round: int) -> pd.DataFrame:
    col_map = {
        'College ID': 'college_id',
        'College Name': 'college_name',
        'Course ID': 'course_id',
        'Course Name': 'course_name',
        'Status': 'status',
        'Seat Type': 'seat_type',
        'Stage': 'stage',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    base_cols  = [c for c in ['college_id','college_name','course_id',
                               'course_name','status','seat_type','stage'] if c in df.columns]
    merit_cols = [c for c in df.columns if c.endswith(' Merit')]
    categories = [c.replace(' Merit', '') for c in merit_cols]

    rows = []
    for cat in categories:
        mc  = f"{cat} Merit"
        pc  = f"{cat} Percentile"
        if mc not in df.columns or pc not in df.columns:
            continue
        sub = df[base_cols + [mc, pc]].copy()
        sub = sub.dropna(subset=[pc])
        sub = sub[pd.to_numeric(sub[pc], errors='coerce') > 0]
        sub['category']   = cat
        sub['merit']      = pd.to_numeric(sub[mc], errors='coerce')
        sub['percentile'] = pd.to_numeric(sub[pc], errors='coerce')
        sub = sub.drop(columns=[mc, pc])
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result['year']      = year
    result['cap_round'] = cap_round
    return result


def _normalise_seat_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns and clean the Course Name field."""
    col_map = {
        'College ID':    'college_id',
        'College Name':  'college_name',
        'Status':        'status',
        'CAP Seats':     'cap_seats',
        'Choice Code':   'choice_code',
        'SI':            'sanctioned_intake',
        'MS Seats':      'ms_seats',
        'TFWS_Seats':    'tfws_seats',
        'EWS_Seats':     'ews_seats',
        'Orphan':        'orphan_seats',
        'SL_Total':      'sl_total',
        'HU_Total':      'hu_total',
        'OHU_Total':     'ohu_total',
        'PWD_Total':     'pwd_total',
        'DEF_Total':     'def_total',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Course Name column has trailing junk (numbers from adjacent columns)
    # Split on 2+ consecutive spaces and take the first part
    if 'Course Name' in df.columns:
        df['course_name'] = df['Course Name'].astype(str).str.split(r'\s{2,}').str[0].str.strip()
        df = df.drop(columns=['Course Name'])
    elif 'course_name' not in df.columns:
        df['course_name'] = ''

    return df


def get_available_branches(cutoff_df: pd.DataFrame) -> list:
    if cutoff_df.empty:
        return []
    return sorted(cutoff_df['course_name'].dropna().unique().tolist())


def get_available_colleges(cutoff_df: pd.DataFrame) -> list:
    if cutoff_df.empty:
        return []
    return sorted(cutoff_df['college_name'].dropna().unique().tolist())
