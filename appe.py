import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================================
# 1) Model loading
# ======================================
@st.cache_resource
def load_model():
    data_path = "vaxpatch_synthetic_timeseries_1850.csv"
    df = pd.read_csv(data_path)

    feature_cols = [
        "age",
        "baseline_temp_c",
        "baseline_hr_bpm",
        "baseline_hrv_rmssd_ms",
        "baseline_spo2",
        "baseline_rr_bpm",
        "baseline_activity_index",
        "baseline_crp_mg_l",
        "baseline_il6_pg_ml",
        "baseline_tnf_alpha_pg_ml",
        "baseline_ferritin_ng_ml",
        "baseline_lymph_pct",
        "baseline_neutro_pct",
        "temp_slope_0_24",
        "hr_slope_0_24",
        "spo2_slope_0_24",
        "hrv_slope_0_24",
        "rr_slope_0_24",
        "activity_slope_0_24",
    ]

    X = df[feature_cols]
    y = df["early_risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    return model


def compute_slope(baseline: float, last: float, hours: float = 24.0) -> float:
    """Slope = (last - baseline) / hours."""
    try:
        return (float(last) - float(baseline)) / float(hours)
    except Exception:
        return 0.0


# ======================================
# 2) Preset profiles (Low / Moderate / High)
# ======================================
LOW_PRESET = {
    "age": 25,
    "sex": "Male",
    "baseline_crp": 0.5,
    "baseline_il6": 0.3,
    "baseline_tnf": 1.5,
    "baseline_ferritin": 75.0,
    "baseline_lymph_pct": 35.0,
    "baseline_neutro_pct": 50.0,
    "baseline_temp": 36.7,
    "last_temp": 36.7,
    "baseline_hr": 70,
    "last_hr": 70,
    "baseline_rr": 14,
    "last_rr": 14,
    "baseline_spo2": 98.0,
    "last_spo2": 98.0,
    "baseline_hrv": 60.0,
    "last_hrv": 60.0,
    "baseline_activity": 0.6,
    "last_activity": 0.6,
}

MODERATE_PRESET = {
    "age": 35,
    "sex": "Male",
    "baseline_crp": 1.2,
    "baseline_il6": 0.8,
    "baseline_tnf": 2.0,
    "baseline_ferritin": 100.0,
    "baseline_lymph_pct": 30.0,
    "baseline_neutro_pct": 58.0,
    "baseline_temp": 36.8,
    "last_temp": 37.0,
    "baseline_hr": 72,
    "last_hr": 76,
    "baseline_rr": 14,
    "last_rr": 16,
    "baseline_spo2": 98.0,
    "last_spo2": 97.0,
    "baseline_hrv": 60.0,
    "last_hrv": 50.0,
    "baseline_activity": 0.6,
    "last_activity": 0.5,
}

HIGH_PRESET = {
    "age": 55,
    "sex": "Male",
    "baseline_crp": 8.0,
    "baseline_il6": 6.0,
    "baseline_tnf": 10.0,
    "baseline_ferritin": 300.0,
    "baseline_lymph_pct": 20.0,
    "baseline_neutro_pct": 70.0,
    "baseline_temp": 37.5,
    "last_temp": 38.2,
    "baseline_hr": 90,
    "last_hr": 110,
    "baseline_rr": 20,
    "last_rr": 28,
    "baseline_spo2": 95.0,
    "last_spo2": 92.0,
    "baseline_hrv": 50.0,
    "last_hrv": 25.0,
    "baseline_activity": 0.5,
    "last_activity": 0.2,
}


def apply_preset(preset: dict):
    """Store preset in session_state then rerun."""
    for k, v in preset.items():
        st.session_state[k] = v
    st.rerun()


def init_session_defaults():
    """Initialize session_state with normal-ish defaults if not set."""
    if "age" not in st.session_state:
        for k, v in LOW_PRESET.items():
            st.session_state[k] = v


# ======================================
# 3) Streamlit app
# ======================================
def main():
    st.set_page_config(
        page_title="Manaaty – Early Immune Risk",
        layout="wide",
        page_icon="🧬",
    )

    init_session_defaults()

    # ---------- Global Dark Neon CSS ----------
    st.markdown(
        """
        <style>
        /* Background & text */
        body {
            background-color: #050713;
            color: #e4e6eb;
        }
        .main {
            background-color: #050713;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Generic card */
        .m-card {
            background: radial-gradient(circle at top left, #1b2140 0, #0c0f1c 45%, #050713 100%);
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid #242a43;
            box-shadow: 0 0 24px rgba(0, 200, 255, 0.12);
            margin-bottom: 18px;
        }

        .m-title {
            font-size: 18px;
            font-weight: 600;
            color: #d0dcff;
            margin-bottom: 10px;
        }

        h1, h2, h3, h4 {
            color: #f5f6ff;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #060815;
            border-right: 1px solid #15182b;
        }

        /* Inputs label */
        .stNumberInput label, .stTextInput label, .stSelectbox label {
            color: #b8bedc !important;
            font-weight: 500 !important;
        }

        /* Generic button (main area) */
        .stButton>button {
            border-radius: 999px;
            padding: 10px 16px;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #303553, #191b2b);
            color: #fff;
            border: 1px solid #414872;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #3b4270, #20233a);
            border-color: #5c6cff;
        }

        /* Sidebar preset buttons: prevent wrapping + full width */
        section[data-testid="stSidebar"] .stButton>button {
            white-space: nowrap;
            width: 100%;
            height: 40px;
            font-size: 14px;
        }

        /* Risk card */
        .risk-card {
            border-radius: 16px;
            padding: 18px;
            margin-top: 8px;
        }

        .risk-low {
            background: rgba(46, 204, 113, 0.12);
            border: 1px solid rgba(46, 204, 113, 0.4);
            color: #2ecc71;
        }
        .risk-mod {
            background: rgba(243, 156, 18, 0.12);
            border: 1px solid rgba(243, 156, 18, 0.4);
            color: #f1c40f;
        }
        .risk-high {
            background: rgba(231, 76, 60, 0.18);
            border: 1px solid rgba(231, 76, 60, 0.6);
            color: #ff6b6b;
        }

        /* Vital pill row */
        .vital-pill {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 20, 45, 0.95);
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 6px;
            border: 1px solid #222849;
            font-size: 13px;
            color: #dde3ff;
        }
        .vital-label {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .vital-icon {
            font-size: 15px;
        }
        .vital-value {
            font-weight: 600;
        }

        .subsection-title {
            font-size: 14px;
            font-weight: 600;
            color: #9fa9ff;
            margin: 10px 0 4px 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Header ----------
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            """
            <h1 style="margin-bottom:6px;">Early Immune Activation Dashboard</h1>
            <p style="margin-top:0; margin-bottom:4px; color:#d0dcff; font-weight:600;">
                Clinical Prototype – Manaaty
            </p>
            <p style="color:#9ca3c7; font-size:14px; margin-top:0;">
                AI-assisted risk stratification using patch-based vitals and inflammatory biomarkers.
            </p>
            """,
            unsafe_allow_html=True,
        )
    # العمود الثاني فاضي الآن (كان فيه st.info قبل)
    with header_col2:
        st.empty()

    st.markdown("---")

    # ---------- Load model ----------
    with st.spinner("Loading Manaaty AI model..."):
        model = load_model()

    # ======================================
    # Sidebar controls
    # ======================================
    st.sidebar.title("🩺 Manaaty Controls")

    st.sidebar.caption("Use presets or adjust inputs to explore different immune activation patterns.")

    st.sidebar.subheader("Quick Presets")
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("Low", key="preset_low"):
            apply_preset(LOW_PRESET)
    with c2:
        if st.button("Med", key="preset_med"):
            apply_preset(MODERATE_PRESET)
    with c3:
        if st.button("High", key="preset_high"):
            apply_preset(HIGH_PRESET)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Patient Information")

    patient_id = st.sidebar.text_input("Patient ID", value="P-001")
    age = st.sidebar.number_input(
        "Age (years)", min_value=0, max_value=110, value=int(st.session_state["age"])
    )
    sex = st.sidebar.selectbox(
        "Sex",
        ["Not specified", "Male", "Female"],
        index=["Not specified", "Male", "Female"].index(st.session_state["sex"])
        if st.session_state.get("sex") in ["Not specified", "Male", "Female"]
        else 0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Inflammatory Biomarkers")

    baseline_crp = st.sidebar.number_input(
        "CRP (mg/L)", 0.0, 300.0, float(st.session_state["baseline_crp"])
    )
    baseline_il6 = st.sidebar.number_input(
        "IL-6 (pg/mL)", 0.0, 500.0, float(st.session_state["baseline_il6"])
    )
    baseline_tnf = st.sidebar.number_input(
        "TNF-α (pg/mL)", 0.0, 500.0, float(st.session_state["baseline_tnf"])
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Patch Vitals (0–24h)")

    baseline_temp = st.sidebar.number_input(
        "Baseline Temperature (°C)", 34.0, 42.0, float(st.session_state["baseline_temp"]), step=0.1
    )
    last_temp = st.sidebar.number_input(
        "Temperature at 24h (°C)", 34.0, 42.0, float(st.session_state["last_temp"]), step=0.1
    )

    baseline_hr = st.sidebar.number_input(
        "Baseline Heart Rate (bpm)", 30, 200, int(st.session_state["baseline_hr"])
    )
    last_hr = st.sidebar.number_input(
        "Heart Rate at 24h (bpm)", 30, 200, int(st.session_state["last_hr"])
    )

    baseline_rr = st.sidebar.number_input(
        "Baseline Respiratory Rate (breaths/min)", 5, 60, int(st.session_state["baseline_rr"])
    )
    last_rr = st.sidebar.number_input(
        "Respiratory Rate at 24h (breaths/min)", 5, 60, int(st.session_state["last_rr"])
    )

    baseline_spo2 = st.sidebar.number_input(
        "Baseline SpO₂ (%)", 70.0, 100.0, float(st.session_state["baseline_spo2"]), step=0.1
    )
    last_spo2 = st.sidebar.number_input(
        "SpO₂ at 24h (%)", 70.0, 100.0, float(st.session_state["last_spo2"]), step=0.1
    )

    baseline_hrv = st.sidebar.number_input(
        "Baseline HRV (RMSSD ms)", 5.0, 200.0, float(st.session_state["baseline_hrv"]), step=1.0
    )
    last_hrv = st.sidebar.number_input(
        "HRV at 24h (RMSSD ms)", 5.0, 200.0, float(st.session_state["last_hrv"]), step=1.0
    )

    baseline_activity = st.sidebar.number_input(
        "Baseline Activity Index (0–1)", 0.0, 1.0, float(st.session_state["baseline_activity"]), step=0.05
    )
    last_activity = st.sidebar.number_input(
        "Activity Index at 24h (0–1)", 0.0, 1.0, float(st.session_state["last_activity"]), step=0.05
    )

    # Update session_state (بدون الثلاثة اللي الدكتور ما يبيها في الواجهة)
    st.session_state["age"] = age
    st.session_state["sex"] = sex
    st.session_state["baseline_crp"] = baseline_crp
    st.session_state["baseline_il6"] = baseline_il6
    st.session_state["baseline_tnf"] = baseline_tnf
    st.session_state["baseline_temp"] = baseline_temp
    st.session_state["last_temp"] = last_temp
    st.session_state["baseline_hr"] = baseline_hr
    st.session_state["last_hr"] = last_hr
    st.session_state["baseline_rr"] = baseline_rr
    st.session_state["last_rr"] = last_rr
    st.session_state["baseline_spo2"] = baseline_spo2
    st.session_state["last_spo2"] = last_spo2
    st.session_state["baseline_hrv"] = baseline_hrv
    st.session_state["last_hrv"] = last_hrv
    st.session_state["baseline_activity"] = baseline_activity
    st.session_state["last_activity"] = last_activity

    # ======================================
    # Main layout (like the screenshot)
    # ======================================
    left_col, right_col = st.columns([1.05, 1.7])

    # ------- Left: Input Data Summary -------
    with left_col:
        st.markdown('<div class="m-card">', unsafe_allow_html=True)
        st.markdown('<div class="m-title">Input Data</div>', unsafe_allow_html=True)

        # Vitals
        st.markdown('<div class="subsection-title">Vitals</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">🌡️</span>Temperature</div>
                <div class="vital-value">{last_temp:.1f} °C</div>
            </div>
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">💓</span>Heart Rate</div>
                <div class="vital-value">{last_hr:.0f} bpm</div>
            </div>
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">🌬️</span>Respiratory Rate</div>
                <div class="vital-value">{last_rr:.0f} breaths</div>
            </div>
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">🫁</span>Oxygen Saturation</div>
                <div class="vital-value">{last_spo2:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Biomarkers (بدون Ferritin / Lymph / Neutro)
        st.markdown('<div class="subsection-title">Inflammatory Biomarkers</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">➕</span>CRP</div>
                <div class="vital-value">{baseline_crp:.1f} mg/L</div>
            </div>
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">➕</span>IL-6</div>
                <div class="vital-value">{baseline_il6:.1f} pg/mL</div>
            </div>
            <div class="vital-pill">
                <div class="vital-label"><span class="vital-icon">➕</span>TNF-α</div>
                <div class="vital-value">{baseline_tnf:.1f} pg/mL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Patient summary
        st.markdown('<div class="subsection-title">Patient Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="vital-pill">
                <div class="vital-label">Patient ID</div>
                <div class="vital-value">{patient_id}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ------- Right: Risk card + Trend -------
    with right_col:
        # === Assessment Card ===
        st.markdown('<div class="m-card">', unsafe_allow_html=True)
        st.markdown('<div class="m-title">Early Immune Activation Risk</div>', unsafe_allow_html=True)

        run_button = st.button("Run Assessment", use_container_width=True)

        pred_class = None
        pred_proba = None

        if run_button:
            temp_slope = compute_slope(baseline_temp, last_temp)
            hr_slope = compute_slope(baseline_hr, last_hr)
            spo2_slope = compute_slope(baseline_spo2, last_spo2)
            hrv_slope = compute_slope(baseline_hrv, last_hrv)
            rr_slope = compute_slope(baseline_rr, last_rr)
            activity_slope = compute_slope(baseline_activity, last_activity)

            # قيم Ferritin / Lymph / Neutro من session_state فقط (مخفية عن الواجهة)
            ferritin_val = st.session_state.get("baseline_ferritin", 150.0)
            lymph_val = st.session_state.get("baseline_lymph_pct", 30.0)
            neutro_val = st.session_state.get("baseline_neutro_pct", 60.0)

            input_row = pd.DataFrame(
                [
                    {
                        "age": age,
                        "baseline_temp_c": baseline_temp,
                        "baseline_hr_bpm": baseline_hr,
                        "baseline_hrv_rmssd_ms": baseline_hrv,
                        "baseline_spo2": baseline_spo2,
                        "baseline_rr_bpm": baseline_rr,
                        "baseline_activity_index": baseline_activity,
                        "baseline_crp_mg_l": baseline_crp,
                        "baseline_il6_pg_ml": baseline_il6,
                        "baseline_tnf_alpha_pg_ml": baseline_tnf,
                        "baseline_ferritin_ng_ml": ferritin_val,
                        "baseline_lymph_pct": lymph_val,
                        "baseline_neutro_pct": neutro_val,
                        "temp_slope_0_24": temp_slope,
                        "hr_slope_0_24": hr_slope,
                        "spo2_slope_0_24": spo2_slope,
                        "hrv_slope_0_24": hrv_slope,
                        "rr_slope_0_24": rr_slope,
                        "activity_slope_0_24": activity_slope,
                    }
                ]
            )

            pred_class = int(model.predict(input_row)[0])
            pred_proba = model.predict_proba(input_row)[0]

            risk_labels = {0: "Low", 1: "Moderate", 2: "High"}
            risk_classes = {0: "risk-low", 1: "risk-mod", 2: "risk-high"}
            risk_icons = {0: "✅", 1: "⚠️", 2: "🚨"}

            interpretations = {
                0: "Signal consistent with **low early immune activation**. Continue routine monitoring.",
                1: "There are **moderate early changes**. Consider closer follow-up and clinical correlation.",
                2: "Strong signal of **high early immune activation**. Prioritize clinical review and action.",
            }

            r_label = risk_labels.get(pred_class, "Unknown")
            r_css = risk_classes.get(pred_class, "risk-low")
            r_icon = risk_icons.get(pred_class, "ℹ️")
            r_text = interpretations.get(pred_class, "")

            st.markdown(
                f"""
                <div class="risk-card {r_css}">
                    <div style="font-size:20px; font-weight:600; margin-bottom:4px;">
                        {r_icon} {r_label} Risk
                    </div>
                    <div style="font-size:13px; color:#fcefff;">
                        {r_text}
                    </div>
                    <div style="font-size:10px; color:#aaaaaa; margin-top:8px;">
                        *AI-assisted prediction – not a medical diagnosis.*
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability distribution
            st.markdown("**Class Probabilities (Low / Moderate / High)**")
            labels = ["Low", "Moderate", "High"]
            for i, label in enumerate(labels):
                p = float(pred_proba[i])
                st.write(f"{label}: **{p:.2f}**")
                st.progress(int(p * 100))

        else:
            st.caption("Click **Run Assessment** to generate risk level and probabilities.")

        st.markdown("</div>", unsafe_allow_html=True)

        # === Trend Analysis Card ===
        st.markdown('<div class="m-card">', unsafe_allow_html=True)
        st.markdown('<div class="m-title">Trend Analysis (0–24 hours)</div>', unsafe_allow_html=True)

        # Simple 3-point trend (0h, 12h, 24h) for look only
        hours = [0, 12, 24]
        temp_trend = [baseline_temp, (baseline_temp + last_temp) / 2, last_temp]
        hr_trend = [baseline_hr, (baseline_hr + last_hr) / 2, last_hr]
        rr_trend = [baseline_rr, (baseline_rr + last_rr) / 2, last_rr]
        crp_trend = [baseline_crp * 0.7, baseline_crp * 0.9, baseline_crp]

        trend_df = pd.DataFrame(
            {
                "Temperature": temp_trend,
                "Heart Rate": hr_trend,
                "Resp. Rate": rr_trend,
                "CRP": crp_trend,
            },
            index=hours,
        )

        st.line_chart(trend_df)

        st.markdown(
            f"""
            <div style="margin-top:10px; font-size:13px; color:#a5afdd;">
                <b>Patient Summary</b><br/>
                Age: <span style="color:#ffffff;">{age}</span> &nbsp;&nbsp;
                Sex: <span style="color:#ffffff;">{sex}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
