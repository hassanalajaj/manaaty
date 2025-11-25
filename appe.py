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

    # Balanced to avoid bias toward High
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

# 🔁 Medium معدّل بحيث يعطي Moderate (قيم أخف)
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
    )

    init_session_defaults()

    st.markdown(
        """
        <h1 style="margin-bottom:0;">Manaaty – Early Immune Risk Dashboard</h1>
        <h4 style="margin-top:4px;color:#1f77b4;">
            Clinical Prototype – Manaaty Project
        </h4>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "Manaaty helps clinicians monitor early immune activation using vitals "
        "and inflammatory biomarkers (demo simulation)."
    )

    st.markdown("---")

    with st.spinner("Loading Manaaty AI model..."):
        model = load_model()

    # ========== Sidebar ==========

    st.sidebar.header("Quick Presets (Demo Cases)")
    col_p1, col_p2, col_p3 = st.sidebar.columns(3)
    with col_p1:
        if st.button("Low"):
            apply_preset(LOW_PRESET)
    with col_p2:
        if st.button("Medium"):
            apply_preset(MODERATE_PRESET)
    with col_p3:
        if st.button("High"):
            apply_preset(HIGH_PRESET)

    st.sidebar.markdown("---")
    st.sidebar.header("Patient Information")

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
    st.sidebar.header("Inflammatory Biomarkers")

    baseline_crp = st.sidebar.number_input(
        "CRP (mg/L)", 0.0, 300.0, float(st.session_state["baseline_crp"])
    )
    baseline_il6 = st.sidebar.number_input(
        "IL-6 (pg/mL)", 0.0, 500.0, float(st.session_state["baseline_il6"])
    )
    baseline_tnf = st.sidebar.number_input(
        "TNF-α (pg/mL)", 0.0, 500.0, float(st.session_state["baseline_tnf"])
    )
    baseline_ferritin = st.sidebar.number_input(
        "Ferritin (ng/mL)", 0.0, 2000.0, float(st.session_state["baseline_ferritin"])
    )
    baseline_lymph_pct = st.sidebar.number_input(
        "Lymphocyte %", 0.0, 100.0, float(st.session_state["baseline_lymph_pct"])
    )
    baseline_neutro_pct = st.sidebar.number_input(
        "Neutrophil %", 0.0, 100.0, float(st.session_state["baseline_neutro_pct"])
    )

    # تحديث القيم في session_state بناءً على إدخال المستخدم
    st.session_state["age"] = age
    st.session_state["sex"] = sex
    st.session_state["baseline_crp"] = baseline_crp
    st.session_state["baseline_il6"] = baseline_il6
    st.session_state["baseline_tnf"] = baseline_tnf
    st.session_state["baseline_ferritin"] = baseline_ferritin
    st.session_state["baseline_lymph_pct"] = baseline_lymph_pct
    st.session_state["baseline_neutro_pct"] = baseline_neutro_pct

    # ========== Main vitals layout ==========
    st.subheader("Manaaty Patch Vitals (0–24 hours)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Temperature (°C)")
        baseline_temp = st.number_input(
            "Baseline temperature",
            34.0,
            42.0,
            float(st.session_state["baseline_temp"]),
            step=0.1,
        )
        last_temp = st.number_input(
            "Temperature at 24h",
            34.0,
            42.0,
            float(st.session_state["last_temp"]),
            step=0.1,
        )

        st.markdown("### Heart Rate (bpm)")
        baseline_hr = st.number_input(
            "Baseline HR",
            30,
            200,
            int(st.session_state["baseline_hr"]),
        )
        last_hr = st.number_input(
            "HR at 24h",
            30,
            200,
            int(st.session_state["last_hr"]),
        )

        st.markdown("### Respiratory Rate (breaths/min)")
        baseline_rr = st.number_input(
            "Baseline RR",
            5,
            60,
            int(st.session_state["baseline_rr"]),
        )
        last_rr = st.number_input(
            "RR at 24h",
            5,
            60,
            int(st.session_state["last_rr"]),
        )

    with col2:
        st.markdown("### Oxygen Saturation (SpO₂, %)")
        baseline_spo2 = st.number_input(
            "Baseline SpO₂",
            70.0,
            100.0,
            float(st.session_state["baseline_spo2"]),
            step=0.1,
        )
        last_spo2 = st.number_input(
            "SpO₂ at 24h",
            70.0,
            100.0,
            float(st.session_state["last_spo2"]),
            step=0.1,
        )

        st.markdown("### Heart Rate Variability (HRV, RMSSD ms)")
        baseline_hrv = st.number_input(
            "Baseline HRV",
            5.0,
            200.0,
            float(st.session_state["baseline_hrv"]),
            step=1.0,
        )
        last_hrv = st.number_input(
            "HRV at 24h",
            5.0,
            200.0,
            float(st.session_state["last_hrv"]),
            step=1.0,
        )

        st.markdown("### Activity Index (0–1)")
        baseline_activity = st.number_input(
            "Baseline activity index",
            0.0,
            1.0,
            float(st.session_state["baseline_activity"]),
            step=0.05,
        )
        last_activity = st.number_input(
            "Activity index at 24h",
            0.0,
            1.0,
            float(st.session_state["last_activity"]),
            step=0.05,
        )

    # تحديث vitals في session_state
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

    st.markdown("---")

    # ========== Prediction ==========
    if st.button("Run Manaaty Early Risk Assessment"):
        temp_slope = compute_slope(baseline_temp, last_temp)
        hr_slope = compute_slope(baseline_hr, last_hr)
        spo2_slope = compute_slope(baseline_spo2, last_spo2)
        hrv_slope = compute_slope(baseline_hrv, last_hrv)
        rr_slope = compute_slope(baseline_rr, last_rr)
        activity_slope = compute_slope(baseline_activity, last_activity)

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
                    "baseline_ferritin_ng_ml": baseline_ferritin,
                    "baseline_lymph_pct": baseline_lymph_pct,
                    "baseline_neutro_pct": baseline_neutro_pct,
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
        risk_colors = {
            0: ("✅", "#27ae60"),
            1: ("⚠️", "#f39c12"),
            2: ("🚨", "#e74c3c"),
        }

        icon, color = risk_colors.get(pred_class, ("ℹ️", "#7f8c8d"))
        risk_text = risk_labels.get(pred_class, "Unknown")

        st.markdown("### Manaaty Early Immune Activation Risk")
        st.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:12px;
                border:1px solid #dddddd;
                background-color:#f9f9f9;">
                <h3 style="margin:0;">
                    {icon} Risk level:
                    <span style="color:{color};">{risk_text}</span>
                </h3>
                <p style="margin-top:10px; margin-bottom:6px;">
                    Probability (Low / Moderate / High):
                </p>
                <p style="margin:0;">
                    {pred_proba[0]:.2f} / {pred_proba[1]:.2f} / {pred_proba[2]:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Patient Summary (Manaaty)")
        st.markdown(
            f"""
            - **Patient ID:** {patient_id}  
            - **Age:** {age}  
            - **Sex:** {sex}  
            """,
        )


if __name__ == "__main__":
    main()
