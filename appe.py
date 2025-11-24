import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    )
    model.fit(X_train, y_train)

    return model


def compute_slope(baseline: float, last: float, hours: float = 24.0) -> float:
    try:
        return (float(last) - float(baseline)) / float(hours)
    except Exception:
        return 0.0


def main():
    st.set_page_config(
        page_title="VaxPatch – Early Immune Risk",
        layout="wide",
    )

    st.markdown(
        """
        <h1 style="margin-bottom:0;">VaxPatch Early Immune Risk Dashboard</h1>
        <h4 style="margin-top:4px;color:#1f77b4;">
            Clinical Prototype – Hail Health Cluster
        </h4>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "This app simulates how a VaxPatch device could help clinicians monitor "
        "early immune activation using vitals and inflammatory biomarkers."
    )

    st.markdown("---")

    with st.spinner("Loading AI model..."):
        model = load_model()

    st.sidebar.header("Patient Information")
    patient_id = st.sidebar.text_input("Patient ID", value="P-001")
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=110, value=35)
    sex = st.sidebar.selectbox("Sex", ["Not specified", "Male", "Female"])

    st.sidebar.markdown("---")
    st.sidebar.header("Inflammatory Biomarkers")

    baseline_crp = st.sidebar.number_input("CRP (mg/L)", 0.0, 300.0, 5.0)
    baseline_il6 = st.sidebar.number_input("IL-6 (pg/mL)", 0.0, 500.0, 4.0)
    baseline_tnf = st.sidebar.number_input("TNF-α (pg/mL)", 0.0, 500.0, 6.0)
    baseline_ferritin = st.sidebar.number_input("Ferritin (ng/mL)", 0.0, 2000.0, 150.0)
    baseline_lymph_pct = st.sidebar.number_input("Lymphocyte %", 0.0, 100.0, 25.0)
    baseline_neutro_pct = st.sidebar.number_input("Neutrophil %", 0.0, 100.0, 60.0)

    st.subheader("Patch Vitals (0–24 hours)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Temperature (°C)")
        baseline_temp = st.number_input(
            "Baseline temperature", 34.0, 42.0, 37.0, step=0.1
        )
        last_temp = st.number_input(
            "Temperature at 24h", 34.0, 42.0, 38.0, step=0.1
        )

        st.markdown("### Heart Rate (bpm)")
        baseline_hr = st.number_input("Baseline HR", 30, 200, 80)
        last_hr = st.number_input("HR at 24h", 30, 200, 100)

        st.markdown("### Respiratory Rate (breaths/min)")
        baseline_rr = st.number_input("Baseline RR", 5, 60, 16)
        last_rr = st.number_input("RR at 24h", 5, 60, 20)

    with col2:
        st.markdown("### Oxygen Saturation (SpO₂, %)")
        baseline_spo2 = st.number_input(
            "Baseline SpO₂", 70.0, 100.0, 98.0, step=0.1
        )
        last_spo2 = st.number_input(
            "SpO₂ at 24h", 70.0, 100.0, 96.0, step=0.1
        )

        st.markdown("### Heart Rate Variability (HRV, RMSSD ms)")
        baseline_hrv = st.number_input(
            "Baseline HRV", 5.0, 200.0, 60.0, step=1.0
        )
        last_hrv = st.number_input(
            "HRV at 24h", 5.0, 200.0, 40.0, step=1.0
        )

        st.markdown("### Activity Index (0–1)")
        baseline_activity = st.number_input(
            "Baseline activity index", 0.0, 1.0, 0.7, step=0.05
        )
        last_activity = st.number_input(
            "Activity index at 24h", 0.0, 1.0, 0.3, step=0.05
        )

    st.markdown("---")

    if st.button("Run Early Risk Assessment"):
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

        st.markdown("### Early Immune Activation Risk")
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

        st.markdown("### Patient Summary")
        st.markdown(
            f"""
            - **Patient ID:** {patient_id}  
            - **Age:** {age}  
            - **Sex:** {sex}  
            """,
        )


if __name__ == "__main__":
    main()
