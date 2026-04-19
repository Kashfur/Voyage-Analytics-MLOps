
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="JourneyIQ | Travel Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.hero-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem 2.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
}
.hero-banner h1 { font-size: 2.2rem; margin: 0; }
.hero-banner p  { font-size: 1rem; opacity: 0.82; margin-top: 0.4rem; }
.result-box {
    background: linear-gradient(90deg, #e0f2fe, #f0fdf4);
    border: 1px solid #93c5fd; border-radius: 10px;
    padding: 1.2rem 1.5rem; margin-top: 1rem;
}
.result-box h3 { color: #1d4ed8; margin-top: 0; }
.footer { text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1>✈️ JourneyIQ — Travel Intelligence Platform</h1>
    <p>Predict flight fares · Classify traveller profiles · Discover personalised hotels</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ML models…")
def load_artefacts():
    art = {}
    paths = {
        "fare_model":      "models/flight_price_model.joblib",
        "encoders":        "models/label_encoders.joblib",
        "feature_cols":    "models/fare_feature_cols.joblib",
        "gender_model":    "models/gender_clf_model.joblib",
        "gender_enc":      "models/gender_label_encoder.joblib",
        "company_enc":     "models/company_label_encoder.joblib",
        "hotel_matrix":    "models/hotel_tfidf_matrix.joblib",
        "hotels_df":       "models/hotels_metadata.joblib",
        "hotel_vectoriser":"models/hotel_vectoriser.joblib",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            art[key] = joblib.load(path)
    return art


@st.cache_data(show_spinner=False)
def load_raw_data():
    data = {}
    for fname in ["flights", "users", "hotels"]:
        p = f"data/{fname}.csv"
        if os.path.exists(p):
            data[fname] = pd.read_csv(p)
    return data


artefacts = load_artefacts()
raw       = load_raw_data()

# Real city options from the dataset
CITIES = [
    "Recife (PE)", "Florianopolis (SC)", "Brasilia (DF)",
    "Aracaju (SE)", "Salvador (BH)", "Campo Grande (MS)",
    "Sao Paulo (SP)", "Natal (RN)", "Rio de Janeiro (RJ)"
]
FLIGHT_TYPES = ["economic", "firstClass", "premium"]
AGENCIES     = ["FlyingDrops", "CloudFy", "Rainbow"]

# Real company options
COMPANIES = ["4You", "Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA"]


tab1, tab2, tab3, tab4 = st.tabs([
    "🛫 Fare Estimator",
    "👤 Traveller Classifier",
    "🏨 Hotel Discovery",
    "📊 Data Explorer"
])

# TAB 1 — FARE ESTIMATOR
with tab1:
    st.subheader("Real-Time Flight Fare Estimator")
    st.caption("Brazilian domestic routes · Prices in local currency (BRL)")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        origin      = st.selectbox("Departure City", CITIES, index=0)
        destination = st.selectbox("Arrival City",   CITIES, index=2)
        flight_type = st.radio("Cabin Class", FLIGHT_TYPES, horizontal=True)
    with c2:
        agency   = st.selectbox("Airline Agency", AGENCIES)
        duration = st.slider("Flight Duration (hours)", 0.44, 2.44, 1.46, step=0.05,
                             help="Real range in dataset: 0.44 – 2.44 hrs")
        distance = st.slider("Route Distance (km)", 168, 938, 563, step=10,
                             help="Real range in dataset: 168 – 938 km")

    run_btn = st.button("🔍 Predict Fare", type="primary")

    if run_btn:
        payload = {
            "from": origin, "to": destination,
            "flightType": flight_type, "agency": agency,
            "time": duration, "distance": distance
        }

        predicted = None
        source    = "Local model"

        # Try live API first
        try:
            resp = requests.post(
                "http://localhost:5050/api/v1/predict/fare",
                json=payload, timeout=2
            )
            if resp.status_code == 200:
                d = resp.json()
                predicted = d.get("predicted_price")
                low  = d.get("confidence_band", {}).get("low",  predicted * 0.90)
                high = d.get("confidence_band", {}).get("high", predicted * 1.10)
                source = "REST API"
        except Exception:
            pass

        # Fall back to local model
        if predicted is None and "fare_model" in artefacts:
            enc  = artefacts.get("encoders", {})
            fcols = artefacts.get("feature_cols",
                    ["from","to","flightType","agency","time","distance","speed_proxy"])

            def safe_enc(col, val):
                e = enc.get(col)
                if e:
                    try:   return int(e.transform([str(val).strip()])[0])
                    except: return -1
                return -1

            spd = distance / max(duration, 0.1)
            row_map = {
                "from": safe_enc("from", origin),
                "to":   safe_enc("to", destination),
                "flightType": safe_enc("flightType", flight_type),
                "agency":     safe_enc("agency", agency),
                "time":        duration,
                "distance":    distance,
                "speed_proxy": spd,
            }
            fv = np.array([[row_map.get(f, 0) for f in fcols]])
            predicted = float(artefacts["fare_model"].predict(fv)[0])
            low  = predicted * 0.90
            high = predicted * 1.10

        if predicted is not None:
            st.markdown(f"""
            <div class="result-box">
                <h3>Estimated Fare</h3>
                <div style="font-size:2.5rem;font-weight:800;color:#065f46;">
                    R$ {predicted:,.2f}
                </div>
                <div style="color:#6b7280;margin-top:0.4rem;">
                    Confidence range: R${low:,.0f} – R${high:,.0f}
                    &nbsp;|&nbsp; Source: <strong>{source}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted,
                title={"text": "Predicted Fare (R$)"},
                gauge={
                    "axis": {"range": [300, 1800]},
                    "bar":  {"color": "#2563eb"},
                    "steps": [
                        {"range": [300,  700],  "color": "#dcfce7"},
                        {"range": [700,  1200], "color": "#fef9c3"},
                        {"range": [1200, 1800], "color": "#fee2e2"},
                    ],
                }
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run `python train_fare_model.py --data data/flights.csv` first to train the model.")

# TAB 2 — TRAVELLER CLASSIFIER
with tab2:
    st.subheader("Traveller Profile Classifier")
    st.caption("Predict traveller gender from age and company affiliation.")

    c1, c2 = st.columns(2)
    with c1:
        age     = st.slider("Age", 21, 65, 35)
    with c2:
        company = st.selectbox("Company", COMPANIES)

    classify_btn = st.button("🔮 Classify Profile", type="primary")

    if classify_btn:
        if "gender_model" in artefacts and "company_enc" in artefacts:
            le_company = artefacts["company_enc"]
            le_gender  = artefacts["gender_enc"]
            clf        = artefacts["gender_model"]

            try:
                company_code = int(le_company.transform([company.strip()])[0])
            except ValueError:
                company_code = 0

            fv    = np.array([[age, company_code]])
            pred  = clf.predict(fv)[0]
            proba = clf.predict_proba(fv)[0]

            classes = le_gender.classes_   # ['female', 'male']
            gender  = le_gender.inverse_transform([pred])[0]
            conf    = float(max(proba)) * 100
            icon    = "👩" if gender == "female" else "👨"

            st.success(f"{icon}  Predicted Gender: **{gender.capitalize()}**  (confidence: {conf:.1f}%)")

            fig = px.bar(
                x=[c.capitalize() for c in classes],
                y=[p * 100 for p in proba],
                labels={"x": "Gender", "y": "Probability (%)"},
                color=[c.capitalize() for c in classes],
                color_discrete_map={"Female": "#ec4899", "Male": "#3b82f6"},
                title="Classification Confidence"
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Show company breakdown from actual data
            if "users" in raw:
                users_df = raw["users"].copy()
                users_df = users_df[users_df["gender"].str.lower() != "none"]
                breakdown = users_df.groupby(["company","gender"]).size().reset_index(name="count")
                fig2 = px.bar(breakdown, x="company", y="count", color="gender",
                              barmode="group",
                              color_discrete_map={"female":"#ec4899","male":"#3b82f6","none":"gray"},
                              title="Gender Distribution by Company (Training Data)")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Run `python train_gender_model.py --data data/users.csv` first.")
# TAB 3 — HOTEL DISCOVERY
with tab3:
    st.subheader("Personalised Hotel Discovery Engine")
    st.caption("Based on 9 hotels across 9 Brazilian cities.")

    pref_city   = st.selectbox("Preferred City", CITIES)
    budget      = st.select_slider("Budget Preference",
                                   options=["budget","mid-range","comfortable","luxury"])
    top_n       = st.slider("Number of Recommendations", 1, 9, 5)

    discover_btn = st.button("🔍 Find Hotels", type="primary")

    if discover_btn:
        if all(k in artefacts for k in ["hotel_vectoriser","hotel_tfidf_matrix","hotels_df"]):
            vectoriser   = artefacts["hotel_vectoriser"]
            tfidf_matrix = artefacts["hotel_tfidf_matrix"]
            hotels_df    = artefacts["hotels_df"]

            query = f"{pref_city.lower()} {budget}"
            q_vec = vectoriser.transform([query])
            scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

            top_idx = scores.argsort()[::-1][:top_n]
            results = hotels_df.iloc[top_idx].copy()
            results["match_score"] = scores[top_idx]

            for _, row in results.iterrows():
                with st.expander(
                    f"🏨 {row['name']} — {row['place']}  |  R${row['avg_price']:.2f}/night"
                ):
                    col_a, col_b = st.columns(2)
                    col_a.metric("Avg Price / Night", f"R$ {row['avg_price']:.2f}")
                    col_b.metric("Avg Stay",          f"{row['avg_days']:.1f} nights")
                    st.metric("Total Bookings", int(row["booking_count"]))
                    st.progress(
                        float(min(row["match_score"], 1.0)),
                        text=f"Match score: {row['match_score']:.2%}"
                    )
        else:
            st.warning("Run `python train_recommender.py --data data/hotels.csv` first.")

#DATA EXPLORER
with tab4:
    st.subheader("Dataset Explorer")

    if raw:
        ds_choice = st.selectbox("Choose dataset", list(raw.keys()))
        df = raw[ds_choice]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows",           f"{len(df):,}")
        col_b.metric("Columns",        len(df.columns))
        col_c.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("View first 100 rows"):
            st.dataframe(df.head(100), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("Plot distribution of:", numeric_cols)
            fig = px.histogram(df, x=col, nbins=40,
                               title=f"Distribution of '{col}'",
                               color_discrete_sequence=["#3b82f6"])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Extra: flights price by flight type
        if ds_choice == "flights" and "flightType" in df.columns:
            fig2 = px.box(df, x="flightType", y="price", color="flightType",
                          title="Price Distribution by Flight Type",
                          color_discrete_map={
                              "economic":"#3b82f6",
                              "firstClass":"#8b5cf6",
                              "premium":"#f59e0b"
                          })
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.scatter(df.sample(min(5000, len(df))),
                              x="distance", y="price", color="flightType",
                              title="Price vs Distance (sample of 5,000)",
                              opacity=0.5)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Place flights.csv, users.csv, hotels.csv in the data/ folder.")

st.markdown("""
<div class="footer">
    JourneyIQ Travel Intelligence Platform &nbsp;·&nbsp;
    Python · Flask · Streamlit · XGBoost · scikit-learn · Docker · Kubernetes<br>
    Masters Project — MLOps Specialisation
</div>
""", unsafe_allow_html=True)
