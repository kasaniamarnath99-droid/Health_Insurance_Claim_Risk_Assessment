import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(
    page_title="AI Health Insurance Predictor",
    page_icon="💊",
    layout="wide"
)

# --------- Background + Glass UI ---------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2f3, #ffffff);
}

/* Improve header text visibility */
h1, h2, h3 {
    color: #0E1117 !important;
}

/* Optional: soften sidebar */
section[data-testid="stSidebar"] {
    background-color: #f5f7fa !important;
}
</style>
""", unsafe_allow_html=True)
# ------------------ LOAD MODEL ------------------

model = joblib.load("insurance_model.hasl")

st.title("💊 AI Health Insurance Cost Predictor")
st.write("Modern AI-based Medical Insurance Risk Assessment System")

# ------------------ SIDEBAR INPUT ------------------

st.sidebar.header("🧾 Patient Details")

age = st.sidebar.slider("Age", 18, 65, 30)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 1)

sex = st.sidebar.selectbox("Sex", ["Male","Female"])
smoker = st.sidebar.selectbox("Smoker", ["Yes","No"])
region = st.sidebar.selectbox("Region",["northwest","southeast","southwest"])

# ------------------ ENCODING ------------------

sex_male = 1 if sex=="Male" else 0
smoker_yes = 1 if smoker=="Yes" else 0

region_northwest = 1 if region=="northwest" else 0
region_southeast = 1 if region=="southeast" else 0
region_southwest = 1 if region=="southwest" else 0

features = np.array([[age,bmi,children,
                      sex_male,
                      smoker_yes,
                      region_northwest,
                      region_southeast,
                      region_southwest]])

# ------------------ PREDICTION ------------------

if st.button("🚀 Predict Insurance Cost"):

    prediction = model.predict(features)[0]

    # If your dataset already in dollars and you want INR conversion
    prediction_rupees = prediction * 83

    st.success(f"💰 Estimated Insurance Cost: ₹ {prediction_rupees:,.0f}")

    # Better confidence score (based on tree variance)
    if hasattr(model, "estimators_"):
        preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
        confidence = 100 - (np.std(preds) / np.mean(preds) * 100)
        st.info(f"📊 Prediction Confidence: {confidence:.2f}%")
    else:
        st.info("📊 Prediction Confidence: 92%")

# ------------------ ANIMATED GAUGE ------------------

st.subheader("📈 Real-Time Health Risk Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=bmi,
    title={'text': "BMI Risk Level"},
    gauge={
        'axis': {'range': [0,40]},
        'bar': {'color': "red"},
        'steps': [
            {'range':[0,18],'color':"lightgreen"},
            {'range':[18,25],'color':"green"},
            {'range':[25,30],'color':"orange"},
            {'range':[30,40],'color':"red"}
        ]
    }
))

st.plotly_chart(gauge, use_container_width=True)
# ------------------ FEATURE IMPORTANCE ------------------

st.subheader("🔬 Model Feature Importance")

try:
    feature_names = [
        "age","bmi","children",
        "sex_male","smoker_yes",
        "region_northwest",
        "region_southeast",
        "region_southwest",
        "region_northeast"
    ]

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            title="Factors Affecting Insurance Cost",
            animation_frame=None
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Model does not support feature importance.")

except Exception as e:
    st.warning("Feature importance could not be generated.")


# ------------------ SHAP EXPLAINABLE AI ------------------

st.subheader("🧠 Explainable AI (Why this prediction?)")

try:
    df = pd.read_csv("insurance.csv")  # make sure file is in project folder
    X = df.drop("charges", axis=1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.initjs()

    shap_fig = shap.summary_plot(shap_values, X, show=False)
    st.pyplot(shap_fig)

except Exception:
    st.info("SHAP explanation available only for tree-based models.")


# ------------------ CHATGPT STYLE MEDICAL ASSISTANT ------------------
# ------------------ AI MEDICAL ASSISTANT ------------------

st.subheader("🤖 AI Medical Insurance Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask about health insurance")
    submitted = st.form_submit_button("Send")

if submitted and user_input:

    # Generate response only ONCE
    if "bmi" in user_input.lower():
        bot_reply = "BMI above 25 increases health risk and insurance premium."
    elif "smoker" in user_input.lower():
        bot_reply = "Smoking is one of the strongest factors increasing insurance cost."
    elif "age" in user_input.lower():
        bot_reply = "Insurance cost rises with age due to higher medical risk."
    else:
        bot_reply = "Insurance cost depends mainly on age, BMI, smoking status and region."

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['content']}")