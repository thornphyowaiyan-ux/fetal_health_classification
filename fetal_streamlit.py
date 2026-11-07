import streamlit as st
import pickle
import os
import pandas as pd


st.set_page_config(page_title='Fetal Health Classification App')

st.subheader("📊 About the Fetal Health Dataset")

# Highlight the core clinical mission using info and metric blocks
st.info(
    "**Clinical Context:** Cardiotocograms (CTGs) provide a simple, cost-accessible option to assess fetal health, "
    "helping healthcare professionals prevent preventable child and maternal mortality in alignment with UN Sustainable Development Goals."
)

st.markdown(
    "📊 **Dataset Summary:** 2,126 Records &nbsp;|&nbsp; 3 Classes (Normal, Suspect, Pathological) &nbsp;|&nbsp; 21 Features &nbsp;|&nbsp; 85.92% Accuracy &nbsp;|&nbsp; Medical Disclaimer & Read Sign Acknowledgment"
)
with st.expander("📖 Read Full Dataset Background & Citation"):
  st.markdown("""
    ### Overview
    This dataset contains features extracted from Cardiotocogram exams, classified by expert obstetricians into three main states:
    * **Normal**
    * **Suspect**
    * **Pathological**
    

    ### Citation
   credit the original authors:
    > Ayres de Campos et al. (2000) *SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms.* J Matern Fetal Med 5:311-318.
    """)


st.warning(
    "⚠️ **Medical Disclaimer:** This tool is designed for educational and predictive modeling purposes only. "
    "The predictions generated are not medical diagnoses. Always consult a qualified healthcare professional "
    "for clinical interpretation, diagnosis, and medical advice."
)

read_acknowledged = st.checkbox(
    "I have read and understand that this model is for informational purposes only and does not replace professional medical consultation."
)

if not read_acknowledged:
    st.info("Please check the box above to proceed to the classification dashboard.")
    st.stop()

st.success("Disclaimer accepted. You can now run your predictions.")

df=pd.read_csv('fetal_health_cleaned.csv')
df.columns=df.columns.str.strip()
st.sidebar.markdown("**Student Name:** Thorn Phyo Wai Yan")
st.sidebar.markdown("**Student ID:** PIUS-20240076")
def load_model():
    with open("fetal_classification_final.pkl", "rb") as f:
        model = pickle.load(f)
    return model

logo_path = "parami.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

data = {
    "Model": ["Logistic Regression"],
    "Accuracy": [85.92],
    "F1-Score": [84.78],
    "Precision": [84.48],
    "Recall": [86.14],
}
comparison_df = pd.DataFrame(data)


df_melted = comparison_df.melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)


st.subheader("Pre Logistic Regression Model Performance Metrics")
st.bar_chart(df_melted, x="Metric", y="Score", color="Metric", stack=False)

with st.expander("View confusion score Table"):
    st.dataframe(comparison_df.set_index("Model"))

st.subheader("Patient Cardiotocography (CTG) Input Form")
st.markdown("Please enter the clinical parameters below or expand sections to adjust details.")

# Grouping inputs inside logical expanders to reduce visual clutter
with st.expander("1. Baseline & Heart Rate Metrics", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.slider("Baseline value", min_value=int(df['baseline value'].min()), max_value=int(df['baseline value'].max()), value=143, step=1)
        var2 = st.slider("Accelerations", min_value=float(df['accelerations'].min()), max_value=float(df['accelerations'].max()), value=0.007, step=0.001)
        var3 = st.slider("Fetal movement", min_value=float(df['fetal_movement'].min()), max_value=float(df['fetal_movement'].max()), value=0.018, step=0.001)
    with col2:
        var4 = st.slider("Uterine contractions", min_value=float(df['uterine_contractions'].min()), max_value=float(df['uterine_contractions'].max()), value=0.000, step=0.001)

with st.expander("2. Decelerations & Variability Analysis"):
    col1, col2 = st.columns(2)
    with col1:
        var5 = st.slider("Light decelerations", min_value=float(df['light_decelerations'].min()), 
                         max_value=float(df['light_decelerations'].max()), value=0.000, step=0.001)
        var6 = st.slider("Severe decelerations", min_value=float(df['severe_decelerations'].min()), 
                         max_value=float(df['severe_decelerations'].max()), value=0.000, step=0.001)
        var7 = st.slider("Prolongued decelerations", min_value=float(df['prolongued_decelerations'].min()), 
                         max_value=float(df['prolongued_decelerations'].max()), value=0.000, step=0.001)
        var8 = st.slider("Abnormal short term variability", min_value=int(df['abnormal_short_term_variability'].min()), 
                         max_value=int(df['abnormal_short_term_variability'].max()), value=44, step=1)
    with col2:
        var9 = st.slider("Mean value of short term variability", min_value=float(df['mean_value_of_short_term_variability'].min()), 
                         max_value=float(df['mean_value_of_short_term_variability'].max()), value=1.300, step=0.001)
        var10 = st.slider("Percentage abnormal long term variability", min_value=float(df['percentage_of_time_with_abnormal_long_term_variability'].min()), 
                          max_value=float(df['percentage_of_time_with_abnormal_long_term_variability'].max()), value=0.000, step=0.001)
        var11 = st.slider("Mean value of long term variability", min_value=float(df['mean_value_of_long_term_variability'].min()), 
                          max_value=float(df['mean_value_of_long_term_variability'].max()), value=11.400, step=0.001)

with st.expander("3. Histogram Metrics"):
    col1, col2 = st.columns(2)
    with col1:
        var12 = st.slider("Histogram width", min_value=int(df['histogram_width'].min()), max_value=int(df['histogram_width'].max()), value=110, step=1)
        var13 = st.slider("Histogram min", min_value=int(df['histogram_min'].min()), max_value=int(df['histogram_min'].max()), step=1)
        var14 = st.slider("Histogram max", min_value=int(df['histogram_max'].min()), max_value=int(df['histogram_max'].max()), step=1)
        var15 = st.slider("Histogram number of peaks", min_value=int(df['histogram_number_of_peaks'].min()), max_value=int(df['histogram_number_of_peaks'].max()), step=1)
        var16 = st.slider("Histogram number of zeroes", min_value=int(df['histogram_number_of_zeroes'].min()), max_value=int(df['histogram_number_of_zeroes'].max()), step=1)
    with col2:
        var17 = st.slider("Histogram mode", min_value=int(df['histogram_mode'].min()), max_value=int(df['histogram_mode'].max()), step=1)
        var18 = st.slider("Histogram mean", min_value=int(df['histogram_mean'].min()), max_value=int(df['histogram_mean'].max()), step=1)
        var19 = st.slider("Histogram median", min_value=int(df['histogram_median'].min()), max_value=int(df['histogram_median'].max()), step=1)
        var20 = st.slider("Histogram variance", min_value=int(df['histogram_variance'].min()), max_value=int(df['histogram_variance'].max()), step=1)
        var21 = st.slider("Histogram tendency", min_value=int(df['histogram_tendency'].min()), max_value=int(df['histogram_tendency'].max()), step=1)
# Prediction execution
st.write("")
predict_button = st.button("Predict Fetal Health", type="primary", use_container_width=True)

if predict_button:
    arr = [[
        var1, var2, var3, var4, var5, var6, var7, var8, var9, var10,
        var11, var12, var13, var14, var15, var16, var17, var18, var19,
        var20, var21
    ]]
    model = load_model()
    result = model.predict(arr)
    fetal_labels = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
    predicted_class = result[0]
    fetal_status = fetal_labels.get(predicted_class, "Unknown")

    if fetal_status == "Normal":
        st.success(f"The predicted fetal health is: **{fetal_status}**")
    elif fetal_status == "Suspect":
        st.warning(f"The predicted fetal health is: **{fetal_status}**")
    else:
        st.error(f"The predicted fetal health is: **{fetal_status}**")
predict_button = st.button("Predict fetal health")



with st.expander("Dataset"):
    st.write("The dataset contains 2,126 records with 21 features and 3 classes (Normal, Suspect, Pathological).")
    st.write("You can explore the dataset below:")
    st.dataframe(df.head(30))

st.subheader(" Cardiotocograms (CTGs)")
st.image("image_fx-20.webp", caption="Cardiotocograms (CTGs)", width=400)

st.image ("ctg.jpg", caption="Cardiotocograms (CTGs)", width=400)