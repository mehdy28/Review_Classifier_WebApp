import streamlit as st
import pandas as pd
from joblib import load
from lime import lime_text
import re
import os

# Create a component folder if it doesn't exist
component_folder = "components"
if not os.path.exists(component_folder):
    os.makedirs(component_folder)

@st.cache_resource
def lod_vect_and_model():
    vectorizer_path = os.path.join(component_folder, 'vectorizer.joblib')
    classifier_path = os.path.join(component_folder, 'classifier.joblib')
    text_vectoriser = load(vectorizer_path)
    classif = load(classifier_path)

    return text_vectoriser, classif

text_vectoriser, classif = lod_vect_and_model()

def vectorize_text(texts):
    text_transformed = text_vectoriser.transform(texts)
    return text_transformed

def pred_class(texts):
    return classif.predict(vectorize_text(texts))

def pred_probs(texts):
    return classif.predict_proba(vectorize_text(texts))

def create_colored_review(review, word_contributions):
    tokens = re.findall(text_vectoriser.token_pattern, review)
    modified_review = ""
    for token in tokens:
        if token in word_contributions["Word"].values:
            idx = word_contributions["Word"].values.tolist().index(token)
            contribution = word_contributions.iloc[idx]["Contribution"]
            modified_review += ":green[{}]".format(token) if contribution > 0 else ":red[{}]".format(token)
            modified_review += " "
        else:
            modified_review += token
            modified_review += " "
    return modified_review

explainer = lime_text.LimeTextExplainer(class_names=classif.classes_)

st.title("Review Classification :green[Positive] vs :red[Negative]")
review = st.text_area(label="Enter Review Here: ", value="Enjoy", height=28)

submit = st.button("Classify")

if submit and review:
    col1, col2 = st.columns(2, gap="medium")

    prediction, probs = pred_class([review]), pred_probs([review])
    prediction, probs = prediction[0], probs[0]
    with col1:
        st.markdown('### Prediction: {}'.format(prediction))
        st.metric(label="Confidence", value="{:.2f}".format(probs[1] * 100 if prediction == "positive" else probs[0] * 100))

        explanation = explainer.explain_instance(review, classifier_fn=pred_probs, num_features=50)
        word_contribution = pd.DataFrame(explanation.as_list(), columns=["Word", "Contribution"])
        modified_review = create_colored_review(review, word_contribution)
        st.write(modified_review)
    with col2:
        fig = explanation.as_pyplot_figure()
        fig.set_figheight(12)
        st.pyplot(fig, use_container_width=True)
