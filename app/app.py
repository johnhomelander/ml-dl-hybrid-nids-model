from flask import *
import numpy as np
import pandas as pd
import json
import joblib
import tensorflow as tf

app=Flask('ml-dl-hybrid-model')

clf_xgb = joblib.load('../xgb_model.pkl')
lstm_model = tf.keras.models.load_model('../best_lstm_model.h5')

selector_mi = joblib.load('../selector_mi.pkl')
scaler_pca = joblib.load('../scaler_pca.pkl')
pca = joblib.load('../pca.pkl')
scaler = joblib.load('../scaler.pkl')

with open('selected_features.json', 'r') as f:
    selected_features = json.load(f)

features = selected_features['selected_features']

def preprocess_input(df):
    df_selected=df[selected_features]

    features_ml = df.reshape(1,-1)
    features_lstm = df.reshape(1,1,-1)

    X_mi=selector_mi.transform(features_ml)
    X_mi_scaled = scaler_pca.transform(X_mi)
    X_pca = pca.transform(X_mi_scaled)
    X_final = scaler_final.transform(X_pca)

    return X_final


@app.post('/predict')
def predict():
    data = request.get_json()
    features = pd.DataFrame(data)

    X_final = preprocess_input(features)

    dmatrix = xgb.DMatrix(X_final)
    xgb_prediction = clf_xgb.predict(dmatrix)
    xgb_probs = clf_xgb.predict_proba(X_final)

    threshold = 0.5
    max_prob = np.max(xgb_probs)

    if max_prob < threshold:
        prediction = lstm_model.predict(features_lstm)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        model_used="LSTM"

    else:
        predicted_class = int(np.argmax(xgb_probs,axis=1)[0])
        model_used = "XGB"


    return jsonify({
        "prediction":xgb_prediction,
        "predicted_class":predicted_class,
        "model_used":model_used,
        "confidence":float(max_prob),
    })

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
