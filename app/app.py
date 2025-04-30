from flask import *
import numpy as np
import pandas as pd
import json
import joblib
import tensorflow as tf

class_name={0:'BENIGN',7: 'FTP-Patator',11: 'SSH-Patator',1: 'Bot',10: 'PortScan'
 ,12:'Web Attack � Brute Force',14: 'Web Attack � XSS',13:
 'Web Attack � Sql Injection',6: 'DoS slowloris',5:'DoS Slowhttptest'
 ,4:'DoS Hulk',3: 'DoS GoldenEye',8: 'Heartbleed',2: 'DDoS',9: 'Infiltration'}

feature_mapping = {
   "Dst Port":" Destination Port",
    "Flow Duration":" Flow Duration",
    "Total Fwd Packet":" Total Fwd Packets",
    "Total Bwd packets":" Total Backward Packets",
    "Total Length of Fwd Packet":"Total Length of Fwd Packets",
    "Total Length of Bwd Packet":" Total Length of Bwd Packets",
    "Fwd Packet Length Max":" Fwd Packet Length Max",
    "Fwd Packet Length Min":" Fwd Packet Length Min",
    "Fwd Packet Length Mean":" Fwd Packet Length Mean",
    "Fwd Packet Length Std":" Fwd Packet Length Std",
    "Bwd Packet Length Max":"Bwd Packet Length Max",
    "Bwd Packet Length Min":" Bwd Packet Length Min",
    "Bwd Packet Length Mean":" Bwd Packet Length Mean",
    "Bwd Packet Length Std":" Bwd Packet Length Std",
    "Flow Bytes/s":"Flow Bytes/s",
    "Flow Packets/s":" Flow Packets/s",
    "Flow IAT Mean":" Flow IAT Mean",
    "Flow IAT Std":" Flow IAT Std",
    "Flow IAT Max":" Flow IAT Max",
    "Flow IAT Min":" Flow IAT Min",
    "Fwd IAT Total":"Fwd IAT Total",
    "Fwd IAT Mean":" Fwd IAT Mean",
    "Fwd IAT Std":" Fwd IAT Std",
    "Fwd IAT Max":" Fwd IAT Max",
    "Fwd IAT Min":" Fwd IAT Min",
    "Bwd IAT Total":"Bwd IAT Total",
    "Bwd IAT Mean":" Bwd IAT Mean",
    "Bwd IAT Std":" Bwd IAT Std",
    "Bwd IAT Max":" Bwd IAT Max",
    "Bwd IAT Min":" Bwd IAT Min",
    "Fwd PSH Flags":"Fwd PSH Flags",
    "Bwd PSH Flags":" Bwd PSH Flags",
    "Fwd URG Flags":" Fwd URG Flags",
    "Bwd URG Flags":" Bwd URG Flags",
    "Fwd Header Length":" Fwd Header Length",
    "Bwd Header Length":" Bwd Header Length",
    "Fwd Packets/s":"Fwd Packets/s",
    "Bwd Packets/s":" Bwd Packets/s",
    "Packet Length Min":" Min Packet Length",
    "Packet Length Max":" Max Packet Length",
    "Packet Length Mean":" Packet Length Mean",
    "Packet Length Std":" Packet Length Std",
    "Packet Length Variance":" Packet Length Variance",
    "FIN Flag Count":"FIN Flag Count",
    "SYN Flag Count":" SYN Flag Count",
    "RST Flag Count":" RST Flag Count",
    "PSH Flag Count":" PSH Flag Count",
    "ACK Flag Count":" ACK Flag Count",
    "URG Flag Count":" URG Flag Count",
    "CWR Flag Count":" CWE Flag Count",
    "ECE Flag Count":" ECE Flag Count",
    "Down/Up Ratio":" Down/Up Ratio",
    "Average Packet Size":" Average Packet Size",
    "Fwd Segment Size Avg":" Avg Fwd Segment Size",
    "Bwd Segment Size Avg":" Avg Bwd Segment Size",
    "Fwd Header Length.1":" Fwd Header Length.1",
    "Fwd Bytes/Bulk Avg":"Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg":" Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg":" Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg":" Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg":" Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg":"Bwd Avg Bulk Rate",
    "Subflow Fwd Packets":"Subflow Fwd Packets",
    "Subflow Fwd Bytes":" Subflow Fwd Bytes",
    "Subflow Bwd Packets":" Subflow Bwd Packets",
    "Subflow Bwd Bytes":" Subflow Bwd Bytes",
    "FWD Init Win Bytes":"Init_Win_bytes_forward",
    "Bwd Init Win Bytes":" Init_Win_bytes_backward",
    "Fwd Act Data Pkts":" act_data_pkt_fwd",
    "Fwd Seg Size Min":" min_seg_size_forward",
    "Active Mean":"Active Mean",
    "Active Std":" Active Std",
    "Active Max":" Active Max",
    "Active Min":" Active Min",
    "Idle Mean":"Idle Mean",
    "Idle Std":" Idle Std",
    "Idle Max":" Idle Max",
    "Idle Min":" Idle Min"}

app=Flask('ml-dl-hybrid-model')

clf_xgb = joblib.load('../models/xgb_model.pkl')
lstm_model = tf.keras.models.load_model('../models/best_lstm_model.h5')

selector_mi = joblib.load('../models/selector_mi.pkl')
scaler_pca = joblib.load('../models/scaler_pca.pkl')
pca = joblib.load('../models/pca.pkl')
scaler = joblib.load('../models/scaler.pkl')

with open('features.json', 'r') as f:
    selected_features = json.load(f)['selected_features']

def preprocess_input(df):
    df_selected=df[selected_features]
    arr = df_selected.values

    features_ml = arr

    X_mi=selector_mi.transform(features_ml)
    X_mi_scaled = scaler_pca.transform(X_mi)
    X_pca = pca.transform(X_mi_scaled)
    X_final = scaler.transform(X_pca)

    features_lstm = X_final.reshape(X_final.shape[0],1,X_final.shape[1])
    return X_final,features_lstm


@app.post('/predict')
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame(data)
        print(features)
        print(features.shape)

        features.rename(columns=feature_mapping, inplace=True)
        X_final,features_lstm = preprocess_input(features)

        xgb_prediction = clf_xgb.predict(X_final)
        xgb_probs = clf_xgb.predict_proba(X_final)

        threshold = 0.7
        max_prob = np.max(xgb_probs)

        print(X_final)
        if max_prob < threshold:
            prediction = lstm_model.predict(features_lstm)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            model_used="LSTM"
            return jsonify({
                "prediction":prediction,
                "predicted_class":predicted_class,
                "class_name":class_name[predicted_class],
                "model_used":model_used,
                #"confidence":float(max_prob),
            })

        else:
            predicted_class = int(np.argmax(xgb_probs,axis=1)[0])
            model_used = "XGB"
            return jsonify({
                "prediction":xgb_prediction.tolist(),
                "predicted_class":predicted_class,
                "class_name":class_name[predicted_class],
                "model_used":model_used,
                "confidence":float(max_prob),
            })
    except Exception as e:
        return jsonify({"error":str(e)})


if __name__=="__main__":
    app.run(debug=False,host='0.0.0.0',port=5000)
