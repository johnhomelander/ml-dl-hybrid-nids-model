import pandas as pd
import numpy as np
import json,requests
import time

df=pd.read_csv('../testing/2025-04-08_Flow.csv')
columns_to_remove = ['Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Protocol','Src Port','Label']
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
url='http://127.0.0.1:5000/predict'

#print(df['Fwd Header Length'])
df['Fwd Header Length.1'] = df['Fwd Header Length']
print(df[['Fwd Header Length','Fwd Header Length.1']])
print(df.columns,len(df.columns))

batch_size = 100
for idx, row in df.iterrows():
    data = row.to_frame().T.to_dict(orient='records')  # One record per row

#for i in range(0, len(df), batch_size):
#    batch = df.iloc[i:i + batch_size]
#    data = batch.to_dict(orient='records')
    #print(data)
    try:
        response = requests.post(url, json=data)
        result = response.json()
        print(result)


        # Optional delay to mimic real-time streaming
        #time.sleep(0.5)

    except Exception as e:
        print(f"Error with record #{idx + 1}: {e}")
#for i in range(1909):
#    print(dts[i])
#    payload = {
