import requests
import pandas as pd
import json

test_data = [{
    " Destination Port": 80,
    " Flow Duration": 12345,
    "Total Length of Fwd Packets": 987,
    " Total Length of Bwd Packets": 543,
    " Fwd Packet Length Max": 1500,
    "Bwd Packet Length Max": 1200,
    " Bwd Packet Length Mean": 600,
    "Flow Bytes/s": 1.2e5,
    " Flow IAT Max": 3456,
    " Fwd IAT Max": 789,
    " Max Packet Length": 1500,
    " Packet Length Mean": 900,
    " Packet Length Std": 300,
    " Packet Length Variance": 90000,
    " Average Packet Size": 1000,
    " Avg Bwd Segment Size": 800,
    " Subflow Fwd Bytes": 2345,
    " Subflow Bwd Bytes": 1234,
    "Init_Win_bytes_forward": 5840,
    " Init_Win_bytes_backward": 2920
}]

response = requests.post("http://127.0.0.1:5000/predict", json=test_data)
print("Response:", response.json())
