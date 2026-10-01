@echo off
python src\build_graph.py
python src\build_features.py
python src\detect_anomalies.py
streamlit run app\dashboard.py
