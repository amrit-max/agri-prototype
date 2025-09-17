# Agri Prototype

This project demonstrates a mini-version of an AI-powered agriculture monitoring system using hyperspectral data and sensor fusion.

## Structure
- data/: Place your datasets here (Indian Pines, sensor_data.csv)
- notebooks/: Jupyter notebook for training and visualizing
- app/: Backend (FastAPI) and Frontend (Streamlit)
- models/: Saved ML models

## Usage
1. Run `notebooks/hyperspectral_demo.ipynb` to preprocess data and train model.
2. Start backend: `uvicorn app.backend:app --reload`
3. Start dashboard: `streamlit run app/dashboard.py`
