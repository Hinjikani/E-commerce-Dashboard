# E-Commerce Dashboard

## First Step - Setup Virtual Environment

### Setup Environment - Shell/Terminal
```
# Make sure you are in the project's directory
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Setup Environment - MacOS/Linux
```
# Make sure you are in the project's directory
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Setup Environment - Conda
```
# Make sure you are in the project's directory
conda create --name dashboard-environment python=3.11.99
conda activate dashboard-environment
pip install -r requirements.txt
```

## Second Step - Run Streamlit Dashboard
```
cd dashboard
streamlit run dashboard.py
```

## Dashboard can be Seen on this Link
https://e-commerce-dashboard-pascal.streamlit.app/
