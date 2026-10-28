import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Readmission Risk Predictor', page_icon='🩸', layout='centered')

THRESHOLD = 0.40
FOLDER = os.path.dirname(os.path.abspath(__file__))

CSS = '''
<style>
.stApp {
    background-color: #f0f7f6;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64'%3E%3Cpath d='M32 25v14M25 32h14' stroke='%230f766e' stroke-opacity='0.13' stroke-width='3' stroke-linecap='round' fill='none'/%3E%3C/svg%3E");
}
header[data-testid="stHeader"] { background: transparent; }
.block-container {
    background: #ffffff;
    border-radius: 20px;
    border-top: 6px solid #0f766e;
    padding: 2rem 2rem 1.5rem 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 8px 30px rgba(15, 118, 110, 0.12);
}
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #064e3b 0%, #0f766e 55%, #0e7490 100%);
    padding: 1.8rem 2rem 3.6rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { color: white; margin: 0; font-size: 1.9rem; }
.hero p { color: #ccfbf1; margin: 0.4rem 0 0.9rem 0; font-size: 1rem; }
.chip {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    margin-right: 0.4rem;
    font-size: 0.8rem;
}
.ecg { position: absolute; left: 0; bottom: 0; width: 100%; height: 48px; }
.ecg path {
    fill: none;
    stroke: #99f6e4;
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 700;
    stroke-dashoffset: 700;
    animation: beat 3.5s linear infinite;
}
@keyframes beat {
    0% { stroke-dashoffset: 700; }
    70% { stroke-dashoffset: 0; }
    100% { stroke-dashoffset: 0; opacity: 0; }
}
.result-card {
    border-radius: 16px;
    padding: 1.4rem;
    text-align: center;
    margin-top: 1rem;
    border: 2px solid;
}
.result-title { font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; }
.result-text { font-size: 1rem; color: #334155; }
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0f766e, #0e7490);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 0;
    font-size: 1.1rem;
    font-weight: 600;
}
div.stButton > button:hover { opacity: 0.9; color: white; border: none; }
.small-note { text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 1rem; }
</style>
'''

HERO = '''
<div class="hero">
<h1>🩸 Diabetes Readmission Risk</h1>
<p>Estimate how likely a diabetic patient is to be readmitted to hospital after discharge.</p>
<span class="chip">130 US hospitals</span><span class="chip">1999 to 2008</span><span class="chip">70,000 patients</span>
<svg class="ecg" viewBox="0 0 600 48" preserveAspectRatio="none">
<path d="M0 30 L120 30 L140 30 L152 8 L166 46 L180 18 L192 30 L330 30 L350 30 L362 8 L376 46 L390 18 L402 30 L600 30"/>
</svg>
</div>
'''

LABELS = {
    'race': 'Race',
    'gender': 'Gender',
    'age': 'Age group',
    'admission_type': 'Admission type',
    'discharge_disposition': 'Discharged to',
    'admission_source': 'Admission source',
    'time_in_hospital': 'Days in hospital',
    'num_lab_procedures': 'Lab tests done',
    'num_procedures': 'Procedures done',
    'num_medications': 'Medications given',
    'number_outpatient': 'Outpatient visits (past year)',
    'number_emergency': 'Emergency visits (past year)',
    'number_inpatient': 'Inpatient stays (past year)',
    'diag_1': 'Primary diagnosis',
    'diag_2': 'Secondary diagnosis',
    'diag_3': 'Additional diagnosis',
    'number_diagnoses': 'Total diagnoses',
    'A1Cresult': 'A1C result',
    'metformin': 'Metformin',
    'glipizide': 'Glipizide',
    'pioglitazone': 'Pioglitazone',
    'rosiglitazone': 'Rosiglitazone',
    'insulin': 'Insulin',
    'change': 'Medication changed',
    'diabetesMed': 'Diabetes medication prescribed',
}

TABS = {
    '👤 Patient': ['race', 'gender', 'age'],
    '🏥 Admission': ['admission_type', 'admission_source', 'discharge_disposition'],
    '🛏️ Hospital stay': ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                         'number_outpatient', 'number_emergency', 'number_inpatient'],
    '🩺 Diagnoses': ['diag_1', 'diag_2', 'diag_3', 'number_diagnoses'],
    '💊 Medications': ['A1Cresult', 'diabetesMed', 'change', 'insulin', 'metformin',
                       'glipizide', 'pioglitazone', 'rosiglitazone'],
}


def load_pickle(name):
    with open(os.path.join(FOLDER, name), 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_files():
    model = load_pickle('model.pkl')
    encoders = load_pickle('encoders.pkl')
    columns = load_pickle('columns.pkl')
    info = load_pickle('num_info.pkl')
    return model, encoders, columns, info


def gauge(prob, color):
    arc = 251.3 * prob
    return f'''<svg viewBox="0 0 200 120" style="width:260px;">
<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e2e8f0" stroke-width="16" stroke-linecap="round"/>
<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{arc} 251.3"/>
<text x="100" y="92" text-anchor="middle" font-size="30" font-weight="700" fill="{color}">{prob * 100:.0f}%</text>
</svg>'''


model, encoders, columns, info = load_files()
num_info = info['num_info']
age_values = info['age_values']

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

inputs = {}


def make_input(col, box):
    label = LABELS.get(col, col)
    if col in encoders:
        inputs[col] = box.selectbox(label, list(encoders[col].classes_))
    elif col == 'age':
        inputs[col] = box.selectbox(label, age_values, format_func=lambda a: f'{a - 5} to {a + 5} years')
    else:
        low, high, mid = num_info[col]
        inputs[col] = box.number_input(label, min_value=low, max_value=high, value=mid, step=1)


tabs = st.tabs(list(TABS.keys()))
for tab, names in zip(tabs, TABS.values()):
    with tab:
        left, right = st.columns(2)
        for i, col in enumerate(names):
            if col in columns:
                make_input(col, left if i % 2 == 0 else right)

st.write('')

if st.button('Predict readmission risk'):
    row = pd.DataFrame([inputs])
    for col in encoders:
        row[col] = encoders[col].transform(row[col])
    row = row[columns]

    prob = float(model.predict_proba(row)[0, 1])

    if prob >= THRESHOLD:
        color, bg, icon, title = '#dc2626', '#fef2f2', '⚠️', 'High risk'
        text = 'This patient is likely to be readmitted. Consider extra follow-up care after discharge.'
    else:
        color, bg, icon, title = '#16a34a', '#f0fdf4', '✅', 'Lower risk'
        text = 'This patient is less likely to be readmitted. Standard follow-up should be enough.'

    st.markdown(f'''<div class="result-card" style="background:{bg}; border-color:{color};">
{gauge(prob, color)}
<div class="result-title" style="color:{color};">{icon} {title}</div>
<div class="result-text">{text}</div>
</div>''', unsafe_allow_html=True)

st.markdown(
    '<div class="small-note">Screening aid only, not a diagnosis. Model ROC-AUC is 0.65 and the high-risk cutoff is 40%.</div>',
    unsafe_allow_html=True)