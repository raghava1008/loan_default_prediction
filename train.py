import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


print('COMMENCED')
# Step 1: Load Data
df = pd.read_csv("accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv", low_memory=False)

print('DF LOADED') 

# Step 2: Create Binary Target
default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
df = df[df['loan_status'].notna()]
df['defaulted'] = df['loan_status'].isin(default_statuses).astype(int)

# Step 3: Feature Selection
features = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose',
    'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'application_type'
]

print('FEATURE SELECTED')
df_model = df[features + ['defaulted']].copy()

# Step 4: Clean emp_length
def clean_emp_length(val):
    if pd.isna(val): return np.nan
    val = str(val).strip().lower()
    if '<' in val: return 0
    if '10+' in val: return 10
    if 'n/a' in val: return np.nan
    digits = ''.join([c for c in val if c.isdigit()])
    return int(digits) if digits else np.nan

df_model['emp_length'] = df_model['emp_length'].apply(clean_emp_length)

print('MODEL CLEANED')

# Step 5: Encode categorical variables
categorical_cols = ['term', 'grade', 'home_ownership', 'verification_status', 'purpose', 'application_type']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Step 6: Drop missing values
df_model.dropna(inplace=True)

# Step 7: Split and Train
X = df_model.drop('defaulted', axis=1)
y = df_model['defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Step 8: Evaluation
y_pred = clf.predict(X_test)
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Save model
with open('loan_default_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved as 'loan_default_model.pkl'")
