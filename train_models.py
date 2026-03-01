import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('data/engineered_data.csv')

FEATURES = ['lms_logins', 'attendance_pct', 'sentiment_score', 'submission_delay',
            'forum_posts', 'video_completion', 'login_slope', 'attendance_drop',
            'delay_variance', 'sentiment_decline', 'engagement_volatility',
            'sudden_drop', 'behavioral_drift_score']

X = df[FEATURES].fillna(0)
y_risk = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)

# MODEL 1: Burnout Risk Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print("=== Burnout Risk Classifier ===")
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print(classification_report(y_test, preds, target_names=['Low','Medium','High']))

# MODEL 2: Dropout Probability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegression(max_iter=1000, random_state=42)
y_dropout = (df['dropout_prob'] > 0.3).astype(int)  # fixed threshold
print(f"Dropout class distribution: {y_dropout.value_counts().to_dict()}")
lr.fit(X_scaled, y_dropout)
print("=== Dropout Classifier ===")
print(f"Accuracy: {accuracy_score(y_dropout, lr.predict(X_scaled)):.2%}")

# MODEL 3: Behavioral Clustering
cluster_features = ['behavioral_drift_score', 'engagement_volatility',
                    'sentiment_decline', 'attendance_drop', 'sudden_drop']
X_cluster = StandardScaler().fit_transform(df[cluster_features].fillna(0))
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster)

cluster_profiles = df.groupby('cluster')[['behavioral_drift_score','sentiment_decline',
                                           'sudden_drop','attendance_drop']].mean()
print("\n=== Cluster Profiles ===")
print(cluster_profiles)

cluster_names = {
    cluster_profiles['behavioral_drift_score'].idxmax(): 'Gradual Burnout',
    cluster_profiles['sudden_drop'].idxmax(): 'Sudden Drop-off',
    cluster_profiles['sentiment_decline'].idxmax(): 'Emotional Distress',
}
for i in range(4):
    if i not in cluster_names:
        cluster_names[i] = 'Chronic Disengagement'

df['cluster_label'] = df['cluster'].map(cluster_names)
df.to_csv('data/final_data.csv', index=False)

pickle.dump(rf, open('models/burnout_classifier.pkl', 'wb'))
pickle.dump(lr, open('models/dropout_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(kmeans, open('models/kmeans.pkl', 'wb'))
pickle.dump(cluster_names, open('models/cluster_names.pkl', 'wb'))
print("\nAll models saved.")
