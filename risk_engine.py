import pandas as pd
import numpy as np

def calculate_risk_score(row):
    engagement_score = max(0, -row['login_slope'] / 5)
    attendance_score = max(0, row['attendance_drop'])
    delay_score = min(1, row['submission_delay'] / 7)
    sentiment_score = max(0, row['sentiment_decline'] / 2)
    drift_score = row['behavioral_drift_score']

    raw = (0.30 * engagement_score +
           0.25 * attendance_score +
           0.20 * delay_score +
           0.15 * sentiment_score +
           0.10 * drift_score)

    return round(min(100, raw * 100), 1)

def get_risk_level(score):
    if score <= 30:
        return "Low", "#2ecc71"
    elif score <= 60:
        return "Medium", "#f39c12"
    else:
        return "High", "#e74c3c"

def get_top_triggers(row):
    triggers = {
        f"LMS login decline (slope: {row['login_slope']})": max(0, -row['login_slope']),
        f"Attendance drop: {round(row['attendance_drop']*100, 1)}%": max(0, row['attendance_drop']),
        f"Submission delay: {row['submission_delay']} days late": min(1, row['submission_delay']/7),
        f"Sentiment decline: {round(row['sentiment_decline'], 2)}": max(0, row['sentiment_decline']/2),
        f"Behavioral drift score: {row['behavioral_drift_score']}": row['behavioral_drift_score']
    }
    sorted_triggers = sorted(triggers.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_triggers[:3]]

def get_intervention(risk_level, cluster_label):
    base = {
        "Low": "Send a gentle engagement reminder email.",
        "Medium": "Schedule a peer mentor check-in within 3 days.",
        "High": "Immediate counselor alert + academic advisor flag required."
    }
    cluster_addition = {
        "Gradual Burnout": "Suggest workload management workshop.",
        "Sudden Drop-off": "Urgent welfare check — sudden behavioral change detected.",
        "Emotional Distress": "Priority counseling referral — emotional indicators elevated.",
        "Chronic Disengagement": "Consider motivational intervention program."
    }
    return base.get(risk_level, "") + " " + cluster_addition.get(cluster_label, "")

df = pd.read_csv('data/final_data.csv')
df['risk_score'] = df.apply(calculate_risk_score, axis=1)
df['risk_level'] = df['risk_score'].apply(lambda s: get_risk_level(s)[0])
df['top_triggers'] = df.apply(lambda r: get_top_triggers(r), axis=1)
df['intervention'] = df.apply(lambda r: get_intervention(r['risk_level'], r['cluster_label']), axis=1)

df.to_csv('data/scored_data.csv', index=False)
print("Risk scoring done.")
print(df[['student_id','week','risk_score','risk_level','cluster_label']].head(16))
