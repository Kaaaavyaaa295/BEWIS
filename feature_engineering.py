import pandas as pd
import numpy as np

df = pd.read_csv('data/raw_data.csv')

engineered = []

for sid in df['student_id'].unique():
    student = df[df['student_id'] == sid].sort_values('week')
    
    for i, row in student.iterrows():
        week = row['week']
        history = student[student['week'] <= week]
        
        if len(history) >= 2:
            login_slope = np.polyfit(history['week'], history['lms_logins'], 1)[0]
        else:
            login_slope = 0
        
        week1_attendance = student[student['week'] == 1]['attendance_pct'].values[0]
        attendance_drop = week1_attendance - row['attendance_pct']
        
        delay_variance = history['submission_delay'].var() if len(history) > 1 else 0
        
        week1_sentiment = student[student['week'] == 1]['sentiment_score'].values[0]
        sentiment_decline = week1_sentiment - row['sentiment_score']
        
        mean_logins = history['lms_logins'].mean()
        engagement_volatility = history['lms_logins'].std() / (mean_logins + 1e-5)
        
        if week >= 2:
            prev_logins = student[student['week'] == week-1]['lms_logins'].values[0]
            sudden_drop = 1 if (prev_logins > 0 and (row['lms_logins'] / (prev_logins + 1e-5)) < 0.5) else 0
        else:
            sudden_drop = 0
        
        drift_score = (0.4 * max(0, -login_slope/10) + 
                      0.35 * max(0, attendance_drop) + 
                      0.25 * max(0, sentiment_decline/2))
        
        engineered.append({
            **row.to_dict(),
            'login_slope': round(login_slope, 3),
            'attendance_drop': round(attendance_drop, 3),
            'delay_variance': round(delay_variance, 3),
            'sentiment_decline': round(sentiment_decline, 3),
            'engagement_volatility': round(engagement_volatility, 3),
            'sudden_drop': sudden_drop,
            'behavioral_drift_score': round(drift_score, 3)
        })

eng_df = pd.DataFrame(engineered)

eng_df['risk_label'] = eng_df['true_risk'].map({'low': 0, 'medium': 1, 'high': 2})

eng_df['dropout_prob'] = eng_df.apply(lambda r: 
    round(min(1.0, max(0.0, 
        0.3 * r['attendance_drop'] + 
        0.3 * r['behavioral_drift_score'] + 
        0.2 * (r['submission_delay']/7) + 
        0.2 * max(0, r['sentiment_decline']/2)
    )), 2), axis=1)

eng_df.to_csv('data/engineered_data.csv', index=False)
print("Feature engineering done:", eng_df.shape)
print(eng_df[['student_id','week','behavioral_drift_score','dropout_prob','risk_label']].head(10))
