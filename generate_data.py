import pandas as pd
import numpy as np

np.random.seed(42)
n_students = 200
n_weeks = 8
records = []

for student_id in range(1, n_students + 1):
    true_risk = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.35, 0.25])
    
    base_logins = np.random.randint(8, 20) if true_risk == 'low' else \
                  np.random.randint(4, 12) if true_risk == 'medium' else \
                  np.random.randint(1, 8)
    
    base_attendance = np.random.uniform(0.75, 1.0) if true_risk == 'low' else \
                      np.random.uniform(0.5, 0.8) if true_risk == 'medium' else \
                      np.random.uniform(0.2, 0.6)
    
    base_sentiment = np.random.uniform(0.2, 1.0) if true_risk == 'low' else \
                     np.random.uniform(-0.2, 0.4) if true_risk == 'medium' else \
                     np.random.uniform(-1.0, 0.1)

    for week in range(1, n_weeks + 1):
        decay = (week - 1) * (0.02 if true_risk == 'low' else 0.06 if true_risk == 'medium' else 0.12)
        
        logins = max(0, int(base_logins * (1 - decay) + np.random.randint(-2, 3)))
        attendance = max(0, min(1, base_attendance - decay + np.random.uniform(-0.05, 0.05)))
        sentiment = max(-1, min(1, base_sentiment - decay + np.random.uniform(-0.1, 0.1)))
        delay = max(0, np.random.randint(0, 2) if true_risk == 'low' else \
                       np.random.randint(1, 4) if true_risk == 'medium' else \
                       np.random.randint(2, 7))
        forum_posts = max(0, np.random.randint(2, 8) - int(decay * 5))
        video_completion = max(0, min(1, np.random.uniform(0.6, 1.0) - decay))

        records.append({
            'student_id': student_id,
            'week': week,
            'true_risk': true_risk,
            'lms_logins': logins,
            'attendance_pct': round(attendance, 2),
            'sentiment_score': round(sentiment, 2),
            'submission_delay': delay,
            'forum_posts': forum_posts,
            'video_completion': round(video_completion, 2)
        })

df = pd.DataFrame(records)
df.to_csv('data/raw_data.csv', index=False)
print("Dataset created:", df.shape)
print(df.head())
