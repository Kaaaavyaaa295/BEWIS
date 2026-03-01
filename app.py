import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ast

st.set_page_config(page_title="BEWIS", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .trigger-box { background: #1e2130; padding: 10px; border-radius: 8px; margin: 5px 0; color: white; }
    .header-title { font-size: 36px; font-weight: 900; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/scored_data.csv')
    df['top_triggers'] = df['top_triggers'].apply(ast.literal_eval)
    return df

df = load_data()

st.markdown('<p class="header-title">🧠 BEWIS — Behavioral Early Warning & Intervention System</p>', unsafe_allow_html=True)
st.caption("Predicting student burnout and dropout risk before academic performance declines.")
st.divider()

st.sidebar.title("Student Selector")
student_ids = sorted(df['student_id'].unique())
selected_id = st.sidebar.selectbox("Select Student ID", student_ids)

student_data = df[df['student_id'] == selected_id].sort_values('week')
latest = student_data.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Risk Score", f"{latest['risk_score']}/100")
col2.metric("Dropout Probability", f"{round(latest['dropout_prob']*100, 1)}%")
col3.metric("Behavioral Cluster", latest['cluster_label'])
col4.metric("Risk Level", latest['risk_level'])

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("📈 Risk Evolution Timeline")

    intervention_week = st.slider(
        "🎯 What if counselor intervened at week...",
        min_value=1, max_value=7, value=4
    )

    simulated_scores = []
    for _, row in student_data.iterrows():
        if row['week'] <= intervention_week:
            simulated_scores.append(row['risk_score'])
        else:
            weeks_after = row['week'] - intervention_week
            reduction = min(0.6, weeks_after * 0.15)
            simulated_score = max(5, row['risk_score'] * (1 - reduction))
            simulated_scores.append(round(simulated_score, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=student_data['week'],
        y=student_data['risk_score'],
        mode='lines+markers',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
        name='Actual Risk (No Intervention)'
    ))
    fig.add_trace(go.Scatter(
        x=student_data['week'],
        y=simulated_scores,
        mode='lines+markers',
        line=dict(color='#2ecc71', width=3, dash='dash'),
        marker=dict(size=10),
        name=f'Simulated (Intervened Week {intervention_week})'
    ))
    fig.add_vline(
        x=intervention_week,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Intervention at Week {intervention_week}",
        annotation_font_color="yellow"
    )
    fig.add_hrect(y0=0, y1=30, fillcolor="#2ecc71", opacity=0.08, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="#f39c12", opacity=0.08, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="#e74c3c", opacity=0.08, line_width=0)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Week', color='white', dtick=1),
        yaxis=dict(title='Risk Score', color='white', range=[0, 100]),
        font=dict(color='white'), height=380,
        legend=dict(font=dict(color='white')),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    actual_final = student_data['risk_score'].iloc[-1]
    simulated_final = simulated_scores[-1]
    risk_saved = round(actual_final - simulated_final, 1)

    if risk_saved > 0:
        st.markdown(f"""
        <div style="background:#2ecc7122; border-left: 4px solid #2ecc71;
        padding: 12px; border-radius: 8px; color: white; margin-top: 10px;">
            ✅ <strong>Early intervention at Week {intervention_week} could have reduced
            risk score by {risk_saved} points</strong> — from {actual_final} down to {simulated_final}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Behavioral Metrics Over Time")
    metric_choice = st.selectbox("View metric",
        ['lms_logins', 'attendance_pct', 'sentiment_score',
         'submission_delay', 'behavioral_drift_score'])

    fig2 = px.line(student_data, x='week', y=metric_choice,
                   markers=True, color_discrete_sequence=['#3498db'])
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), height=280,
        xaxis=dict(color='white', dtick=1), yaxis=dict(color='white')
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("🎯 Risk Gauge")
    score = latest['risk_score']
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'color': 'white', 'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': '#e74c3c' if score > 60 else '#f39c12' if score > 30 else '#2ecc71'},
            'steps': [
                {'range': [0, 30], 'color': '#1a3a2a'},
                {'range': [30, 60], 'color': '#3a2e10'},
                {'range': [60, 100], 'color': '#3a1515'}
            ],
            'threshold': {'line': {'color': 'white', 'width': 4}, 'value': score}
        },
        number={'font': {'color': 'white', 'size': 40}}
    ))
    gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'}, height=280
    )
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("🔍 Key Risk Triggers")
    for trigger in latest['top_triggers']:
        st.markdown(f'<div class="trigger-box">⚠️ {trigger}</div>', unsafe_allow_html=True)

    st.subheader("💡 Recommended Intervention")
    level = latest['risk_level']
    color = "#e74c3c" if level == "High" else "#f39c12" if level == "Medium" else "#2ecc71"
    st.markdown(f"""
    <div style="background:{color}22; border-left: 4px solid {color};
    padding: 15px; border-radius: 8px; color: white;">
        <strong>{level} Risk Action:</strong><br>{latest['intervention']}
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("🏫 Cohort Risk Overview")
latest_week = df[df['week'] == df['week'].max()]

col1, col2, col3 = st.columns(3)
col1.metric("🔴 High Risk Students", len(latest_week[latest_week['risk_level'] == 'High']))
col2.metric("🟡 Medium Risk Students", len(latest_week[latest_week['risk_level'] == 'Medium']))
col3.metric("🟢 Low Risk Students", len(latest_week[latest_week['risk_level'] == 'Low']))

fig3 = px.histogram(latest_week, x='risk_score', color='risk_level',
                    color_discrete_map={'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'},
                    nbins=20, title="Risk Score Distribution — Full Cohort")
fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                   font=dict(color='white'), height=300)
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.pie(latest_week, names='cluster_label', title="Behavioral Cluster Distribution",
              color_discrete_sequence=px.colors.qualitative.Set2)
fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300)
st.plotly_chart(fig4, use_container_width=True)

st.caption("BEWIS — Behavioral Early Warning & Intervention System | Built for student success")