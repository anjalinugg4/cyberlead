import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from dotenv import load_dotenv


load_dotenv()

from huggingface_hub import InferenceClient

hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
# Load trained model
model = joblib.load("models/xgb_model.joblib")


st.set_page_config(page_title="CyberLead Dashboard", layout="wide")
st.title("CyberLead: Smart Sales Pipeline for Cybersecurity Teams")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV of leads", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)

    # Predict conversion probabilities
    scores = model.predict_proba(df)[:, 1]
    df["Conversion Score"] = scores.round(3)

    high_score_thresh = 0.7
    high_threat_thresh = 6


        # --- Threat Risk Score Calculation ---
    df["threat_alerts_norm"] = df["threat_alerts_detected"] / df["threat_alerts_detected"].max()

    industry_risk_weights = {
        "Finance": 1.0,
        "Healthcare": 0.9,
        "Technology": 0.7,
        "Retail": 0.6,
        "Education": 0.5,
        "Other": 0.4,
    }
    df["industry_risk"] = df["industry"].map(industry_risk_weights).fillna(0.4)

    region_impact_weights = {
        "North America": 1.0,
        "Europe": 0.9,
        "Asia": 0.8,
        "South America": 0.7,
        "Other": 0.6,
    }
    df["region_impact"] = df["region"].map(region_impact_weights).fillna(0.6)

    df["Threat Risk Score"] = (
        0.5 * df["threat_alerts_norm"] +
        0.3 * df["industry_risk"] +
        0.2 * df["region_impact"]
    ).round(3)


    # Missed hot leads
    df["Important Miss"] = (
        (df["Conversion Score"] > high_score_thresh) &
        (df["threat_alerts_detected"] > high_threat_thresh) &
        (df["deal_stage"] == "Closed Lost")
    )

    # Hot leads worth pursuing
    df["Hot Lead"] = (
        (df["Conversion Score"] > high_score_thresh) &
        (df["threat_alerts_detected"] > high_threat_thresh) &
        (df["deal_stage"] != "Closed Lost")
    )

        # Actionable Recommendations
    def recommend_action(row):
        if row["Important Miss"]:
            return "Re-Engage"
        elif row["Hot Lead"]:
            return "Follow Up"
        elif row["Conversion Score"] > 0.6 and row["deal_stage"] in ["Proposal", "Qualified"]:
            return "Send Contract"
        elif row["Conversion Score"] > 0.5 and row["deal_stage"] == "Prospect":
            return "Send Proposal"
        elif row["Conversion Score"] < 0.3 or row["deal_stage"] == "Closed Lost":
            return "Ignore"
        return "Review"

    df["Recommendation"] = df.apply(recommend_action, axis=1)


    # Add jitter
    df["x_jittered"] = df["threat_alerts_detected"] + np.random.normal(0, 0.15, size=len(df))
    df["y_jittered"] = df["Conversion Score"] + np.random.normal(0, 0.01, size=len(df))

  # Sidebar filters
    with st.sidebar:
        st.subheader("Filters")

        stage_filter = st.multiselect(
            "Filter by Deal Stage", 
            df["deal_stage"].unique(), 
            default=df["deal_stage"].unique()
        )
        rep_filter = st.multiselect("Filter by Rep", df["rep_assigned"].unique())
        industry_filter = st.multiselect("Filter by Industry", df["industry"].unique())
        region_filter = st.multiselect("Filter by Region", df["region"].unique())
        recommendation_filter = st.multiselect("Filter by Recommendation", df["Recommendation"].unique())
        risk_filter = st.slider("Minimum Threat Risk Score", min_value=0.0, max_value=1.0, value=0.0)
        df = df[df["Threat Risk Score"] >= risk_filter]



        if stage_filter:
            df = df[df["deal_stage"].isin(stage_filter)]
        if rep_filter:
            df = df[df["rep_assigned"].isin(rep_filter)]
        if industry_filter:
            df = df[df["industry"].isin(industry_filter)]
        if region_filter:
            df = df[df["region"].isin(region_filter)]
        if recommendation_filter:
            df = df[df["Recommendation"].isin(recommendation_filter)]


        # Download filtered leads
        st.markdown("### Download Filtered Leads")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_leads.csv",
            mime="text/csv",
        )

    if "show_hot_leads" not in st.session_state:
        st.session_state["show_hot_leads"] = False
    if "show_missed_leads" not in st.session_state:
        st.session_state["show_missed_leads"] = False

    st.subheader("Summary Insights")

    total_leads = len(df)
    converted_leads = df["converted"].sum()
    conversion_rate = round(converted_leads / total_leads * 100, 1) if total_leads > 0 else 0
    hot_leads = df["Hot Lead"].sum()
    missed_opps = df["Important Miss"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", total_leads)
    col2.metric("Conversion Rate", f"{conversion_rate}%")

    if col3.button(f"Hot Leads\n{hot_leads}"):
        st.session_state["show_hot_leads"] = not st.session_state["show_hot_leads"]

    if col4.button(f"Missed Leads\n{missed_opps}"):
        st.session_state["show_missed_leads"] = not st.session_state["show_missed_leads"]


    if st.session_state["show_hot_leads"]:
        st.subheader("Hot Leads Details")
        st.dataframe(
            df[df["Hot Lead"]].drop(columns=["x_jittered", "y_jittered"]).reset_index(drop=True)
        )

    if st.session_state["show_missed_leads"]:
        st.subheader("Missed Hot Leads Details")
        st.dataframe(
            df[df["Important Miss"]].drop(columns=["x_jittered", "y_jittered"]).reset_index(drop=True)
        )



    # Color-coded table
    def color_score(val):
        if val > 0.7:
            return "background-color: lightgreen"
        elif val < 0.3:
            return "background-color: salmon"
        else:
            return "background-color: lightyellow"

    def color_score(val):
        if val > 0.7:
            return "background-color: lightgreen"
        elif val < 0.3:
            return "background-color: salmon"
        else:
            return "background-color: lightyellow"

    def color_risk(val):
        if val > 0.7:
            return "background-color: red"
        elif val > 0.4:
            return "background-color: orange"
        else:
            return "background-color: lightgreen"

    display_df = df.drop(columns=["x_jittered", "y_jittered"])
    st.subheader("Scored Leads with Risk")
    st.dataframe(
        display_df.style
        .applymap(color_score, subset=["Conversion Score"])
        .applymap(color_risk, subset=["Threat Risk Score"])
    )



    # Score Distribution
    st.subheader("Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Conversion Score"], bins=10, kde=True, ax=ax1)
    st.pyplot(fig1)

    # Deal Funnel
    # st.subheader("Deal Stage Funnel")
    # fig2, ax2 = plt.subplots()
    # stage_counts = df["deal_stage"].value_counts().sort_index()
    # ax2.bar(stage_counts.index, stage_counts.values, color="skyblue")
    # ax2.set_ylabel("Number of Leads")
    # ax2.set_xlabel("Deal Stage")
    # ax2.tick_params(axis='x', labelrotation=30, labelsize=9)
    # st.pyplot(fig2)


    st.subheader("Deal Stage Funnel")

    # Dropdown to choose stack option
    stack_option = st.selectbox(
        "Group bars by:",
        options=["None", "Rep", "Industry"],
        index=0,
    )

    # If no grouping, just show total counts per deal stage
    if stack_option == "None":
        stage_counts = df["deal_stage"].value_counts().sort_index()
        funnel_df = stage_counts.reset_index()
        funnel_df.columns = ["Deal Stage", "Count"]

        fig_funnel = px.bar(
            funnel_df,
            x="Deal Stage",
            y="Count",
            text="Count",
            title="Lead Counts by Deal Stage",
            color_discrete_sequence=px.colors.sequential.Blues,
        )
        fig_funnel.update_traces(textposition="outside")
        fig_funnel.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            xaxis_title="Deal Stage",
            yaxis_title="Number of Leads",
        )

    # If grouping selected
    else:
        group_col = "rep_assigned" if stack_option == "Rep" else "industry"
        funnel_df = df.groupby(["deal_stage", group_col]).size().reset_index(name="Count")

        fig_funnel = px.bar(
            funnel_df,
            x="deal_stage",
            y="Count",
            color=group_col,
            barmode="stack",  # Use "group" for side-by-side bars
            title=f"Lead Counts by Deal Stage (Stacked by {stack_option})",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_funnel.update_layout(
            xaxis_title="Deal Stage",
            yaxis_title="Number of Leads",
            plot_bgcolor="white",
        )

    # Show chart
    st.plotly_chart(fig_funnel, use_container_width=True)



    # Scatter Chart


    st.subheader("Threat Alerts vs Conversion Score")

    # Toggle checkboxes to show/hide points
    show_all = st.checkbox("Show All Leads", value=True)
    show_hot = st.checkbox("Highlight Hot Leads", value=True)
    show_missed = st.checkbox("Highlight Missed Hot Leads", value=True)

    # Base plot DataFrame
    plot_df = df.copy()
    plot_df["Lead Type"] = "Other"

    if show_hot:
        plot_df.loc[plot_df["Hot Lead"], "Lead Type"] = "Hot Lead"
    if show_missed:
        plot_df.loc[plot_df["Important Miss"], "Lead Type"] = "Missed Hot Lead"

    # Filter for show_all = False
    if not show_all:
        plot_df = plot_df[plot_df["Lead Type"].isin(["Hot Lead", "Missed Hot Lead"])]

    # Scatter plot
    fig_scatter = px.scatter(
        plot_df,
        x="x_jittered",
        y="y_jittered",
        color="deal_stage",
        symbol="Lead Type",
        symbol_map={
            "Hot Lead": "diamond",
            "Missed Hot Lead": "x",
            "Other": "circle"
        },
        size_max=12,
        opacity=0.6,
        hover_data={
            "company_name": True,
            "Conversion Score": True,
            "Threat Risk Score": True,
            "deal_stage": True,
            "rep_assigned": True,
            "Lead Type": True,
            "x_jittered": False,
            "y_jittered": False,
        },
        title="Threat Alerts vs Conversion Score by Lead Type and Deal Stage",
        labels={"x_jittered": "Threat Alerts (jittered)", "y_jittered": "Conversion Score (jittered)"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    fig_scatter.update_layout(
        legend_title="Deal Stage",
        plot_bgcolor="white",
        height=500,
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
