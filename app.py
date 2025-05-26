import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import joblib

# Set the page config to wide mode
st.set_page_config(layout="wide")

# Load data and preprocess
@st.cache_data
def load_data():
    olympic_data = pd.read_csv('Summer_Olympic_medals.csv') 
    medal_counts = olympic_data.groupby(['Country', 'Year']).size().reset_index(name='Total_Medals')
    sport_count = olympic_data.groupby(['Country', 'Year'])['Sport'].nunique().reset_index(name='Sport_Count')
    medal_types = olympic_data.pivot_table(index=['Country', 'Year'], columns='Medal',
                                           aggfunc='size', fill_value=0).reset_index()
    medal_types.columns.name = None
    medal_types.rename(columns={'Gold': 'Gold_Count', 'Silver': 'Silver_Count', 'Bronze': 'Bronze_Count'}, inplace=True)
    medal_counts = medal_counts.merge(sport_count, on=['Country', 'Year'])
    medal_counts = medal_counts.merge(medal_types, on=['Country', 'Year'], how='left')
    medal_counts.fillna(0, inplace=True)
    return olympic_data, medal_counts

@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_medal_predictor.pkl')
    le_country = joblib.load('label_encoder_country.pkl')
    return rf_model, le_country

olympic_data, medal_counts = load_data()
rf_model, le_country = load_models()

st.title("📊 Olympics Data Analysis & Medal Prediction")

countries = sorted(medal_counts['Country'].dropna().astype(str).unique())

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Option:", ["Exploratory Data Analysis", "Medal Prediction"])

def st_pyplot(fig):
    plt.tight_layout()
    st.pyplot(fig)

def two_col_section(title, info_text, fig_left, fig_right):
    st.markdown(f"### {title}")
    st.markdown(info_text)
    col1, col2 = st.columns(2)
    with col1:
        st_pyplot(fig_left)
    with col2:
        st_pyplot(fig_right)

if option == "Exploratory Data Analysis":

    # Set a consistent figure size for all plots
    fig_size = (18, 9)

    # Section 1: Top Countries & Medals Over Time
    top_25 = medal_counts.groupby('Country')['Total_Medals'].sum().reset_index().sort_values('Total_Medals', ascending=False).head(25)
    fig1, ax1 = plt.subplots(figsize=fig_size)
    sns.barplot(y=top_25['Country'], x=top_25['Total_Medals'], palette='viridis', ax=ax1)
    ax1.set_title('Top 25 Countries by Total Olympic Medals')
    ax1.set_xlabel('Number of Medals')
    ax1.set_ylabel('Country')
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.0f', label_type='edge')

    medals_by_year = medal_counts.groupby('Year')['Total_Medals'].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=fig_size)
    sns.lineplot(data=medals_by_year, x='Year', y='Total_Medals', marker='o', linewidth=2.5, ax=ax2)
    ax2.set_title('Total Medals Awarded Over the Years')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Medals')
    ax2.set_xticks(medals_by_year['Year'])
    ax2.set_xticklabels(medals_by_year['Year'], rotation=45)
    for x, y in zip(medals_by_year['Year'], medals_by_year['Total_Medals']):
        ax2.text(x, y + 5, str(y), ha='center')

    info1 = """
    - United States leads with the highest Olympic medals.  
    - Total medals awarded have generally increased over time except slight dip in 2004.
    """
    two_col_section("🏆 Top 25 Countries & Medals Over Time", info1, fig1, fig2)

    # Section 2: Sports & Gender Distribution
    top_sports = olympic_data['Sport'].value_counts()
    fig3, ax3 = plt.subplots(figsize=fig_size)
    sns.barplot(x=top_sports.values, y=top_sports.index, palette='viridis', ax=ax3)
    ax3.set_title("Sports by Number of Events")
    ax3.set_xlabel("Number of Events")
    ax3.set_ylabel("Sport")
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.0f')

    gender_dist = olympic_data['Gender'].value_counts()
    fig4, ax4 = plt.subplots(figsize=fig_size)
    ax4.pie(gender_dist, labels=gender_dist.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax4.set_title("Medal Distribution by Gender")
    ax4.axis('equal')

    info2 = """
    - Aquatics has the highest number of events.  
    - Around 60% medals are won by men.
    """
    two_col_section("🚹🚺 Sports & Gender Distribution", info2, fig3, fig4)

    # Section 3: Medal Types & Gender Participation Over Years
    medal_counts_pivot = olympic_data.groupby(['Country', 'Medal']).size().unstack(fill_value=0)
    medal_counts_pivot['Total'] = medal_counts_pivot.sum(axis=1)
    top_10 = medal_counts_pivot.sort_values('Total', ascending=False).head(10)[['Gold', 'Silver', 'Bronze']]

    fig5, ax5 = plt.subplots(figsize=fig_size)
    colors = ['#FFD700', '#C0C0C0', '#cd7f32']
    bottom = np.zeros(len(top_10))
    for idx, medal in enumerate(top_10.columns):
        bars = ax5.bar(top_10.index, top_10[medal], bottom=bottom, color=colors[idx], label=medal)
        bottom += top_10[medal].values
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f'{int(height)}',
                        ha='center', va='center', fontsize=10)
    ax5.set_title('Top 10 Countries by Medal Type (Stacked)')
    ax5.set_xlabel('Country')
    ax5.set_ylabel('Number of Medals')
    ax5.legend(title='Medal Type')
    plt.xticks(rotation=45)

    filtered_data = olympic_data[olympic_data['Year'] >= 1976]
    gender_participation = filtered_data.groupby(['Year', 'Gender']).size().unstack(fill_value=0).sort_index()
    fig6, ax6 = plt.subplots(figsize=fig_size)
    colors = ['#1f77b4', '#ff7f0e']
    bottom = np.zeros(len(gender_participation))
    x_pos = range(len(gender_participation))
    for idx, gender in enumerate(gender_participation.columns):
        values = gender_participation[gender]
        bars = ax6.bar(x_pos, values, bottom=bottom, label=gender, color=colors[idx])
        bottom += values.values
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f'{int(height)}',
                        ha='center', va='center', fontsize=9, color='white')
    ax6.set_title('Year-wise Gender Participation in Olympic Events')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Number of Participants')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(gender_participation.index.astype(int), rotation=45)
    ax6.legend(title='Gender')

    info3 = """
    - United States leads all medal types Gold, Silver, Bronze.  
    - Men’s participation has been consistently higher than women’s since 1976.
    """
    two_col_section("🥇🥈🏅 Medal Types & Gender Participation Over Years", info3, fig5, fig6)

    # Section 4: Country-Sport Medal Combinations (Treemap)
    st.markdown("### Top 30 Country-Sport Medal Combinations (Treemap)")
    medal_counts_cs = olympic_data.groupby(['Country', 'Sport']).size().reset_index(name='Medal_Count')
    top_medals = medal_counts_cs.sort_values('Medal_Count', ascending=False).head(30)
    top_medals['Label'] = top_medals['Country'] + " - " + top_medals['Sport'] + "\n(" + top_medals['Medal_Count'].astype(str) + ")"
    colors = sns.color_palette('viridis', len(top_medals))

    fig7, ax7 = plt.subplots(figsize=fig_size)
    squarify.plot(sizes=top_medals['Medal_Count'], label=top_medals['Label'], alpha=0.9,
                  color=colors, text_kwargs={'fontsize':9, 'color':'white', 'weight':'bold'}, ax=ax7)
    ax7.axis('off')
    plt.title('🌲 Top 30 Country-Sport Medal Combinations (Treemap)')
    st_pyplot(fig7)

elif option == "Medal Prediction":
    st.header("Medal Prediction for Next Olympics")

    selected_country = st.selectbox("Select Country:", countries)

    recent_year = medal_counts['Year'].max()
    st.write(f"Latest year in dataset: {recent_year}")

    recent_data = medal_counts[(medal_counts['Country'] == selected_country) & (medal_counts['Year'] == recent_year)]

    if recent_data.empty:
        st.warning(f"No data available for {selected_country} in year {recent_year}.")
    else:
        actual_medal_count = int(recent_data['Total_Medals'].values[0])
        st.write(f"**Actual Medal Count ({recent_year}):** {actual_medal_count}")

        next_year = recent_year + 4
        features = recent_data[['Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']].copy()
        features['Year'] = next_year
        features['Country_encoded'] = le_country.transform([selected_country])[0]

        X_pred = features[['Year', 'Country_encoded', 'Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']]
        predicted_medals_log = rf_model.predict(X_pred)[0]
        predicted_medals = max(0, int(round(np.expm1(predicted_medals_log))))

        st.write(f"**Predicted Medal Count for next Olympics:** {predicted_medals}")

        # Actual vs Predicted medals for 2008 (model validation)
        st.subheader("Model Check: Actual vs Predicted for 2008")
        df_2008 = medal_counts[medal_counts['Year'] == 2008].copy()
        df_2008['Country_encoded'] = le_country.transform(df_2008['Country'])
        X_2008 = df_2008[['Year', 'Country_encoded', 'Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']]
        y_2008_actual = df_2008['Total_Medals']
        y_2008_pred_log = rf_model.predict(X_2008)
        y_2008_pred = np.expm1(y_2008_pred_log)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(df_2008['Country'], y_2008_actual, marker='o', label='Actual Medals', linewidth=2)
        ax.plot(df_2008['Country'], y_2008_pred, marker='s', label='Predicted Medals', linewidth=2)
        ax.set_title('Top Countries by Actual vs Predicted Medals (2008)')
        ax.set_xlabel('Country')
        ax.set_ylabel('Total Medals')
        plt.xticks(rotation=90)
        ax.legend()
        plt.tight_layout()
        st_pyplot(fig)


elif option == "Medal Prediction":
    st.header("Medal Prediction for Next Olympics")

    selected_country = st.selectbox("Select Country:", countries)

    recent_year = medal_counts['Year'].max()
    st.write(f"Latest year in dataset: {recent_year}")

    recent_data = medal_counts[(medal_counts['Country'] == selected_country) & (medal_counts['Year'] == recent_year)]

    if recent_data.empty:
        st.warning(f"No data available for {selected_country} in year {recent_year}.")
    else:
        actual_medal_count = int(recent_data['Total_Medals'].values[0])
        st.write(f"**Actual Medal Count ({recent_year}):** {actual_medal_count}")

        next_year = recent_year + 4
        features = recent_data[['Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']].copy()
        features['Year'] = next_year
        features['Country_encoded'] = le_country.transform([selected_country])[0]

        X_pred = features[['Year', 'Country_encoded', 'Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']]
        predicted_medals_log = rf_model.predict(X_pred)[0]
        predicted_medals = max(0, int(round(np.expm1(predicted_medals_log))))

        st.write(f"**Predicted Medal Count for next Olympics:** {predicted_medals}")

        # Actual vs Predicted medals for 2008 (model validation)
        st.subheader("Model Check: Actual vs Predicted for 2008")
        df_2008 = medal_counts[medal_counts['Year'] == 2008].copy()
        df_2008['Country_encoded'] = le_country.transform(df_2008['Country'])
        X_2008 = df_2008[['Year', 'Country_encoded', 'Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']]
        y_2008_actual = df_2008['Total_Medals']
        y_2008_pred_log = rf_model.predict(X_2008)
        y_2008_pred = np.expm1(y_2008_pred_log)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(df_2008['Country'], y_2008_actual, marker='o', label='Actual Medals', linewidth=2)
        ax.plot(df_2008['Country'], y_2008_pred, marker='s', label='Predicted Medals', linewidth=2)
        ax.set_title('Top Countries by Actual vs Predicted Medals (2008)')
        ax.set_xlabel('Country')
        ax.set_ylabel('Total Medals')
        plt.xticks(rotation=90)
        ax.legend()
        plt.tight_layout()
        st_pyplot(fig)


elif option == "Medal Prediction":
    st.header("Medal Prediction for Next Olympics")

    selected_country = st.selectbox("Select Country:", countries)

    recent_year = medal_counts['Year'].max()
    st.write(f"Latest year in dataset: {recent_year}")

    recent_data = medal_counts[(medal_counts['Country'] == selected_country) & (medal_counts['Year'] == recent_year)]

    if recent_data.empty:
        st.warning(f"No data available for {selected_country} in year {recent_year}.")
    else:
        actual_medal_count = int(recent_data['Total_Medals'].values[0])
        st.write(f"**Actual Medal Count ({recent_year}):** {actual_medal_count}")

        next_year = recent_year + 4
        features = recent_data[['Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']].copy()
        features['Year'] = next_year
        features['Country_encoded'] = le_country.transform([selected_country])[0]

        X_pred = features[['Year', 'Country_encoded', 'Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']]
        predicted_medals_log = rf_model.predict(X_pred)[0]
        predicted_medals = max(0, int(round(np.expm1(predicted_medals_log))))

        st.write(f"**Predicted Medal Count for next Olympics:** {predicted_medals}")

        # Actual vs Predicted medals for 2008 (model validation)
        st.subheader("Model Check: Actual vs Predicted for 2008")
        df_2008 = medal_counts[medal_counts['Year'] == 2008].copy()
        df_2008['Country_encoded'] = le_country.transform(df_2008['Country'])
        X_2008 = df_2008[['Year', 'Country_encoded', 'Sport_Count', 'Gold_Count', 'Silver_Count', 'Bronze_Count']]
        y_2008_actual = df_2008['Total_Medals']
        y_2008_pred_log = rf_model.predict(X_2008)
        y_2008_pred = np.expm1(y_2008_pred_log)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(df_2008['Country'], y_2008_actual, marker='o', label='Actual Medals', linewidth=2)
        ax.plot(df_2008['Country'], y_2008_pred, marker='s', label='Predicted Medals', linewidth=2)
        ax.set_title('Top Countries by Actual vs Predicted Medals (2008)')
        ax.set_xlabel('Country')
        ax.set_ylabel('Total Medals')
        plt.xticks(rotation=90)
        ax.legend()
        plt.tight_layout()
        st_pyplot(fig)

st.markdown(
    """
    <hr>
    <div style='text-align: center; padding: 20px 0;'>
        <p style='font-size: 1.2em; font-family: "Helvetica Neue", sans-serif; color: #a3a8b8;'>
             Olympics Data Analytics & Prediction Dashboard | Created with by <a href='https://github.com/rabhinav0906' target='_blank' style='color: #4bc0c0; text-decoration: none;'>Abhinav Rai</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)