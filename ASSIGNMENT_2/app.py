import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import geopandas as gpd
import altair as alt
from vega_datasets import data
import streamlit as st
import zipfile
import requests



# Loading datasets

@st.cache_data # Cache the function to enhance performance 
def load_data():
        
    # Defination of url-paths
    url1 = 'https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_0.csv.zip'
    url2 = 'https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_1.csv.zip'
    url3 = 'https://github.com/aaubs/ds-master/raw/main/data/assignments_datasets/KIVA/kiva_loans_part_2.csv.zip'

    # Loading the urls into requests to download data

    response1 = requests.get(url1)

    response2 = requests.get(url2)

    response3 = requests.get(url3)

    # Saves the .zip data as files

    with open("kiva_loans_part_0.csv.zip", "wb") as file:
        file.write(response1.content)

    with open("kiva_loans_part_1.csv.zip", "wb") as file:
        file.write(response2.content)

    with open("kiva_loans_part_2.csv.zip", "wb") as file:
        file.write(response3.content)

    # Unzip the files to get .csv

    with zipfile.ZipFile("kiva_loans_part_0.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()

    with zipfile.ZipFile("kiva_loans_part_1.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()

    with zipfile.ZipFile("kiva_loans_part_2.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()

    # Loading partial datasets

    data_part1 = pd.read_csv("kiva_loans_part_0.csv")

    data_part2 = pd.read_csv("kiva_loans_part_1.csv")

    data_part3 = pd.read_csv("kiva_loans_part_2.csv")

    # Combining the datasets into one df using pd.concat

    
    df = pd.concat([data_part1, data_part2, data_part3])
    
 
    df = df.drop(['tags', 'use', 'currency', 'country_code','activity','id','partner_id',"region"], axis=1)
    
    #Deliting duplication of borrower_genders  

    df['borrower_genders'] = df['borrower_genders'].apply(lambda gender: gender.split(',')[0] if pd.notna(gender) else gender)
    #Dropping missing values
    df.dropna(inplace=True)  
    
    return df
    

# Load the data using the defined function
df=load_data()




st.title("KIVA - Microloans")
st.sidebar.header("Filters 📊")
    

##########################################
##########################################
# Sidebar filter: country Group
selected_country = st.sidebar.multiselect("Select a specific country", df['country'].unique().tolist())
if not selected_country:
    selected_country=df['country'].unique().tolist()
filtered_df = df[df['country'].isin(selected_country)]

# Sidebar filter: sector Group
selected_sector_group = st.sidebar.multiselect("Select a specific sector", filtered_df['sector'].unique().tolist())
if not selected_sector_group:
    selected_sector_group=filtered_df['sector'].unique().tolist()
else:
    pass
filtered_df = filtered_df[filtered_df['sector'].isin(selected_sector_group)]


#
# Sidebar filter: borrower_genders Group
selected_sector_genders = st.sidebar.multiselect("Select a specific genders", df['borrower_genders'].unique().tolist())
if not selected_sector_genders:
    selected_sector_genders=filtered_df['borrower_genders'].unique().tolist()
filtered_df = filtered_df[filtered_df['borrower_genders'].isin(selected_sector_genders)]



visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Statistical Data", 
     "Data focused by sector",
     "General Data"])

###################################################
###################################################

    # Calculate Z-scores
z_scores = zscore(filtered_df['loan_amount'])

# Get boolean array indicating the presence of outliers
# Using 2 & -2 z_scores to get 95% of data within 2 standard deviations
filtered_df['outlier_loan_amount'] = (z_scores > 2) | (z_scores < -2)


#Removing outliers
filtered_df = filtered_df[~filtered_df['outlier_loan_amount']]    
filtered_df = filtered_df.drop(["outlier_loan_amount"], axis=1)

# Sidebar filter: Monthly Income Range
min_loan = int(filtered_df['loan_amount'].min())
max_loan = int(filtered_df['loan_amount'].max())
loan_range = st.sidebar.slider("Select loan Range 💰", min_loan, max_loan, (min_loan, max_loan))
filtered_df = filtered_df[(filtered_df['loan_amount'] >= loan_range[0]) & (filtered_df['loan_amount'] <= loan_range[1])]
##################################################
##################################################
if visualization_option=="Statistical Data":
    
    
    # Display dataset overview
    st.dataframe(filtered_df.groupby(['country'])[['loan_amount', 'funded_amount']].agg(['mean', 'sum', 'max']))
    
    ##########################################
    correlation_matrix = filtered_df[['funded_amount', 'loan_amount', 'lender_count', 'term_in_months']].corr(method='spearman')
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    # Create the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    # Set the title
    ax.set_title('Correlation Heatmap')
    # Display the plot
    st.pyplot(fig)
    ########################################
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=filtered_df, y='loan_amount', palette='gist_rainbow')
    plt.title('Box Plot of Loan')
    plt.ylabel('Loan Amount')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    ########################################
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=filtered_df, x='sector', y='loan_amount', palette='gist_rainbow')
    plt.title('Box Plot of Loan')
    plt.xlabel('Sector')
    plt.ylabel('Loan Amount')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    

##########################################
if visualization_option=="Data focused by sector":

    plt.figure(figsize=(22, 8))
    sns.histplot(filtered_df, x='loan_amount', hue='sector', multiple='stack', kde=True, palette='gist_rainbow')
    plt.title('Distribution of Loan Amounts visualised by sector')
    plt.xlabel('loan Amount')
    plt.ylabel('frequency')
    st.pyplot(plt)

    ############################################
    # kiva plot - Density graph - SECTOR
    plt.figure(figsize=(10, 6))
    sns.kdeplot(filtered_df, x='loan_amount', hue='sector', fill=True, palette='gist_rainbow')
    plt.xlabel('Loan Amount')
    plt.ylabel('Density')
    plt.title('KDE Plot of Loan Amount by Sector')
    st.pyplot(plt)
    ###############################################
    # Create a pivot table for the heatmap
    
    outcome_heatmap_data = filtered_df.pivot_table(index='country', columns='sector', values='loan_amount', aggfunc='count', fill_value=0)

    # Plot the heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(outcome_heatmap_data, annot=True, fmt="d", cmap="coolwarm")

    # Set titles and labels
    plt.title('Heatmap, loan amount by sector & country')
    plt.xlabel('Sector')
    plt.ylabel('Country')
    plt.xticks(rotation=45)
    st.pyplot(plt)


#########################################

if visualization_option=="General Data":
    # kiva plot - Density graph
    fig, ax = plt.subplots()
    sns.kdeplot(filtered_df['loan_amount'], label='loan_amount', shade=True).set_title('loan_amount')
    plt.tight_layout()
    st.pyplot(fig)


    ############################
    country_loan_sum = filtered_df.groupby('country')['loan_amount'].sum()

    plt.figure(figsize=(14, 8))
    sns.barplot(x=country_loan_sum.index , y=country_loan_sum.values, palette="viridis")
    plt.title('Total Loan Amount For Each Country')
    plt.xlabel('Country')
    plt.ylabel('Total Loan Amount In US$ Millions')  # managed to add labessls (J)
    plt.xticks(rotation=90)
    st.pyplot(plt)

    ########################################

    # Ensure 'funded_time' is in a datetime format
    filtered_df['funded_time'] = pd.to_datetime(filtered_df['funded_time'])

    # Set the date column as the index
    filtered_df.set_index('funded_time', inplace=True)

    # Resample the data to a monthly frequency
    funded_trend = filtered_df.resample('M').size()


    # Plot the frequency of searches over time with improvements
    plt.figure(figsize=(14, 8))  # Increased figure size for better readability
    plt.plot(funded_trend, label='Total Funded Loans', color='lightblue', linewidth=1)  # Original data in a lighter color for comparison

    # Formatting the x-axis to display month and year
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))  # Show month and year on the x-axis
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Reset index to ensure 'funded_time' returns back as a column in the dataframe
    filtered_df.reset_index(inplace=True)

    # Add labels, title, and legend
    plt.title('Frequency of Funded Loans Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Funded Loans', fontsize=14)
    plt.legend(fontsize=12)

    # Add grid lines with better styling for clarity
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    st.pyplot(plt)