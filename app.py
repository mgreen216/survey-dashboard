import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import os
import json
import sys
from io import StringIO
import re
import plotnine as p9
from plotnine import *

# Set page configuration
st.set_page_config(
    page_title="Academic Survey Analysis Dashboard",
    layout="wide",
    page_icon="📊"
)

# Configure OpenAI API
def setup_openai_api():
    """Set up the OpenAI API key from Streamlit secrets or environment variable."""
    try:
        # First try to get from Streamlit secrets
        api_key = st.secrets.get("openai_api_key")
        if api_key:
            openai.api_key = api_key
            return True
        
        # Then try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            return True
    except Exception as e:
        st.sidebar.error(f"Error accessing API key: {str(e)}")
    return False

# LLM integration for natural language querying
def query_llm(prompt, model="gpt-4"):
    """Query the OpenAI API with a prompt."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analysis assistant specializing in survey data visualization for academic publications. Your responses should include Python code that generates publication-quality visualizations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying OpenAI API: {str(e)}"

def extract_code(response):
    """Extract Python code blocks from the LLM response."""
    code_pattern = r"```python(.*?)```"
    code_blocks = re.findall(code_pattern, response, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    return ""

def extract_explanation(response):
    """Extract explanation text from the LLM response."""
    explanation = re.sub(r"```python.*?```", "", response, flags=re.DOTALL)
    return explanation.strip()

# Data handling functions
def load_data(file):
    """Load survey data from various file formats."""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.csv':
            data = pd.read_csv(file)
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(file)
        elif file_extension == '.json':
            data = pd.read_json(file)
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_data_summary(df):
    """Generate a summary of the dataset for context."""
    summary = {
        "columns": list(df.columns),
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(5).to_dict(),
        "numerical_stats": df.describe().to_dict() if any(df.select_dtypes(include=['number']).columns) else {},
        "categorical_counts": {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object', 'category']).columns[:5]},
    }
    return summary

# Visualization functions for offline mode
def offline_analysis_matplotlib(df, question):
    """Generate matplotlib/seaborn visualizations for offline mode."""
    result = {
        "explanation": "Analysis based on your query:",
        "code": "",
        "fig": None
    }
    
    question_lower = question.lower()
    
    # Get column names mentioned in the question
    possible_columns = [col for col in df.columns if col.lower() in question_lower]
    if not possible_columns and len(df.columns) > 0:
        possible_columns = list(df.columns[:2])
    
    # Distribution analysis
    if any(word in question_lower for word in ["distribution", "histogram", "spread"]):
        if possible_columns and pd.api.types.is_numeric_dtype(df[possible_columns[0]].dtype):
            col = possible_columns[0]
            # Set FiveThirtyEight style for academic-quality visualization
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#3072b3')
            ax.set_title(f"Distribution of {col}", fontsize=16)
            ax.set_xlabel(col, fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            plt.tight_layout()
            
            result["explanation"] = f"The distribution of {col} reveals patterns in your survey responses, highlighting the central tendency and spread of values."
            result["code"] = f"""
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['{col}'].dropna(), kde=True, ax=ax, color='#3072b3')
ax.set_title("Distribution of {col}", fontsize=16)
ax.set_xlabel("{col}", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
plt.tight_layout()
"""
            result["fig"] = fig
    
    # Correlation/relationship analysis
    elif any(word in question_lower for word in ["correlation", "relationship", "scatter"]):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=df[col1], y=df[col2], ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            ax.set_title(f"Relationship between {col1} and {col2}", fontsize=16)
            ax.set_xlabel(col1, fontsize=14)
            ax.set_ylabel(col2, fontsize=14)
            plt.tight_layout()
            
            # Calculate correlation
            corr = df[[col1, col2]].corr().iloc[0,1]
            
            result["explanation"] = f"The scatter plot shows the relationship between {col1} and {col2}, with a correlation coefficient of {corr:.2f}. " + \
                                   (f"This indicates a strong positive relationship." if corr > 0.5 else 
                                    f"This indicates a strong negative relationship." if corr < -0.5 else 
                                    f"This indicates a weak relationship.")
            result["code"] = f"""
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x=df['{col1}'], y=df['{col2}'], ax=ax, scatter_kws={{'alpha':0.5}}, line_kws={{'color':'red'}})
ax.set_title("Relationship between {col1} and {col2}", fontsize=16)
ax.set_xlabel("{col1}", fontsize=14)
ax.set_ylabel("{col2}", fontsize=14)
plt.tight_layout()

# Calculate correlation
corr = df[['{col1}', '{col2}']].corr().iloc[0,1]
print(f"Correlation coefficient: {{corr:.2f}}")
"""
            result["fig"] = fig
    
    # Categorical analysis
    elif any(word in question_lower for word in ["categorical", "count", "frequency"]):
        if possible_columns:
            col = possible_columns[0]
            
            plt.style.use('fivethirtyeight')
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts = df[col].value_counts().sort_values(ascending=False).head(10)
            
            # Create bar chart
            bars = sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='Blues_d')
            
            # Add percentage labels
            total = sum(value_counts)
            for i, bar in enumerate(bars.patches):
                percentage = f"{value_counts.values[i]/total*100:.1f}%"
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + 5, 
                       percentage,
                       ha='center', va='bottom', fontsize=11)
            
            plt.xticks(rotation=45, ha='right')
            ax.set_title(f"Frequency of {col} Categories", fontsize=16)
            ax.set_xlabel(col, fontsize=14)
            ax.set_ylabel("Count", fontsize=14)
            plt.tight_layout()
            
            result["explanation"] = f"This chart displays the distribution of responses across different categories of '{col}', highlighting the most common responses in your survey data."
            result["code"] = f"""
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(12, 6))
value_counts = df['{col}'].value_counts().sort_values(ascending=False).head(10)

# Create bar chart
bars = sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='Blues_d')

# Add percentage labels
total = sum(value_counts)
for i, bar in enumerate(bars.patches):
    percentage = f"{{value_counts.values[i]/total*100:.1f}}%"
    ax.text(bar.get_x() + bar.get_width()/2, 
           bar.get_height() + 5, 
           percentage,
           ha='center', va='bottom', fontsize=11)

plt.xticks(rotation=45, ha='right')
ax.set_title("Frequency of {col} Categories", fontsize=16)
ax.set_xlabel("{col}", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()
"""
            result["fig"] = fig
    
    # Time series analysis
    elif any(word in question_lower for word in ["trend", "time", "series", "over time"]):
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols and len(date_cols) > 0:
            date_col = date_cols[0]
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                numeric_col = numeric_cols[0]
                
                # Convert to datetime if not already
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    pass
                
                # Group by date and aggregate
                if pd.api.types.is_datetime64_dtype(df[date_col]):
                    df_grouped = df.groupby(pd.Grouper(key=date_col, freq='M')).agg({numeric_col: 'mean'}).reset_index()
                else:
                    df_grouped = df.groupby(date_col).agg({numeric_col: 'mean'}).reset_index()
                
                plt.style.use('fivethirtyeight')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(df_grouped[date_col], df_grouped[numeric_col], marker='o', linestyle='-', linewidth=2, markersize=8)
                
                ax.set_title(f"Trend of {numeric_col} over {date_col}", fontsize=16)
                ax.set_xlabel(date_col, fontsize=14)
                ax.set_ylabel(numeric_col, fontsize=14)
                
                # Add data labels at each point
                for x, y in zip(df_grouped[date_col], df_grouped[numeric_col]):
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                result["explanation"] = f"This time series plot shows how {numeric_col} has changed over time. The visualization reveals trends and patterns that may indicate important shifts in your survey responses."
                result["code"] = f"""
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(12, 6))

# Convert to datetime if not already
df['{date_col}'] = pd.to_datetime(df['{date_col}'])

# Group by date and aggregate
df_grouped = df.groupby(pd.Grouper(key='{date_col}', freq='M')).agg({{'{numeric_col}': 'mean'}}).reset_index()

ax.plot(df_grouped['{date_col}'], df_grouped['{numeric_col}'], marker='o', linestyle='-', linewidth=2, markersize=8)

ax.set_title(f"Trend of {numeric_col} over {date_col}", fontsize=16)
ax.set_xlabel("{date_col}", fontsize=14)
ax.set_ylabel("{numeric_col}", fontsize=14)

# Add data labels at each point
for x, y in zip(df_grouped['{date_col}'], df_grouped['{numeric_col}']):
    ax.annotate(f"{{y:.1f}}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.xticks(rotation=45)
plt.tight_layout()
"""
                result["fig"] = fig
    
    # Comparison analysis
    elif any(word in question_lower for word in ["compare", "comparison", "group", "difference"]):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            plt.style.use('fivethirtyeight')
            
            # Count unique categories and adapt the chart type
            unique_cats = df[cat_col].nunique()
            
            if unique_cats <= 10:  # For fewer categories, use bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by category and calculate mean
                grouped_data = df.groupby(cat_col)[num_col].agg(['mean', 'std']).reset_index()
                grouped_data = grouped_data.sort_values('mean', ascending=False)
                
                # Create bar chart with error bars
                bars = sns.barplot(x=cat_col, y='mean', data=grouped_data, ax=ax, 
                                  palette='Blues_d', order=grouped_data[cat_col])
                
                # Add error bars
                for i, bar in enumerate(bars.patches):
                    if not pd.isna(grouped_data['std'].iloc[i]):
                        error = grouped_data['std'].iloc[i]
                        ax.errorbar(i, grouped_data['mean'].iloc[i], yerr=error, fmt='none', ecolor='black', capsize=5)
                
                ax.set_title(f"Comparison of {num_col} across {cat_col} Categories", fontsize=16)
                ax.set_xlabel(cat_col, fontsize=14)
                ax.set_ylabel(f"Mean {num_col}", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels
                for i, bar in enumerate(bars.patches):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 0.1, 
                           f"{bar.get_height():.2f}",
                           ha='center', va='bottom', fontsize=11)
                
                plt.tight_layout()
                
                result["explanation"] = f"This chart compares the average {num_col} across different {cat_col} categories in your survey data. The error bars represent the standard deviation, showing the spread of responses within each group."
                result["code"] = f"""
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(12, 6))

# Group by category and calculate mean
grouped_data = df.groupby('{cat_col}')['{num_col}'].agg(['mean', 'std']).reset_index()
grouped_data = grouped_data.sort_values('mean', ascending=False)

# Create bar chart with error bars
bars = sns.barplot(x='{cat_col}', y='mean', data=grouped_data, ax=ax, 
                  palette='Blues_d', order=grouped_data['{cat_col}'])

# Add error bars
for i, bar in enumerate(bars.patches):
    if not pd.isna(grouped_data['std'].iloc[i]):
        error = grouped_data['std'].iloc[i]
        ax.errorbar(i, grouped_data['mean'].iloc[i], yerr=error, fmt='none', ecolor='black', capsize=5)

ax.set_title(f"Comparison of {num_col} across {cat_col} Categories", fontsize=16)
ax.set_xlabel("{cat_col}", fontsize=14)
ax.set_ylabel(f"Mean {num_col}", fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add value labels
for i, bar in enumerate(bars.patches):
    ax.text(bar.get_x() + bar.get_width()/2, 
           bar.get_height() + 0.1, 
           f"{{bar.get_height():.2f}}",
           ha='center', va='bottom', fontsize=11)

plt.tight_layout()
"""
            else:  # For many categories, use box plot or swarm plot
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Create box plot
                sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax, palette='Blues', showfliers=False)
                sns.swarmplot(x=cat_col, y=num_col, data=df, ax=ax, color='black', alpha=0.5, size=4)
                
                ax.set_title(f"Distribution of {num_col} across {cat_col} Categories", fontsize=16)
                ax.set_xlabel(cat_col, fontsize=14)
                ax.set_ylabel(num_col, fontsize=14)
                plt.xticks(rotation=90)
                plt.tight_layout()
                
                result["explanation"] = f"This combined box and swarm plot shows the distribution of {num_col} across different {cat_col} categories. The box shows the median and interquartile range, while the points represent individual responses."
                result["code"] = f"""
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(14, 8))

# Create box plot
sns.boxplot(x='{cat_col}', y='{num_col}', data=df, ax=ax, palette='Blues', showfliers=False)
sns.swarmplot(x='{cat_col}', y='{num_col}', data=df, ax=ax, color='black', alpha=0.5, size=4)

ax.set_title(f"Distribution of {num_col} across {cat_col} Categories", fontsize=16)
ax.set_xlabel("{cat_col}", fontsize=14)
ax.set_ylabel("{num_col}", fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
"""
            
            result["fig"] = fig
    
    # Default analysis if no specific analysis type is detected
    else:
        # Create a general summary dashboard with multiple plots
        fig = plt.figure(figsize=(15, 10))
        plt.style.use('fivethirtyeight')
        
        # Set up a 2x2 grid for multiple plots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Overview text
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.axis('off')
        ax0.text(0.5, 0.5, f"Survey Data Overview\n\n"
                          f"Total Responses: {len(df)}\n"
                          f"Variables: {len(df.columns)}\n"
                          f"Numeric Variables: {len(df.select_dtypes(include=['number']).columns)}\n"
                          f"Categorical Variables: {len(df.select_dtypes(include=['object', 'category']).columns)}",
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12,
                 transform=ax0.transAxes)
        
        # Plot 1: First numeric variable distribution
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            ax1 = fig.add_subplot(gs[0, 1])
            num_col = numeric_cols[0]
            sns.histplot(df[num_col].dropna(), kde=True, ax=ax1, color='#3072b3')
            ax1.set_title(f"Distribution of {num_col}")
        
        # Plot 2: First categorical variable counts
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            ax2 = fig.add_subplot(gs[1, 0])
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts().sort_values(ascending=False).head(10)
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax2, palette='Blues_d')
            plt.xticks(rotation=45, ha='right')
            ax2.set_title(f"Top Categories of {cat_col}")
        
        # Plot 3: Correlation between first two numeric variables
        if len(numeric_cols) >= 2:
            ax3 = fig.add_subplot(gs[1, 1])
            num_col1, num_col2 = numeric_cols[0], numeric_cols[1]
            sns.scatterplot(x=df[num_col1], y=df[num_col2], ax=ax3, alpha=0.6)
            ax3.set_title(f"{num_col1} vs {num_col2}")
        
        plt.tight_layout()
        
        result["explanation"] = "This dashboard provides a general overview of your survey data, including distributions of key variables and relationships between them."
        result["code"] = """
# Create a multi-plot dashboard
fig = plt.figure(figsize=(15, 10))
plt.style.use('fivethirtyeight')

# Set up a 2x2 grid for multiple plots
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Overview text
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis('off')
ax0.text(0.5, 0.5, f"Survey Data Overview\\n\\n"
                  f"Total Responses: {len(df)}\\n"
                  f"Variables: {len(df.columns)}\\n"
                  f"Numeric Variables: {len(df.select_dtypes(include=['number']).columns)}\\n"
                  f"Categorical Variables: {len(df.select_dtypes(include=['object', 'category']).columns)}",
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12,
         transform=ax0.transAxes)

# Add more plots as appropriate for your data
"""
        result["fig"] = fig
    
    return result

def offline_analysis_ggplot(df, question):
    """Generate ggplot2-style visualizations using plotnine for offline mode."""
    result = {
        "explanation": "Analysis based on your query:",
        "code": "",
        "fig": None
    }
    
    question_lower = question.lower()
    
    # Get column names mentioned in the question
    possible_columns = [col for col in df.columns if col.lower() in question_lower]
    if not possible_columns and len(df.columns) > 0:
        possible_columns = list(df.columns[:2])
    
    # Distribution analysis
    if any(word in question_lower for word in ["distribution", "histogram", "spread"]):
        if possible_columns and pd.api.types.is_numeric_dtype(df[possible_columns[0]].dtype):
            col = possible_columns[0]
            
            # Create ggplot-style histogram with density curve
            plot = (
                ggplot(df, aes(x=col)) +
                geom_histogram(aes(y='..density..'), fill='#3072b3', color='white', bins=30, alpha=0.7) +
                geom_density(color='darkred', size=1) +
                theme_minimal() +
                labs(
                    title=f'Distribution of {col}',
                    x=col,
                    y='Density'
                ) +
                theme(
                    plot_title=element_text(size=16, face='bold'),
                    axis_title=element_text(size=12),
                    figure_size=(10, 6)
                )
            )
            
            result["explanation"] = f"The distribution of {col} reveals patterns in your survey responses, highlighting the central tendency and spread of values."
            result["code"] = f"""
from plotnine import *

# Create ggplot-style histogram with density curve
plot = (
    ggplot(df, aes(x='{col}')) +
    geom_histogram(aes(y='..density..'), fill='#3072b3', color='white', bins=30, alpha=0.7) +
    geom_density(color='darkred', size=1) +
    theme_minimal() +
    labs(
        title='Distribution of {col}',
        x='{col}',
        y='Density'
    ) +
    theme(
        plot_title=element_text(size=16, face='bold'),
        axis_title=element_text(size=12),
        figure_size=(10, 6)
    )
)
"""
            result["fig"] = plot.draw()
    
    # Categorical analysis
    elif any(word in question_lower for word in ["categorical", "count", "frequency"]):
        if possible_columns:
            col = possible_columns[0]
            
            # Get top categories for the plot
            top_cats = df[col].value_counts().head(10).reset_index()
            top_cats.columns = [col, 'count']
            
            # Calculate percentages
            top_cats['percentage'] = top_cats['count'] / top_cats['count'].sum() * 100
            
            # Create ggplot-style bar chart
            plot = (
                ggplot(top_cats, aes(x=col, y='count', fill='count')) +
                geom_bar(stat='identity', show_legend=False) +
                geom_text(aes(label='percentage.map("{:.1f}%".format)'), 
                          va='bottom', ha='center', size=10, nudge_y=1) +
                scale_fill_gradient(low='#cce6ff', high='#004d99') +
                theme_minimal() +
                labs(
                    title=f'Frequency of {col} Categories',
                    x=col,
                    y='Count'
                ) +
                theme(
                    plot_title=element_text(size=16, face='bold'),
                    axis_title=element_text(size=12),
                    axis_text_x=element_text(angle=45, hjust=1),
                    figure_size=(10, 6)
                )
            )
            
            result["explanation"] = f"This chart displays the distribution of responses across different categories of '{col}', highlighting the most common responses in your survey data."
            result["code"] = f"""
from plotnine import *

# Get top categories for the plot
top_cats = df['{col}'].value_counts().head(10).reset_index()
top_cats.columns = ['{col}', 'count']

# Calculate percentages
top_cats['percentage'] = top_cats['count'] / top_cats['count'].sum() * 100

# Create ggplot-style bar chart
plot = (
    ggplot(top_cats, aes(x='{col}', y='count', fill='count')) +
    geom_bar(stat='identity', show_legend=False) +
    geom_text(aes(label='percentage.map("{{:.1f}}%".format)'), 
              va='bottom', ha='center', size=10, nudge_y=1) +
    scale_fill_gradient(low='#cce6ff', high='#004d99') +
    theme_minimal() +
    labs(
        title='Frequency of {col} Categories',
        x='{col}',
        y='Count'
    ) +
    theme(
        plot_title=element_text(size=16, face='bold'),
        axis_title=element_text(size=12),
        axis_text_x=element_text(angle=45, hjust=1),
        figure_size=(10, 6)
    )
)
"""
            result["fig"] = plot.draw()
    
    # Comparison analysis
    elif any(word in question_lower for word in ["compare", "comparison", "group", "difference"]):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Prepare data for plotting
            # Get summary statistics by group
            summary_df = df.groupby(cat_col)[num_col].agg(['mean', 'sem']).reset_index()
            summary_df.columns = [cat_col, 'mean', 'sem']
            
            # Add confidence interval for error bars
            summary_df['lower'] = summary_df['mean'] - 1.96 * summary_df['sem']
            summary_df['upper'] = summary_df['mean'] + 1.96 * summary_df['sem']
            
            # Sort by mean value
            summary_df = summary_df.sort_values('mean', ascending=False)
            
            # Create ggplot-style bar chart with error bars
            plot = (
                ggplot(summary_df, aes(x=cat_col, y='mean', fill=cat_col)) +
                geom_bar(stat='identity', show_legend=False) +
                geom_errorbar(aes(ymin='lower', ymax='upper'), width=0.2) +
                scale_fill_brewer(palette='Blues') +
                theme_minimal() +
                labs(
                    title=f'Comparison of {num_col} by {cat_col}',
                    x=cat_col,
                    y=f'Mean {num_col} (with 95% CI)'
                ) +
                theme(
                    plot_title=element_text(size=16, face='bold'),
                    axis_title=element_text(size=12),
                    axis_text_x=element_text(angle=45, hjust=1),
                    figure_size=(12, 6)
                )
            )
            
            result["explanation"] = f"This chart compares the average {num_col} across different {cat_col} categories, with error bars showing the 95% confidence interval for each group mean."
            result["code"] = f"""
from plotnine import *

# Get summary statistics by group
summary_df = df.groupby('{cat_col}')['{num_col}'].agg(['mean', 'sem']).reset_index()
summary_df.columns = ['{cat_col}', 'mean', 'sem']

# Add confidence interval for error bars
summary_df['lower'] = summary_df['mean'] - 1.96 * summary_df['sem']
summary_df['upper'] = summary_df['mean'] + 1.96 * summary_df['sem']

# Sort by mean value
summary_df = summary_df.sort_values('mean', ascending=False)

# Create ggplot-style bar chart with error bars
plot = (
    ggplot(summary_df, aes(x='{cat_col}', y='mean', fill='{cat_col}')) +
    geom_bar(stat='identity', show_legend=False) +
    geom_errorbar(aes(ymin='lower', ymax='upper'), width=0.2) +
    scale_fill_brewer(palette='Blues') +
    theme_minimal() +
    labs(
        title='Comparison of {num_col} by {cat_col}',
        x='{cat_col}',
        y='Mean {num_col} (with 95% CI)'
    ) +
    theme(
        plot_title=element_text(size=16, face='bold'),
        axis_title=element_text(size=12),
        axis_text_x=element_text(angle=45, hjust=1),
        figure_size=(12, 6)
    )
)
"""
            result["fig"] = plot.draw()
    
    # Default analysis
    else:
        # Create a simple scatter plot of first two numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            # Add a color dimension if there's a categorical column
            if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
                cat_col = df.select_dtypes(include=['object', 'category']).columns[0]
                
                plot = (
                    ggplot(df, aes(x=col1, y=col2, color=cat_col)) +
                    geom_point(alpha=0.7) +
                    stat_smooth(method='lm', se=True, color='black', linetype='dashed') +
                    theme_minimal() +
                    labs(
                        title=f'Relationship between {col1} and {col2}',
                        x=col1,
                        y=col2
                    ) +
                    theme(
                        plot_title=element_text(size=16, face='bold'),
                        axis_title=element_text(size=12),
                        figure_size=(10, 6)
                    )
                )
            else:
                plot = (
                    ggplot(df, aes(x=col1, y=col2)) +
                    geom_point(alpha=0.7, color='#3072b3') +
                    stat_smooth(method='lm', se=True, color='red') +
                    theme_minimal() +
                    labs(
                        title=f'Relationship between {col1} and {col2}',
                        x=col1,
                        y=col2
                    ) +
                    theme(
                        plot_title=element_text(size=16, face='bold'),
                        axis_title=element_text(size=12),
                        figure_size=(10, 6)
                    )
                )
            
            result["explanation"] = f"This scatter plot shows the relationship between {col1} and {col2} with a linear trend line."
            result["code"] = f"""
from plotnine import *

# Create scatter plot with trend line
plot = (
    ggplot(df, aes(x='{col1}', y='{col2}')) +
    geom_point(alpha=0.7, color='#3072b3') +
    stat_smooth(method='lm', se=True, color='red') +
    theme_minimal() +
    labs(
        title='Relationship between {col1} and {col2}',
        x='{col1}',
        y='{col2}'
    ) +
    theme(
        plot_title=element_text(size=16, face='bold'),
        axis_title=element_text(size=12),
        figure_size=(10, 6)
    )
)
"""
            result["fig"] = plot.draw()
        else:
            # Default message if we don't have enough numeric columns
            result["explanation"] = "Could not create a default analysis. Please specify what type of visualization you'd like to see."
    
    return result

def offline_analysis_plotly(df, question):
    """Generate interactive plotly visualizations for offline mode."""
    result = {
        "explanation": "Analysis based on your query:",
        "code": "",
        "fig": None
    }
    
    question_lower = question.lower()
    
    # Get column names mentioned in the question
    possible_columns = [col for col in df.columns if col.lower() in question_lower]
    if not possible_columns and len(df.columns) > 0:
        possible_columns = list(df.columns[:2])
    
    # Distribution analysis
    if any(word in question_lower for word in ["distribution", "histogram", "spread"]):
        if possible_columns and pd.api.types.is_numeric_dtype(df[possible_columns[0]].dtype):
            col = possible_columns[0]
            
            # Create histogram with Plotly
            fig = px.histogram(
                df, x=col, 
                marginal="box",
                opacity=0.7,
                color_discrete_sequence=['#3072b3'],
                title=f"Distribution of {col}"
            )
            
            # Add mean line
            mean_val = df[col].mean()
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red")
            fig.add_annotation(
                x=mean_val,
                y=0.85,
                yref="paper",
                text=f"Mean: {mean_val:.2f}",
                showarrow=True,
                arrowhead=1,
                font=dict(size=12, color="red")
            )
            
            # Update layout
            fig.update_layout(
                font=dict(family="Arial", size=14),
                xaxis_title=col,
                yaxis_title="Count",
                template="simple_white",
                hoverlabel=dict(bgcolor="white", font_size=14),
                width=900, 
                height=500
            )
            
            result["explanation"] = f"The distribution of {col} reveals patterns in your survey responses, highlighting the central tendency and spread of values. The mean value is marked with a red dashed line."
            result["code"] = f"""
import plotly.express as px

# Create histogram with Plotly
fig = px.histogram(
    df, x="{col}", 
    marginal="box",
    opacity=0.7,
    color_discrete_sequence=['#3072b3'],
    title="Distribution of {col}"
)

# Add mean line
mean_val = df["{col}"].mean()
fig.add_vline(x=mean_val, line_dash="dash", line_color="red")
fig.add_annotation(
    x=mean_val,
    y=0.85,
    yref="paper",
    text=f"Mean: {{mean_val:.2f}}",
    showarrow=True,
    arrowhead=1,
    font=dict(size=12, color="red")
)

# Update layout
fig.update_layout(
    font=dict(family="Arial", size=14),
    xaxis_title="{col}",
    yaxis_title="Count",
    template="simple_white",
    hoverlabel=dict(bgcolor="white", font_size=14),
    width=900, 
    height=500
)
"""
            result["fig"] = fig
    
    # Categorical analysis
    elif any(word in question_lower for word in ["categorical", "count", "frequency"]):
        if possible_columns:
            col = possible_columns[0]
            
            # Get value counts and prepare data
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum() * 100).round(1)
            value_counts['label'] = value_counts[col].astype(str) + ' (' + value_counts['percentage'].astype(str) + '%)'
            value_counts = value_counts.sort_values('count', ascending=False).head(15)
            
            # Create interactive bar chart
            fig = px.bar(
                value_counts, 
                x=col, 
                y='count',
                text='percentage',
                color='count',
                color_continuous_scale='Blues',
                title=f"Frequency of {col} Categories"
            )
            
            # Update layout
            fig.update_traces(
                texttemplate='%{text}%',
                textposition='outside'
            )
            fig.update_layout(
                font=dict(family="Arial", size=14),
                xaxis_title=col,
                yaxis_title="Count",
                coloraxis_showscale=False,
                template="simple_white",
                width=900, 
                height=550,
                xaxis={'categoryorder':'total descending'}
            )
            
            result["explanation"] = f"This interactive chart displays the distribution of responses across different categories of '{col}', highlighting the most common responses in your survey data. Percentages are shown above each bar."
            result["code"] = f"""
import plotly.express as px

# Get value counts and prepare data
value_counts = df['{col}'].value_counts().reset_index()
value_counts.columns = ['{col}', 'count']
value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum() * 100).round(1)
value_counts['label'] = value_counts['{col}'].astype(str) + ' (' + value_counts['percentage'].astype(str) + '%)'
value_counts = value_counts.sort_values('count', ascending=False).head(15)

# Create interactive bar chart
fig = px.bar(
    value_counts, 
    x='{col}', 
    y='count',
    text='percentage',
    color='count',
    color_continuous_scale='Blues',
    title="Frequency of {col} Categories"
)

# Update layout
fig.update_traces(
    texttemplate='%{text}%',
    textposition='outside'
)
fig.update_layout(
    font=dict(family="Arial", size=14),
    xaxis_title="{col}",
    yaxis_title="Count",
    coloraxis_showscale=False,
    template="simple_white",
    xaxis={{'categoryorder':'total descending'}}
)
"""
            result["fig"] = fig
    
    # Comparison analysis
    elif any(word in question_lower for word in ["compare", "comparison", "group", "difference"]):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Create a box plot for comparison
            fig = px.box(
                df, 
                x=cat_col, 
                y=num_col,
                color=cat_col,
                points="all",
                notched=True,
                title=f"Comparison of {num_col} by {cat_col}"
            )
            
            # Add a sophisticated violin plot layer
            fig2 = px.violin(
                df, 
                x=cat_col, 
                y=num_col,
                color=cat_col,
                box=True,
                points=False
            )
            
            for trace in fig2.data:
                trace.showlegend = False
                trace.opacity = 0.2
                fig.add_trace(trace)
            
            # Update layout
            fig.update_layout(
                font=dict(family="Arial", size=14),
                xaxis_title=cat_col,
                yaxis_title=num_col,
                template="simple_white",
                boxmode='group',
                showlegend=False,
                width=900, 
                height=600
            )
            
            # Add means as horizontal lines
            for cat in df[cat_col].unique():
                mean_val = df[df[cat_col] == cat][num_col].mean()
                fig.add_shape(
                    type="line",
                    x0=cat, x1=cat,
                    y0=mean_val, y1=mean_val,
                    line=dict(color="red", width=3, dash="dash")
                )
            
            result["explanation"] = f"This visualization compares the distribution of {num_col} across different {cat_col} categories using box plots overlaid with violin plots. The red dashed lines indicate the mean values for each group."
            result["code"] = f"""
import plotly.express as px

# Create a box plot for comparison
fig = px.box(
    df, 
    x='{cat_col}', 
    y='{num_col}',
    color='{cat_col}',
    points="all",
    notched=True,
    title="Comparison of {num_col} by {cat_col}"
)

# Add a violin plot layer
fig2 = px.violin(
    df, 
    x='{cat_col}', 
    y='{num_col}',
    color='{cat_col}',
    box=True,
    points=False
)

for trace in fig2.data:
    trace.showlegend = False
    trace.opacity = 0.2
    fig.add_trace(trace)

# Update layout
fig.update_layout(
    font=dict(family="Arial", size=14),
    xaxis_title="{cat_col}",
    yaxis_title="{num_col}",
    template="simple_white",
    boxmode='group',
    showlegend=False
)

# Add means as horizontal lines
for cat in df['{cat_col}'].unique():
    mean_val = df[df['{cat_col}'] == cat]['{num_col}'].mean()
    fig.add_shape(
        type="line",
        x0=cat, x1=cat,
        y0=mean_val, y1=mean_val,
        line=dict(color="red", width=3, dash="dash")
    )
"""
            result["fig"] = fig
    
    # Default analysis
    else:
        # Create a dashboard with multiple charts
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Set up subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "table"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=["Dataset Overview", "Numeric Distribution", 
                         "Category Distribution", "Correlation Plot"],
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )
        
        # Create table for overview
        overview_data = [
            ["Total Records", len(df)],
            ["Total Variables", len(df.columns)],
            ["Numeric Variables", len(numeric_cols)],
            ["Categorical Variables", len(categorical_cols)],
            ["Missing Values", df.isna().sum().sum()]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"],
                            fill_color='#3072b3',
                            align='left',
                            font=dict(color='white', size=14)),
                cells=dict(values=list(zip(*overview_data)),
                           fill_color='#E5ECF6',
                           align='left',
                           font=dict(size=14)),
                columnwidth=[150, 100]
            ),
            row=1, col=1
        )
        
        # Add histogram for first numeric column
        if len(numeric_cols) > 0:
            num_col = numeric_cols[0]
            hist_data = go.Histogram(
                x=df[num_col],
                marker_color='#3072b3',
                opacity=0.7,
                name=num_col
            )
            fig.add_trace(hist_data, row=1, col=2)
            fig.update_xaxes(title_text=num_col, row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
        
        # Add bar chart for first categorical column
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            cat_counts = df[cat_col].value_counts().head(10)
            bar_data = go.Bar(
                x=cat_counts.index,
                y=cat_counts.values,
                marker_color='#3072b3',
                name=cat_col
            )
            fig.add_trace(bar_data, row=2, col=1)
            fig.update_xaxes(title_text=cat_col, row=2, col=1, tickangle=45)
            fig.update_yaxes(title_text="Count", row=2, col=1)
        
        # Add scatter plot for correlation between first two numeric columns
        if len(numeric_cols) >= 2:
            num_col1, num_col2 = numeric_cols[0], numeric_cols[1]
            scatter_data = go.Scatter(
                x=df[num_col1],
                y=df[num_col2],
                mode='markers',
                marker=dict(
                    color='#3072b3',
                    opacity=0.7,
                    size=8
                ),
                name=f"{num_col1} vs {num_col2}"
            )
            fig.add_trace(scatter_data, row=2, col=2)
            fig.update_xaxes(title_text=num_col1, row=2, col=2)
            fig.update_yaxes(title_text=num_col2, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Survey Data Dashboard",
            showlegend=False,
            template="simple_white",
            height=800,
            width=900,
            font=dict(family="Arial", size=12),
        )
        
        result["explanation"] = "This interactive dashboard provides a comprehensive overview of your survey data, including key metrics, distributions, and relationships between variables."
        result["code"] = """
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set up subplot grid
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "table"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=["Dataset Overview", "Numeric Distribution", 
                 "Category Distribution", "Correlation Plot"],
    vertical_spacing=0.1,
    horizontal_spacing=0.08
)

# Create custom dashboard visualizations based on your data
"""
        result["fig"] = fig
    
    return result

# Streamlit UI functions
def render_sidebar():
    """Render the sidebar with settings and controls."""
    st.sidebar.title("Settings")
    
    # Mode selection
    mode = st.sidebar.radio("Mode", ["Online (OpenAI API)", "Offline (Local Analysis)"])
    
    # OpenAI API settings if online mode
    if mode == "Online (OpenAI API)":
        api_configured = setup_openai_api()
        
        if not api_configured:
            st.sidebar.warning("OpenAI API key not found. Please set it in your environment variables as OPENAI_API_KEY or in the Streamlit secrets.")
            api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
            if api_key:
                openai.api_key = api_key
                st.sidebar.success("API Key set for this session")
                api_configured = True
        
        if api_configured:
            model = st.sidebar.selectbox(
                "Select Model", 
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-16k"], 
                index=0
            )
            temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    # Visualization style selection
    vis_style = st.sidebar.selectbox(
        "Visualization Style",
        ["Plotly (Interactive)", "Matplotlib/Seaborn (FiveThirtyEight)", "ggplot2 (plotnine)"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("About")
    st.sidebar.markdown("""
    **Survey Analysis Dashboard** is a lightweight tool for academic-quality data visualization with natural language querying.
    
    Features:
    - Publication-quality charts
    - Natural language interface
    - Offline and online mode
    - Multiple visualization styles
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 | Survey Analysis Dashboard")
    
    return {
        "mode": mode,
        "model": model if 'model' in locals() else "gpt-4",
        "temperature": temperature if 'temperature' in locals() else 0.1,
        "vis_style": vis_style,
        "api_configured": api_configured if 'api_configured' in locals() else False
    }

def render_header():
    """Render the header section."""
    st.title("📊 Survey Analysis Dashboard")
    st.markdown("""
    An academic-quality visualization tool for survey data with natural language interface.
    Upload your survey data and ask questions in plain English to generate publication-ready charts.
    """)
    
    st.markdown("---")

def render_file_upload():
    """Render the file upload section."""
    st.subheader("Import Survey Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your survey data file (CSV, Excel, or JSON)", type=["csv", "xlsx", "xls", "json"])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_data = st.button("Use Sample Data")
    
    if sample_data:
        # Create sample survey data
        data = {
            'Age': np.random.normal(35, 12, 100).round().astype(int),
            'Gender': np.random.choice(['Male', 'Female', 'Non-binary', 'Prefer not to say'], 100, p=[0.45, 0.45, 0.05, 0.05]),
            'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100, p=[0.3, 0.4, 0.2, 0.1]),
            'Income': np.random.lognormal(10.5, 0.5, 100).round(-3),
            'Satisfaction': np.random.choice([1, 2, 3, 4, 5], 100, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
            'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100, p=[0.5, 0.3, 0.2]),
            'Experience': np.random.randint(0, 25, 100),
            'Feedback': np.random.choice(['Positive', 'Neutral', 'Negative'], 100, p=[0.6, 0.3, 0.1]),
            'Survey_Date': pd.date_range(start='1/1/2023', periods=100)
        }
        sample_df = pd.DataFrame(data)
        
        return sample_df, None
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            return data, uploaded_file
    
    return None, None

def render_data_preview(df):
    """Render the data preview section."""
    st.subheader("Data Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(5), use_container_width=True)
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"- Records: {len(df)}")
        st.write(f"- Variables: {len(df.columns)}")
        st.write(f"- Numeric: {len(df.select_dtypes(include=['number']).columns)}")
        st.write(f"- Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}")
        st.write(f"- Missing Values: {df.isna().sum().sum()}")

def process_query(query, df, settings):
    """Process the user query and generate visualization based on settings."""
    st.subheader("Analysis Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if settings["mode"] == "Online (OpenAI API)":
        if not settings["api_configured"]:
            st.error("OpenAI API is not configured. Please enter your API key in the sidebar.")
            return None, None, None
        
        # Use LLM to generate analysis
        status_text.text("Analyzing your query...")
        progress_bar.progress(25)
        
        # Generate context for the LLM
        data_summary = get_data_summary(df)
        data_context = json.dumps(data_summary, default=str)
        
        vis_style = settings["vis_style"]
        style_prompt = ""
        
        if "Matplotlib" in vis_style:
            style_prompt = "Use Matplotlib and Seaborn with FiveThirtyEight style for visualization. Include styling that makes the charts publication-ready for academic papers."
        elif "ggplot2" in vis_style:
            style_prompt = "Use plotnine (Python's ggplot2 equivalent) for visualization with a style appropriate for academic publications. Focus on clean, publication-ready designs."
        elif "Plotly" in vis_style:
            style_prompt = "Use Plotly for interactive visualization with a style appropriate for academic publications. Make sure the visualization has appropriate annotations and is ready for academic presentation."
        
        prompt = f"""
        I have survey data with the following structure:
        {data_context}
        
        My question is: "{query}"
        
        Please provide:
        1. Python code to analyze the data and create a high-quality visualization to answer this question.
        2. A brief interpretation of the results.
        
        {style_prompt}
        
        The visualization should be publication-quality for academic purposes, with proper titles, labels, and formatting.
        """
        
        progress_bar.progress(50)
        
        llm_response = query_llm(prompt, model=settings["model"])
        code_snippet = extract_code(llm_response)
        explanation = extract_explanation(llm_response)
        
        progress_bar.progress(75)
        
        # Execute the code
        status_text.text("Generating visualization...")
        
        if code_snippet:
            try:
                # Redirect stdout to capture prints
                old_stdout = sys.stdout
                redirected_output = sys.stdout = StringIO()
                
                # Execute in a local namespace
                local_vars = {"df": df, "plt": plt, "sns": sns, "px": px, "go": go, "np": np, "pd": pd}
                exec(code_snippet, globals(), local_vars)
                
                # Get the figure from the locals
                fig = None
                for var_name, var_val in local_vars.items():
                    if var_name not in {"df", "plt", "sns", "px", "go", "np", "pd"} and var_name != "code_snippet":
                        if isinstance(var_val, (plt.Figure, go.Figure, px.Figure)) or \
                           str(type(var_val)).endswith("Figure'>") or \
                           str(type(var_val)).endswith("Axes'>") or \
                           hasattr(var_val, 'draw'):
                            fig = var_val
                            break
                
                # If we couldn't find a figure directly, see if plt has a figure
                if fig is None and "plt" in local_vars:
                    if plt.get_fignums():
                        fig = plt.gcf()
                
                # Check for any output
                output = redirected_output.getvalue()
                
                # Reset stdout
                sys.stdout = old_stdout
                
                progress_bar.progress(100)
                status_text.empty()
                
                return fig, code_snippet, explanation + "\n\n" + output if output else explanation
                
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"Error executing the generated code: {str(e)}")
                st.code(code_snippet, language="python")
                
                progress_bar.progress(100)
                status_text.empty()
                
                return None, code_snippet, explanation
        else:
            st.warning("No code was generated from the LLM response.")
            st.write(llm_response)
            
            progress_bar.progress(100)
            status_text.empty()
            
            return None, None, llm_response
    
    else:  # Offline mode
        # Use pre-built analysis functions
        status_text.text("Analyzing your query...")
        progress_bar.progress(50)
        
        if "Plotly" in settings["vis_style"]:
            result = offline_analysis_plotly(df, query)
        elif "ggplot2" in settings["vis_style"]:
            result = offline_analysis_ggplot(df, query)
        else:  # Matplotlib/Seaborn
            result = offline_analysis_matplotlib(df, query)
        
        progress_bar.progress(100)
        status_text.empty()
        
        return result["fig"], result["code"], result["explanation"]

def render_chart_display(fig, code, explanation):
    """Render the chart display section."""
    if fig is not None:
        # Display the figure
        if isinstance(fig, plt.Figure):
            st.pyplot(fig)
        elif str(type(fig)).endswith("Figure'>"):  # For plotly figures
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(fig, 'draw'):  # For plotnine figures
            try:
                st.pyplot(fig)
            except:
                st.error("Could not display the plotnine figure. Check the code below.")
    
    # Display explanation
    if explanation:
        st.subheader("Interpretation")
        st.markdown(explanation)
    
    # Display code
    if code:
        with st.expander("Show Code", expanded=False):
            st.code(code, language="python")
            
            # Add download button for the code
            code_file = code.encode('utf-8')
            st.download_button(
                label="Download Code",
                data=code_file,
                file_name="survey_analysis.py",
                mime="text/plain"
            )

def render_example_queries():
    """Render example queries for the user."""
    st.markdown("### Example Queries")
    
    example_queries = [
        "Show the distribution of age among respondents",
        "Compare satisfaction scores by gender",
        "What's the relationship between income and education level?",
        "Show the trend of satisfaction over time",
        "Create a dashboard summarizing the key findings",
        "What are the most common reasons for negative feedback?",
        "How does location affect income distribution?"
    ]
    
    cols = st.columns(3)
    for i, query in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(query, key=f"example_{i}"):
                st.session_state.user_query = query
                st.rerun()

def main():
    """Main application entry point."""
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Initialize session state for storing data and query
    if "data" not in st.session_state:
        st.session_state.data = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    
    # Render sidebar and header
    settings = render_sidebar()
    render_header()
    
    # Handle file upload if data not already loaded
    if st.session_state.data is None:
        data, uploaded_file = render_file_upload()
        if data is not None:
            st.session_state.data = data
            if uploaded_file:
                st.session_state.file_name = uploaded_file.name
            else:
                st.session_state.file_name = "sample_survey_data.csv"
    
    # If data is loaded, display data preview and query interface
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.markdown(f"Using data from: **{st.session_state.file_name}**")
        
        # Show data preview
        with st.expander("Data Preview", expanded=False):
            render_data_preview(df)
        
        st.markdown("---")
        st.subheader("Ask a Question About Your Data")
        
        # Input for the user query
        user_query = st.text_area("Enter your question in plain English", value=st.session_state.user_query, height=60)
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("Analyze", type="primary")
        with col2:
            render_example_queries()
        
        if analyze_button and user_query:
            st.session_state.user_query = user_query
            fig, code, explanation = process_query(user_query, df, settings)
            render_chart_display(fig, code, explanation)
        
        # If a query was submitted previously, process it
        elif st.session_state.user_query:
            fig, code, explanation = process_query(st.session_state.user_query, df, settings)
            render_chart_display(fig, code, explanation)

if __name__ == "__main__":
    main()
