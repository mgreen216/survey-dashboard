# Survey Analysis Dashboard

A lightweight web app for academic-quality data visualization with natural language querying. This app allows users to upload survey data and generate publication-ready charts through simple English queries.

## Features

- **Academic-Quality Visualizations**: Create charts styled for academic publications
  - Matplotlib/Seaborn with FiveThirtyEight styling
  - ggplot2-style charts via plotnine
  - Interactive visualizations with Plotly

- **Natural Language Interface**: Ask questions about your survey data in plain English

- **Dual Operating Modes**:
  - **Online Mode**: Uses OpenAI's API to generate custom analysis code
  - **Offline Mode**: Works entirely locally with pre-built analysis templates

## Deployment on Streamlit Cloud

### Option 1: Direct GitHub Deployment

1. Fork this repository to your GitHub account
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select the forked repository
4. Set the main file path to `app.py`
5. Configure your secrets (OpenAI API key) in the Streamlit Cloud dashboard:
   - Go to "Advanced settings" > "Secrets"
   - Add your OpenAI API key as:
     ```
     openai_api_key = "your-key-here"
     ```
6. Deploy the app

### Option 2: Local Development and GitHub

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API key:
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your OpenAI API key to the `secrets.toml` file
5. Run the app locally:
   ```bash
   streamlit run app.py
   ```
6. Push to GitHub and follow the steps in Option 1 for cloud deployment

## Usage

1. Upload your survey data file (CSV, Excel, or JSON) or use the sample data
2. Type your question in natural language (e.g., "Show the distribution of age among respondents")
3. The app will generate and display an appropriate visualization
4. Download the generated code for further customization

## About

This app is designed for academic researchers, data scientists, and survey analysts who need publication-quality visualizations without extensive coding.
