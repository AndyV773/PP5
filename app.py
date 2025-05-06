import streamlit as st
from app_pages.multipage import MultiPage


# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_data_study import page_data_study_body

app = MultiPage(app_name = "Stock Data Analysis") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Data Study", page_data_study_body)

app.run() # Run the app
