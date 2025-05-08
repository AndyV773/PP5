from app_pages.multipage import MultiPage


# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_data_study import page_data_study_body
from app_pages.page_prospect import page_prospect_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_target import page_predict_target_body

# Create an instance of the app
app = MultiPage(app_name="Stock Data Analysis")

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Data Study", page_data_study_body)
app.add_page("Prospect Predictometer", page_prospect_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Prospect Target", page_predict_target_body)


app.run()  # Run the app
