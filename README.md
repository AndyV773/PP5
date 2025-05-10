# Stock Analysis: Phoenix Group Holdings plc (PHNX 2010 - 2025)

## The Predictometer

Welcome,

This is the Code Institute student template for the bring your own data project option in Predictive Analytics. We have preinstalled all of the tools you need to get started. It's perfectly okay to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Template Instructions section of this README.md file and modify the remaining paragraphs for your own project. Please do read the Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use.

## Dataset Content

- The dataset is sourced from [yfinance](https://ranaroussi.github.io/yfinance/), and is publicly available

- The dataset, based on Phoenix Group Holdings plc from 2010 to 2025, consists of 3,788 rows and 8 columns, representing the daily price movement. Each row contains the date/time, open, high, low, close, volume, dividends, and stock splits

| Variable     | Values                                                 | Information                                                                                                                                                                                                                       |
| ------------ | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| date         | 2010-01-04 00:00:00+00:00 to 2024-12-31 00:00:00+00:00 | The date represents a timestamp in datetime format, ranging from January 4, 2010, to December 31, 2024. The format includes the year, month, and day (YYYY-MM-DD), followed by the time (midnight) and the UTC time zone (+00:00) |
| open         | 302.94 - 797.58                                        | open represents the opening price. The first traded price of the day                                                                                                                                                              |
| high         | 310.06 - 821.91                                        | high represents the highest price of the day                                                                                                                                                                                      |
| low          | 300.35 - 791.94                                        | low represents the lowest price of the day                                                                                                                                                                                        |
| close        | 300.35 - 800.31                                        | close represents the closing price. The last traded price of the day                                                                                                                                                              |
| volume       | 0 - 37073433                                           | volume represents the total number of shares traded (bought and sold) during the day                                                                                                                                              |
| dividends    | 0 - 26.65                                              | dividends represent the cash payments or stock distributions that a company gives to its shareholders as a portion of its profits                                                                                                 |
| stock splits | 0                                                      | stock splits represent a company's decision to increase the number of shares by issuing more shares to existing shareholders                                                                                                      |

## Exploratory Features and Targets

To analyze business requirements and identify patterns in the data, we performed feature extraction, which included adding lag features from previous price values and extracting components from the date values. Before doing so, we needed to impute any missing data to prevent gaps from affecting other columns. Upon inspection, there was no missing data, but the Volume column contained values of 0. To handle this, we applied Pandas forward fill (ffill) to propagate the last valid observation forward

Next, we extracted the day of the week and the year from the Date column, after which we dropped the original Date column. We then created two days of lag features, which introduced two rows with NaN values at the start of the dataset. These rows were subsequently dropped. Additionally, to construct the classification and regression targets, we performed a shift of future values, which resulted in one NaN row at the end of the dataset, which we also removed. This brought the total number of dropped rows to three

Finally, we decided to drop the Dividends and Stock Splits columns as they held no meaningful value for the analysis. As a result, our stock dataset now consists of 3,785 rows and 22 columns

| Variable          | Values                                               | Information                                                                          |
| ----------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| year              | 2010 to 2024                                         |                                                                                      |
| weekday           | monday, tuesday, wednesday, thursday, friday, sunday |                                                                                      |
| open              | 302.94 - 797.58                                      | open represents the opening price. The first traded price of the day                 |
| high              | 310.06 - 821.91                                      | high represents the highest price of the day                                         |
| low               | 300.35 - 791.94                                      | low represents the lowest price of the day                                           |
| close             | 300.35 - 800.31                                      | close represents the closing price. The last traded price of the day                 |
| volume            | 34 - 37073433                                        | volume represents the total number of shares traded (bought and sold) during the day |
| pre_open          | 302.94 - 797.58                                      |                                                                                      |
| pre_open_2        | 302.94 - 797.58                                      |                                                                                      |
| pre_high          | 310.06 - 821.91                                      |                                                                                      |
| pre_high_2        | 310.06 - 821.91                                      |                                                                                      |
| pre_low           | 300.35 - 791.94                                      |                                                                                      |
| pre_low_2         | 300.35 - 791.94                                      |                                                                                      |
| pre_close         | 300.35 - 800.31                                      |                                                                                      |
| pre_close_2       | 300.35 - 800.31                                      |                                                                                      |
| pre_vol           | 34 - 37073433                                        |                                                                                      |
| pre_vol_2         | 34 - 37073433                                        |                                                                                      |
| pre_average       | 302.94 - 797.87                                      |                                                                                      |
| pre_average_2     | 302.94 - 797.87                                      |                                                                                      |
| average           | 302.94 - 797.87                                      |                                                                                      |
| tomorrows_average | 302.94 - 797.87                                      |                                                                                      |
| target            | 0 - 1                                                |                                                                                      |

## Business Requirements

BR1: The client wants to uncover key market indicators that are most predictive of whether the stock price will go up or down the next trading day

BR2: The client is looking to have a model developed that can generate daily predictions, enabling more informed decision-making, supporting automated trading strategies, and incorporating risk assessment to evaluate potential losses and market volatility

BR3: The client requires a dashboard that allows them to visualize key information, monitor daily predictions, and interact with data to support day-to-day decision-making

## Agile Methodology

### Epics

- Data Collection and Information Gathering
- Data Study and Visualization
- Data Cleaning, and Preparation
- Model Training, Optimization, and Validation
- Dashboard Planning, Design, and Development
- Dashboard Deployment and Release

### User Stories

- Data Collection and Information Gathering - Business Requirements 1, 2

  - As a developer, I want to import historical stock data from an external data source into a Jupyter Notebook, so that I can conduct a thorough analysis of the dataset

    - Acceptance Criteria:

      - The stock dataset is successfully downloaded from Yahoo Finance
      - Stock data is successfully save to CSV format

- Data Study and Visualization - Business Requirement 1, 3

  - As a developer, I want to visualize the dataset to identify usable information and assess missing values, So that I can better prepare the data for analysis and ensure quality before modeling

    - Acceptance Criteria:

      - A data profile report must be generated
      - Visualize missing data

  - As a developer, I want to extract meaningful features and define the target variable, So that the data is ready for supervised learning and exploratory analysis

    - Acceptance Criteria:

      - Extract features for exploratory analysis
      - Define the target variable for supervised learning

  - As a developer, I want to visualize the correlation and predictive power of all features using heatmaps, So that I can identify patterns and relationships between variables that may inform model design

    - Acceptance Criteria:

      - Analysis Correlation and PPS with a heat map
      - Visualizations should demonstrate the effect of cleaning

- Data Cleaning, and Preparation - Business Requirements 1, 2

  - As a developer, I want to implement a robust data cleaning process so that I can ensure the dataset is accurate, reliable, and of high quality

    - Acceptance Criteria:

      - Extract features and target data
      - All missing or null values in the dataset must be identified
      - Missing values are imputed
      - Visualize the effect of cleaning

- Model Training, Optimization, and Validation - Business Requirements 2

  - As a developer, I want to evaluate the performance of the predictive model so that I can ensure the reliability and accuracy of its predictions

    - Acceptance Criteria:

      - The predictive model must be evaluated to ensure reliability and accuracy of its predictions

  - As a developer I want to measure the model performance so that I can have reliable results with high predictive power

    - Acceptance Criteria:

      - Model evaluation metrics must be calculated (e.g., accuracy, precision, recall, F1-score for classification; RMSE, MAE, R² for regression)

- Dashboard Planning, Design, and Development - Business Requirements 3

  - As a client, I want to access the Streamlit landing page so that I can quickly gain an overview of the project

    - Acceptance Criteria:

      - The client should be able to quickly gain an overview of the project through the Streamlit landing page

- Dashboard Deployment and Release - Business Requirements 3

  - As a developer, I want to initiate the deployment process of my application on Render, or Heroku at an early stage so that I can conduct end-to-end manual deployment testing from the outset

    - Acceptance Criteria:

      - The application must be successfully deployed

## Hypothesis and how to validate?

1. The suspect that correlation patterns between date or volume and key market indicators, are strong enough to identify predictive relationships

2. Historical stock data, including key features like price and volume, can be used in a binary classification model to predict whether tomorrow's average price will be higher or lower than today’s, achieving an accuracy of at least 70%

3. A regression model trained on historical stock data (open, close, and volume) can accurately forecast tomorrow's average price, and this forecast can be used to determine the directional change relative to today’s price

4. Unsupervised learning techniques, such as clustering, can identify distinct market regimes or patterns in historical price and volume data, which can improve the prediction of future price directions and help identify significant market structures

## The rationale to map the business requirements to the Data Visualizations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualizations and ML tasks

## ML Business Case

### Business Requirements:

- The client wants to identify key market indicators that are highly predictive of whether the stock price of Phoenix Group Holdings plc (PHNX) will rise or fall on the next trading day. This includes understanding the influence of technical indicators, historical price movements, and market trends

- The client seeks to have a predictive model developed that can generate daily stock price forecasts for PHNX, support automated trading strategies, and incorporate risk assessment to evaluate potential losses and market volatility. The model should be optimized for accuracy and capable of reflecting real-time market dynamics

- The client requires an interactive dashboard that enables the visualization of key market indicators, daily predictions, and risk assessments. The dashboard should allow the client to monitor stock movements, explore predictions, and interact with data to support day-to-day trading decisions and risk management

### Can Traditional Data Analysis Be Used?

Traditional analysis could be used to observe historical trends and identify basic patterns, but it lacks the capacity for accurate predictions and real-time risk assessment. A machine learning model is necessary to capture complex relationships, seasonality, and market volatility for precise forecasting

### Does the Client Need a Dashboard or API?

The client requires a dashboard to visualize market movements, monitor predictions, and assess risk dynamically

### A Successful Project Outcome for the Client Is Defined As:

- An analysis that identifies the key indicators most correlated with stock price changes to support informed trading decisions

- Accurate daily predictions with a well-defined risk assessment to optimize buy/sell strategies

- A dashboard that allows for real-time monitoring of stock movements and model predictions

### Are There Any Ethical or Privacy Concerns?

The dataset is sourced from [yfinance](https://ranaroussi.github.io/yfinance/) and is publicly available. Therefore, there are no ethical or privacy concerns

### Are There Clear EPICS and User Stories for Agile Implementation?

Yes, EPICS have been defined and user stories have been created, organized, and tracked for agile implementation

**EPICS Are Broken Down As Follows:**

1. Information Gathering and Data Collection

2. Data Visualization, Cleaning, and Preparation

3. Model Training, Optimization, and Validation

4. Dashboard Planning, Design, and Development

5. Dashboard Deployment and Release

### Does the Data Suggest a Particular Model?

**Given the nature of the predictions, a combination of Linear Regression and Logistic Regression is appropriate:**

- Linear Regression will be used to predict the daily stock price values, capturing the relationships between historical prices and future movements

- Logistic Regression will be applied to predict the directional movement of the stock price (up or down) for the next trading day, based on key market indicators

**This approach leverages the strengths of both models:**

- Linear Regression for continuous value estimation (stock price)

- Logistic Regression for binary classification (price increase or decrease)

## What Are the Project Inputs and Intended Outputs?

**Model Inputs:**

- Historical stock data for Phoenix Group Holdings plc (PHNX), including Date features, Open, High, Low, Close, Volume, and exploratory data

**Outputs:**

- Daily price forecasts for PHNX stocks

- Directional movement predictions (up or down) for decision-making

- Risk assessment metrics for evaluating market probability

- Interactive visualizations on a dashboard for user engagement

The model will predict the next day's stock price based on historical patterns and key indicators. The dashboard will enable the client to explore predictions, assess risk, and simulate trading decisions

### What Does Success Look Like?

It is agreed that a R² score of at least 0.75 for both the training set and the test set defines success. In addition, the model must achieve a precision of at least 70% to ensure reliable performance in its predictions. The dashboard must also display predictions and risk metrics with real-time updates and interactive exploration capabilities, providing users with actionable insights and allowing them to dive deeper into the data

### How Will the Client Benefit?

**The client will benefit from:**

- More informed decision-making based on accurate daily predictions

- Optimized trading strategies with integrated risk assessment

- Clear visualization of key metrics for rapid market assessment

- The ability to interact with predictions through a streamlined dashboard, enhancing trading efficiency

## CRISP-DM

| Process                | Description                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Business Understanding | Understand the client's objectives and business requirements, including predictive modeling, risk assessment, and interactive dashboard visualization                           |
| Data Understanding     | Collect and analyze historical stock data for Phoenix Group Holdings plc (PHNX) from 2010 to 2025, identify key variables, and evaluate data quality and completeness           |
| Data Preparation       | Clean, impute, and engineer features, including lag features, date components, and necessary transformations. Handle missing data and optimize the dataset for modeling         |
| Modeling               | Research and build predictive models to generate daily stock price forecasts, assess risk, and identify key indicators of market movement. Optimize for accuracy and robustness |
| Evaluation             | Evaluate model performance against business requirements, including risk assessment and prediction accuracy. Validate results with cross-validation and error analysis          |
| Deployment             | Develop and deploy an interactive dashboard that enables the client to visualize predictions, monitor risk, and interact with real-time data for informed decision-making       |

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
- Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: https://YOUR_APP_NAME.herokuapp.com/
- Set the runtime.txt Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

## Acknowledgements (optional)

- Thank the people who provided support through this project.
