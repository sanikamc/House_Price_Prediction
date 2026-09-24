# Bengaluru House Price Prediction

## Machine Learning Project

## Project Overview

This project focuses on predicting house prices in Bengaluru using machine learning techniques. The project uses historical residential property data to analyse the factors that influence house prices and develop a model for estimating property prices.

The project demonstrates the complete machine learning workflow, including data preprocessing, exploratory data analysis, feature selection, model training, and model evaluation.

## Objective

The main objective of this project is to build a machine learning model that can estimate the price of residential properties in Bengaluru.

The project aims to:
- Analyse Bengaluru house-price data.
- Clean and preprocess the dataset.
- Identify important factors associated with house prices.
- Train a machine learning regression model.
- Evaluate the model using suitable performance metrics.
- Generate house-price predictions.

## Dataset

The project uses the Bengaluru House Data dataset, which contains information about residential properties in Bengaluru.

The dataset includes features such as:
- Area type
- Location
- Size of the property
- Total square feet
- Number of bathrooms
- Number of balconies
- Price

The dataset is used for data cleaning, analysis, visualization, and machine learning-based price prediction.

### Dataset Source

The dataset used in this project is the Bengaluru House Data dataset. The project folder contains the dataset file as `Bengaluru_House_Data.csv`.

**Original dataset source:** Add the exact original dataset URL here before final submission if required by the project guidelines.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code / Google Colab
- GitHub

## Project Workflow

1. Load the Bengaluru house-price dataset.
2. Explore the dataset and understand its features.
3. Handle missing and inconsistent data.
4. Perform data preprocessing.
5. Conduct exploratory data analysis and visualization.
6. Select relevant features for prediction.
7. Split the data into training and testing sets.
8. Train a Linear Regression model.
9. Evaluate the model using performance metrics.
10. Generate house-price predictions.
11. Analyse the results and draw conclusions.

## Machine Learning Model

A **Linear Regression** model is used to predict house prices.

The model uses the following input features:
- Location
- Total square feet
- Number of bathrooms
- Number of balconies
- BHK

The target variable is **house price**.

## Model Evaluation

The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The actual evaluation values are generated when the Python program is executed on the dataset.

## Expected Outcome

The project is expected to produce a machine learning model capable of estimating Bengaluru house prices from property-related features.

The analysis also helps identify patterns in the dataset and understand how factors such as location, property size, BHK, and number of bathrooms are related to house prices.

## Installation and Requirements

Python 3.x is required to run this project.

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

The complete list of dependencies is provided in `requirements.txt`.

## Usage

1. Place `Bengaluru_House_Data.csv` in the same folder as `house_price_prediction.py`.
2. Install the required dependencies.
3. Run the Python file:

```bash
python house_price_prediction.py
```

4. The program will load and preprocess the dataset.
5. Exploratory data analysis charts will be generated.
6. The Linear Regression model will be trained and evaluated.
7. Model performance metrics and a sample house-price prediction will be displayed.
8. Generated charts are saved in the `project_outputs` folder.

## Project Structure

```text
House_Price_Prediction/
│
├── Bengaluru_House_Data.csv
├── house_price_prediction.py
├── requirements.txt
├── README.md
└── Bengaluru_House_Price_Prediction_Report.docx
```

## Project Files

- `Bengaluru_House_Data.csv` – Dataset used for the project
- `house_price_prediction.py` – Main Python machine learning code
- `requirements.txt` – Required Python libraries
- `README.md` – Project documentation
- `Bengaluru_House_Price_Prediction_Report.docx` – Project report

## Conclusion

This project demonstrates the application of data analysis and machine learning to the real-world problem of Bengaluru house-price prediction.

By preprocessing the housing data, analysing important features, and training a regression model, the project aims to provide useful price estimates and insights into Bengaluru's residential property market.

## Future Scope

The project can be further improved by:
- Adding more recent Bengaluru housing data.
- Including additional property features.
- Comparing different machine learning algorithms.
- Performing hyperparameter tuning.
- Developing a web-based interface for price prediction.
- Deploying the model as a web application.
- Adding interactive dashboards for better analysis.

## References

1. Bengaluru House Data dataset – publicly available dataset source.
2. Python Documentation – https://docs.python.org/
3. Pandas Documentation – https://pandas.pydata.org/docs/
4. NumPy Documentation – https://numpy.org/doc/
5. Matplotlib Documentation – https://matplotlib.org/stable/
6. Seaborn Documentation – https://seaborn.pydata.org/
7. Scikit-learn Documentation – https://scikit-learn.org/stable/
