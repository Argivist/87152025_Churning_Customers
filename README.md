# Customer Churn Prediction

This project is about predicting customer churn using machine learning models.

## Dependencies

The project requires the following Python libraries:

- pandas
- numpy
- tensorflow
- keras
- sklearn
- pickle
- streamlit

## Models

Two models are used in this project:

- `CustomerChurn_best.h5`
- `CustomerChurn_final.h5`

These models are loaded using the Keras module.

## Usage

The main function of the script is `main()`, which starts the Streamlit application. The application has a user interface for inputting customer data, and it uses the models to predict customer churn based on the input data.

The prediction function `pred(input,model)` takes an input array and a model (0 for the best model, 1 for the final model), and returns a prediction of whether the customer will churn.

## Running the Application

To run the application, simply run `python.exe -m streamlit run app.py`.
The Streamlit application will start and you can access it in your web browser.
