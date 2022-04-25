import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    city_temp = pd.read_csv(filename, parse_dates=["Date"])
    city_temp['DayOfYear'] = city_temp['Date'].dt.dayofyear
    city_temp.dropna()
    city_temp = city_temp[(city_temp["Temp"] > -71)]
    return city_temp
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp = load_data("C:/Users/User/PycharmProjects/IML.HUJI/datasets/City_Temperature.csv")
    # raise NotImplementedError()

    # Question 2 - Exploring data for specific country
    israel_temp = city_temp[(city_temp["Country"] == "Israel")]
    fig = px.scatter(israel_temp,
                     x="DayOfYear",
                     y="Temp",
                     color=israel_temp["Year"].astype(str),
                     title="Temp in Israel by day of year and year").show()
    israel_temp_monthly = israel_temp.groupby(['Month']).Temp.agg(['std'])
    barfig = px.bar(israel_temp_monthly, labels={"x": "Month", "y": "Temp Std", "title":"Temp std by month"}).show()

    # raise NotImplementedError()

    # Question 3 - Exploring differences between countries

    city_temp_2 = city_temp.groupby(['Country', 'Month'], as_index=False).agg({"Temp": ['mean', 'std']})
    px.line(x=city_temp_2["Month"],
            y=city_temp_2[("Temp", "mean")],
            color=city_temp_2["Country"],
            labels={"x": "Month", "y": "Mean Temp", "color": "Country"},
            error_y=city_temp_2[("Temp", "std")],
            title="Mean temp by month with std error").show()

    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    trainX, trainY, testX, testY = split_train_test(israel_temp["DayOfYear"], israel_temp["Temp"])
    loss = []
    degree = [i for i in range(1, 11)]
    for k in degree:
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(trainX.to_numpy(), trainY.to_numpy())
        k_loss = round(poly_fit.loss(testX.to_numpy(), testY.to_numpy()), 2)
        print(f"loss: {k_loss} for degree {k}")
        loss.append(k_loss)
    px.bar(x=degree,
           y=loss,
           title="Test error as a function of the value of k",
           labels={"x": "Degree", "y": "Error"}).show()
    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    poly_fit = PolynomialFitting(5)
    poly_fit.fit(israel_temp["DayOfYear"].to_numpy(), israel_temp["Temp"].to_numpy())
    countries_list = city_temp[city_temp['Country'] != 'Israel'].Country.unique()
    error = []
    for country in countries_list:
        countryX = city_temp[(city_temp["Country"] == country)]["DayOfYear"]
        countryY = city_temp[(city_temp["Country"] == country)]["Temp"]
        error.append(poly_fit.loss(countryX.to_numpy(), countryY.to_numpy()))
    px.bar(x=countries_list,
           y=error,
           title="Model error in other countries",
           labels={"x": "Country", "y": "Error"}).show()

    # raise NotImplementedError()
