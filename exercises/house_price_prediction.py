import numpy
import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_price = pandas.read_csv(filename)
    house_price.dropna()
    house_price.drop(["long", "lat", "id","date"], axis=1, inplace=True)
    house_price = house_price[(house_price["sqft_living15"] > 0) &
                     (house_price["sqft_lot15"] > 0) &
                     (house_price["sqft_living"] > 0) &
                     (house_price["sqft_lot"] > 0) &
                      (house_price["price"] > 0)]

    house_price = pd.get_dummies(data=house_price, columns=["zipcode"], drop_first=True)
    house_price["date"] = house_price["date"].str[:4]
    house_price = house_price[:].astype(float)
    house_price = house_price[(house_price >= 0).all(1)]
    house_price.head()
    print(house_price)
    return house_price
    # raise NotImplementedError()



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.columns:
        corelation = y.cov(X[feature])/(numpy.std(X[feature])*numpy.std(y))
        print(corelation)
        fig = go.Figure()
        fig.update_layout(
            title= f"House prices in respect for: {feature}, {round(corelation,2)}",
            xaxis_title=f"{feature}",
            yaxis_title="Prices"
        )
        fig.add_scatter(x=X[feature],y=y, mode="markers")
        # fig.write_image(f'{output_path}/{feature}.png')
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    house_price = load_data("C:/Users/User/PycharmProjects/IML.HUJI/datasets/house_prices.csv")
    response = house_price["price"]
    # raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(house_price, response)
    # raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    trainX, trainY, testX, testY = split_train_test(house_price,response)
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
