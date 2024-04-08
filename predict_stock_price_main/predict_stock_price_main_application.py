import datetime

from matplotlib import pyplot as plt
# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
import numpy as np  # Import the NumPy module with alias 'np'
import warnings

from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore")

import pandas as pd
file_path = "../dataset/TSLA.csv"

options = " TESLA STOCK LINEAR REGRESSION PREDICTION , " \
          "TESLA STOCK svm  REGRESSION PREDICTION, " \
          " TESLA STOCK DECISION TREE  PREDICTION, TESLA STOCK BAYESIAN RIDGE REGRESSION  PREDICTION, Exit".split(",")

# Input Start Date
def start_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    start = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    start = start.strftime('%Y-%m-%d')
    return start

# Input End Date
def end_date():
    while True:
        try:
            date_entry = input("Enter end date (MM/DD/YYYY): ")
            end = datetime.datetime.strptime(date_entry, '%m/%d/%Y')
            return end.date()
        except ValueError:
            print("Error: Please enter a valid date in the format MM/DD/YYYY")

# Input Symbols
def input_symbol():
    symbol = input("Enter symbol: ").upper()
    return symbol


# Support Vector Regression
def stock_svr():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df= pd.read_csv(file_path)
    n = len(df.index)
    X = np.array(df['Open']).reshape(n, -1)
    Y = np.array(df['Adj Close']).reshape(n, -1)

    from sklearn.svm import SVR
    dt = SVR()
    dt.fit(X, Y)

    print('_____________Summary:_____________')
    print('Accuracy Score:', dt.score(X, Y))
    print("")
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Adj Close'], dt.predict(X))
    plt.plot(X, dt.predict(X), color='red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices (SVR Regression)')
    plt.show()
    ans = ['1', '2']
    user_input = input("""                  
    What would you like to do next? Enter option 1 or 2.  
   1. Tesla Price Prediction Algorithm  
   2.Click To Exist  
    Command: """)
    while user_input not in ans:
        print("Error: Please enter a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()


# STOCK LINER REGRESSION
def stock_linear_regression():

    df = pd.read_csv(file_path)  # Use pd.read_csv to read the CSV file
    s = start_date()
    e = end_date()
    sym = input_symbol()

    n = len(df.index)
    X = np.array(df['Open']).reshape(n, -1)
    Y = np.array(df['Adj Close']).reshape(n, -1)
    lr = LinearRegression()
    lr.fit(X, Y)
    lr.predict(X)
    print('_____________Summary:_____________')
    print('Estimate intercept coefficient:', lr.intercept_)
    print('Number of coefficients:', len(lr.coef_))
    print('Accuracy Score:', lr.score(X, Y))
    print("")
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Adj Close'], lr.predict(X))
    plt.plot(X, lr.predict(X), color='red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices')
    plt.show()
    print('_____________Summary:_____________')
    print('Estimate intercept coefficient:', lr.intercept_)
    print('Number of coefficients:', len(lr.coef_))
    print('Accuracy Score:', lr.score(X, Y))
    print("")
    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
   1. Tesla Price Prediction Algorithm  
   2.Click To Exist  
Command: """)
    while user_input not in ans:
        print("Error: Please enter a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()



# Decision Tree Regression
def stock_decision_tree_regression():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df = pd.read_csv(file_path)  # Use pd.read_csv to read the CSV file

    n = len(df.index)
    X = np.array(df['Open']).reshape(n, -1)
    Y = np.array(df['Adj Close']).reshape(n, -1)
    sr = DecisionTreeRegressor()
    sr.fit(X, Y)
    predicted = sr.predict(X)
    print('_____________Summary:')
    print('Accuracy Score:', sr.score(X, Y))
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Adj Close'], predicted)
    plt.scatter(X, predicted, color='red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices ( Decision tree  Regression)')
    plt.show()
    print('_____________Summary:_____________')

    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
 1. Tesla Price Prediction Algorithm  
 2.Click To Exist  
Command: """)
    while user_input not in ans:
        print("Error: Please enter a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()


# BAYESIAN RIDE REGRESSION

def stock_bayesian_ridge_regression():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df = pd.read_csv(file_path)  # Use pd.read_csv to read the CSV file

    n = len(df.index)
    X = np.array(df['Open']).reshape(n, -1)
    Y = np.array(df['Adj Close']).reshape(n, -1)
    br = BayesianRidge()
    br.fit(X, Y)
    predicted = br.predict(X)
    print('_____________Summary:_____________')
    print('Estimated intercept coefficient:', br.intercept_)
    print('Number of coefficients:', len(br.coef_))
    print('Accuracy Score:', br.score(X, Y))
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Adj Close'], predicted)
    plt.plot(X, predicted, color='red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices (Bayesian Ridge Regression)')
    plt.show()
    print('_____________Summary:_____________')

    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
 1. Tesla Price Prediction Algorithm  
 2.Click To Exist  
Command: """)
    while user_input not in ans:
        print("Error: Please enter a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()


#***********************************************************************************************************************#
#******************************************************* MAIN MENU **********************************************************#
#***********************************************************************************************************************#  
def menu():
    ans = ['1', '2', '3', '4','5', '0']
    print(""" 
              
                                                      |  PRICE PREDICTION ALGORITHMS |
         ||||||||||||||||||||||||||||||||||||||||  MACHINE LEARNING STOCK PRICE PREDICTION  ||||||||||||||||||||||||||||||||||||||||
     
                 
                  
                  1.TESLA STOCK LINEAR REGRESSION PREDICTION 
                  2.TESLA STOCK SVM  REGRESSION PREDICTION
                  3.TESLA STOCK DECISION TREE  PREDICTION
                  4.TESLA STOCK BAYESIAN RIDGE REGRESSION  PREDICTION
                  5.MAIN MENU
                  0.CLICK EXIT 
                  """)
    user_input = input("Command (0-5): ")
    while user_input not in ans:
        print("Error: Please enter a valid option 0-5")
        user_input=input("Command: ")             
    if user_input == '1':
        stock_linear_regression()
    elif user_input == '2':
        stock_svr()
    elif user_input == '3':
        stock_decision_tree_regression()
    elif user_input == "4":
        stock_bayesian_ridge_regression()
    elif user_input == '5':
        beginning()
    elif user_input == "0":
        exit() 
        
        
#***********************************************************************************************************************#    
#*************************************************** Start of Program **************************************************# 
#***********************************************************************************************************************#  
def beginning():
    print()
    print("| WELCOME TO SUPERVISED  LEARNING  STOCK PRICE PREDICTION |")
    print("""
you can  choose option 1 or 2
              
 1.Tesla Price Prediction Algorithm  
 2.Click To Exist  

""")
    ans = ['1', '2']
    print()

    user_input=input("Please input your option?: ")
    print("""""")


    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input=input("Command: ")
    if user_input=="1":
        menu()
    elif user_input=="2":
        exit()
  
    
#***********************************************************************************************************************#     
beginning()      
