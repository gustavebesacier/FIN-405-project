import wrds
import os
import numpy as np
import pandas as pd

# Connect the WRDS database
# db = wrds.Connection(wrds_username='gbesacier')
# db=wrds.Connection(wrds_username='YOUR_USERNAME_HERE')

START_DATE = '1964-01-01'
END_DATE = '2023-12-31'


def get_market_return():
    """Get the market return from WRDS"""
    query_market = f"""
        SELECT date, vwretd 
        FROM crsp.msi 
        WHERE date >= {START_DATE!r} AND date <= {END_DATE!r};
        """
    RM = db.raw_sql(query_market, date_cols=['date'])
    RM.to_csv("Data/market_return.csv")

    return RM

def get_riskfree_rate():
    """Get the riskfree rate from WRDS"""
    query_tbills = f"""
        SELECT mcaldt, tmytm
        FROM crsp.tfz_mth_rf            
        WHERE kytreasnox = 2000001 
        AND mcaldt >= {START_DATE!r}
        AND mcaldt <= {END_DATE!r};
        """
    RF = db.raw_sql(query_tbills, date_cols=['mcaldt'])
    RF["tmytm"] = np.exp(RF["tmytm"]/12/100) - 1 # adjust compounding
    RF.to_csv("Data/riskfree_rate.csv")
    return RF

def get_stock_returns():
    """Get stock returns for all common stocks of AMEX and NYSE"""

    query_stocks = f"""
        SELECT data.permno, data.date, data.ret, data.shrout, data.prc, mse.siccd, mse.shrcd, mse.exchcd
        FROM crsp.msf AS data
        LEFT JOIN crsp.msenames AS mse
        ON data.permno = mse.permno 
        AND mse.namedt <= data.date
        AND data.date <= mse.nameendt
        WHERE data.date BETWEEN {START_DATE!r} AND {END_DATE!r}
        AND mse.exchcd BETWEEN 1 AND 2
        AND mse.shrcd BETWEEN 10 AND 11;
        """
    RET = db.raw_sql(query_stocks, date_cols=['date'])
    RET.to_csv("Data/stock_returns.csv")

    return RET

def merge_datasets():
    """Merge the three datasets"""
    # merge data between riskfree and ret dataframes
    data = pd.merge(
        left = get_stock_returns(), 
        right = get_riskfree_rate(),
        how = 'left',
        left_on = 'date',
        right_on = 'mcaldt'
    )

    data.drop(['mcaldt', 'shrcd', 'exchcd'], axis = 1, inplace = True) # delete useless columns

    # merge between data and market
    data = pd.merge(
        left = data,
        right = get_market_return(),
        how = 'left',
        left_on = 'date',
        right_on = 'date'
    )

    return data

def clean_semicolumn(data):
    columns = [col for col in data.columns.tolist() if not ":" in col]
    data = data[columns]

    return data

def download_data():
    if not os.path.exists("Data"):
        os.makedirs("Data")
    data = merge_datasets()
    data.to_csv("Data/data.csv")

    return data

def get_data():
    if not os.path.isfile("Data/data.csv"):
        db = wrds.Connection(wrds_username='gbesacier')
        return download_data()
    else:
        return clean_semicolumn(pd.read_csv("Data/data.csv"))