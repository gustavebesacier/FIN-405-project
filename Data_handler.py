import wrds
import copy
import os
import numpy as np
import pandas as pd

USERNAME = 'gbesacier'

# Connect the WRDS database
# db = wrds.Connection(wrds_username='gbesacier')
# db=wrds.Connection(wrds_username='YOUR_USERNAME_HERE')

START_DATE = '1964-01-01'
END_DATE = '2023-12-31'

class Data():

    def __init__(self) -> None:
        if not os.path.isfile("Data/data.csv"):
            print("Data is being processed from WRDS. The operation can take up to 3 minutes.")
            self.db = wrds.Connection(wrds_username=USERNAME)
            print("Data has been loaded from WRDS. Only data with more than 36 observations has been kept.")
            self.data = self.download_data().groupby('permno').filter(lambda permno: len(permno) >= 36)
        else:
            print("Data has been loaded from local files. Only data with more than 36 observations has been kept.")
            self.data =  self.clean_semicolumn(pd.read_csv("Data/data.csv")).groupby('permno').filter(lambda permno: len(permno) >= 36)

        self.__rolling_beta_data = 0


    def get_data(self):
        return self.data
    

    def get_rolling_beta(self, winsorize=True):

        if not isinstance(self.__rolling_beta_data, pd.DataFrame):
            self.__rolling_beta_data = copy.deepcopy(self.set_rolling_beta())

        data = copy.deepcopy(self.__rolling_beta_data)

        if winsorize:
            data['beta'] = data['beta'].clip(data['beta'].quantile(0.05), data['beta'].quantile(0.95))
        
        return data
    

    def set_rolling_beta(self):

        # Compute time varying beta for each stock
        self.data["date"] = pd.to_datetime(self.data["date"], format = "%Y-%m-%d") # set date in the correct format

        self.data["Rm_e"] = self.data["vwretd"] - self.data["tmytm"]
        self.data["R_e"]  = self.data["ret"] - self.data["tmytm"]

        # From PS_5_solution
        cov_nm = self.data.set_index('date').groupby('permno')[['R_e','Rm_e']].rolling(60, min_periods=36).cov()
        beta_n = (cov_nm.iloc[1::2,1].droplevel(2)/cov_nm.iloc[0::2,1].droplevel(2))
        beta_n = beta_n.dropna().reset_index().rename(columns={'Rm_e':'beta'})
        beta_n['date'] = beta_n['date'] + pd.DateOffset(months=1)

        df_beta = pd.merge(
            left = self.data,
            right = beta_n,
            on = ["date", "permno"],
            how = "left"
        )
    
        return df_beta
    

    def get_market_return(self):
        """Get the market return from WRDS"""
        query_market = f"""
            SELECT date, vwretd 
            FROM crsp.msi 
            WHERE date >= {START_DATE!r} AND date <= {END_DATE!r};
            """
        RM = self.db.raw_sql(query_market, date_cols=['date'])
        RM.to_csv("Data/market_return.csv")

        return RM


    def get_riskfree_rate(self):
        """Get the riskfree rate from WRDS"""
        query_tbills = f"""
            SELECT mcaldt, tmytm
            FROM crsp.tfz_mth_rf            
            WHERE kytreasnox = 2000001 
            AND mcaldt >= {START_DATE!r}
            AND mcaldt <= {END_DATE!r};
            """
        RF = self.db.raw_sql(query_tbills, date_cols=['mcaldt'])
        RF["tmytm"] = np.exp(RF["tmytm"]/12/100) - 1 # adjust compounding
        RF.to_csv("Data/riskfree_rate.csv")
        return RF


    def get_stock_returns(self):
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
        RET = self.db.raw_sql(query_stocks, date_cols=['date'])
        RET.to_csv("Data/stock_returns.csv")

        return RET


    def merge_datasets(self):
        """Merge the three datasets"""
        # merge data between riskfree and ret dataframes
        data = pd.merge(
            left = self.get_stock_returns(), 
            right = self.get_riskfree_rate(),
            how = 'left',
            left_on = 'date',
            right_on = 'mcaldt'
        )

        data.drop(['mcaldt', 'shrcd', 'exchcd'], axis = 1, inplace = True) # delete useless columns

        # merge between data and market
        data = pd.merge(
            left = data,
            right = self.get_market_return(),
            how = 'left',
            left_on = 'date',
            right_on = 'date'
        )

        return data


    def clean_semicolumn(self, data):
        columns = [col for col in data.columns.tolist() if not ":" in col]
        data = data[columns]

        return data


    def download_data(self):
        if not os.path.exists("Data"):
            os.makedirs("Data")
        data = self.merge_datasets()
        data.to_csv("Data/data.csv")

        return data

