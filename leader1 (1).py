import pandas as pd
import numpy as np
import statsmodels.api as sm
from base_leader import Leader  

class DataDrivenLeader(Leader):
    def __init__(self, name):
        self.historical_data = pd.DataFrame(columns=['Date', 'Leader_Price', 'Follower_Price'])
        self.model = None
        self.forgetting_factor = 0.90
        super().__init__(name)
  


    def collect_historical_data(self):
        for date in range(1, 101):
            leader_price, follower_price = self.get_price_from_date(date)
            self.historical_data = self.historical_data.append({
                'Date': date,
                'Leader_Price': leader_price,
                'Follower_Price': follower_price
            }, ignore_index=True)
        self.train_model()

    def train_model(self):
        # Training the regression model using historical leader prices to predict follower prices
        X = self.historical_data['Leader_Price'].values.reshape(-1, 1)
        y = self.historical_data['Follower_Price'].values

        # Generate weights with a forgetting factor
        # Newest observation has index -1, oldest has index 0
        n = len(self.historical_data)
        weights = self.forgetting_factor ** np.arange(n)[::-1]
            
        # Add a constant to the model for an intercept
        X = sm.add_constant(X)
            
        # Fit the weighted least squares model
        self.model = sm.WLS(y, X, weights=weights).fit()

    def new_price(self, date: int) -> float:
        if date > 101 :
            previous_day = date - 1
            leader_price, follower_price = self.get_price_from_date(previous_day)
            self.historical_data = self.historical_data.append({
                'Date': previous_day,
                'Leader_Price': leader_price,
                'Follower_Price': follower_price
            }, ignore_index=True)

            self.train_model()


        # Retrieve the most recent (last) entry in the historical data
        last_entry = self.historical_data.iloc[-1]
        last_leader_price = last_entry['Leader_Price']
        last_follower_price = last_entry['Follower_Price']

        # Model predicts the follower's response to this price
        predicted_response = self.model.predict([1, last_leader_price])[0]
        return predicted_response


    def start_simulation(self):
        self.log("Start of simulation")
        self.collect_historical_data()

    def end_simulation(self):
        self.log("End of simulation")


if __name__ == '__main__':
    leader = DataDrivenLeader('3')


