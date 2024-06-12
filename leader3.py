import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from base_leader import Leader 


class DataDrivenLeader(Leader):
    def __init__(self, name):
        self.historical_data = pd.DataFrame(columns=['Date', 'Leader_Price', 'Follower_Price'])
        self.model = None

        super().__init__(name)
  


    def collect_historical_data(self):
        for date in range(1, 100):
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
        self.model = LinearRegression()
        self.model.fit(X, y)
   

    def new_price(self, date: int) -> float:

        day_before = date - 1
        leader_price, follower_price = self.get_price_from_date(day_before)

        if date > 101 and date < 130 :
            self.historical_data = self.historical_data.append({
                'Date': day_before,
                'Leader_Price': leader_price,
                'Follower_Price': follower_price
            }, ignore_index=True)

            self.train_model()

        
        # Model predicts the follower's response to this price
        predicted_response = self.model.predict([[follower_price]])[0]

    
        return predicted_response


    def start_simulation(self):
        self.log("Start of simulation")
        self.collect_historical_data()

    def end_simulation(self):
        self.log("End of simulation")


if __name__ == '__main__':
    leader = DataDrivenLeader('3')


