import pandas as pd
import numpy as np
from base_leader import Leader  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

class DataDrivenLeader(Leader):
    def __init__(self, name):
        self.historical_data = pd.DataFrame(columns=['Date', 'Leader_Price', 'Follower_Price'])
        self.scaler = RobustScaler()
        self.model = None

        super().__init__(name)
  

    def collect_historical_data(self):
        for date in range(1, 101):
            leader_price, follower_price = self.get_price_from_date(date)
            self.historical_data = self.historical_data.append({
                'Date': date,
                'Leader_Price': leader_price,
                'Follower_Price': follower_price
            }, ignore_index=True)
        raw_df = pd.DataFrame(self.historical_data)

        # Apply a rolling mean to smooth the data
        raw_df['Leader_Price'] = raw_df['Leader_Price'].rolling(window=5, min_periods=1).mean()
        raw_df['Follower_Price'] = raw_df['Follower_Price'].rolling(window=5, min_periods=1).mean()

        # Detect and remove outliers
        q_low = raw_df['Follower_Price'].quantile(0.01)
        q_high = raw_df['Follower_Price'].quantile(0.99)
        raw_df = raw_df[(raw_df['Follower_Price'] > q_low) & (raw_df['Follower_Price'] < q_high)]

        self.historical_data = raw_df
        print("Historical data collection complete.")
        self.train_model()

    def train_model(self):
        X = self.historical_data[['Leader_Price', 'Date']]
        y = self.historical_data['Follower_Price']

        # Multiple train-test splits
        best_mse = float('inf')  # Initialize with a large number
        best_model = None
        n_splits = 4  # Number of times to split the data


        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model = GradientBoostingRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            if mse < best_mse:
                best_mse = mse
                best_model = model
            



        self.model = best_model
        print(f"Model training complete with best MSE: {best_mse}")

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
        predicted_response = self.model.predict([[last_leader_price, date]])[0]
            
        return predicted_response


    def start_simulation(self):
        self.log("Start of simulation")
        self.collect_historical_data()

    def end_simulation(self):
        self.log("End of simulation")


if __name__ == '__main__':
    leader = DataDrivenLeader('3')


