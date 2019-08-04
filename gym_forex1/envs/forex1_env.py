import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

INITIAL_ACCOUNT_BALANCE = 10000


class Forex1(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Forex1, self).__init__()

        df = pd.read_csv('./data/GBPUSD_Jan_2019_to_Aug_2019.csv')
        df = df.sort_values('Date_time')
        self.df = df

        self.CurrentMarketLevel = 0
        self.active_trade = 0
        self.profit = 0
        self.reward = 0

        self.account_balance = INITIAL_ACCOUNT_BALANCE

        # Actions of the format hold, Buy, Sell, close
        self.action_space = spaces.Discrete(3)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, ), dtype=np.float16)

    def _get_current_step_data(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        data_current_step = np.array([
            self.df.loc[self.current_step:self.current_step, 'Cycle_12_14_H4_1'].values,
            self.df.loc[self.current_step:self.current_step, 'Cycle_12_14_H4_0'].values,
            self.df.loc[self.current_step:self.current_step, 'CyH4_slope'].values,
            self.df.loc[self.current_step:self.current_step, 'CycleH4age'].values,
            self.df.loc[self.current_step:self.current_step, 'Cycle_12_14_H1_1'].values,
            self.df.loc[self.current_step:self.current_step, 'Cycle_12_14_H1_0'].values,
            self.df.loc[self.current_step:self.current_step, 'ColorStochastic_M5'].values,
            self.df.loc[self.current_step:self.current_step, 'ColorStochastic_M15'].values,
            self.df.loc[self.current_step:self.current_step, 'CyH1_slope'].values,
            self.df.loc[self.current_step:self.current_step, 'CycleH1age'].values,
            self.df.loc[self.current_step:self.current_step, 'Cycle_12_14_M15_1'].values,
            self.df.loc[self.current_step:self.current_step, 'Cycle_12_14_M15_0'].values,
            self.df.loc[self.current_step:self.current_step, 'CyM15_slope'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_blue'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_purple'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_yellow'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_lime'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_orange'].values,
            self.df.loc[self.current_step:self.current_step, 'Last_LagN_yellow'].values,
            self.df.loc[self.current_step:self.current_step, 'Last_LagN_lime'].values,
            self.df.loc[self.current_step:self.current_step, 'Last_LagN_orange'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_yellow_extreme'].values,
            self.df.loc[self.current_step:self.current_step, 'LagN_lime_extreme'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_blue_M15'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_red_M15'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_red_M15_slope'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_trend_M15'].values,
            self.df.loc[self.current_step:self.current_step, 'EMAage'].values,
            self.df.loc[self.current_step:self.current_step, 'EMAdelta'].values,
            self.df.loc[self.current_step:self.current_step, 'Market_to_EMA_blue'].values,
            self.df.loc[self.current_step:self.current_step, 'CurrentMarketLevel'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_red_H1'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_blue_H1'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_trend_H1'].values,
            self.df.loc[self.current_step:self.current_step, 'EMAdelta_H1'].values,
            self.df.loc[self.current_step:self.current_step, 'EMAage_H1'].values,
            self.df.loc[self.current_step:self.current_step, 'EMA_red_H1_slope'].values,
            self.df.loc[self.current_step:self.current_step, 'Market_to_EMA_blue_H1'].values,
        ])
        self.CurrentMarketLevel = data_current_step[30]
        output_data = np.append(data_current_step, [[
                    self.active_trade,
                    self.profit
                ]])

        obs = self._normalize_data(output_data)

        return obs

    def _normalize_data(self, input_data):

        norm_data = input_data
        norm_data[0] = norm_data[0] / 100                    # Cycle_12_14_H4_1
        norm_data[1] = norm_data[1] / 100                    # Cycle_12_14_H4_0
        norm_data[2] = (norm_data[2] + 100) / 200                    # CyH4_slope
        norm_data[3] = norm_data[3] / 3000                    # CycleH4age
        norm_data[4] = norm_data[4] / 100                    # Cycle_12_14_H1_1
        norm_data[5] = norm_data[5] / 100                    # Cycle_12_14_H1_0
        norm_data[6] = norm_data[6] / 100                    # ColorStochastic_M5
        norm_data[7] = norm_data[7] / 100                    # ColorStochastic_M15
        norm_data[8] = (norm_data[8] + 100) / 200                    # CyH1_slope
        norm_data[9] = norm_data[9] / 1000                    # CycleH1age
        norm_data[10] = norm_data[10] / 100                    # Cycle_12_14_M15_1
        norm_data[11] = norm_data[11] / 100                    # Cycle_12_14_M15_0
        norm_data[12] = (norm_data[12] + 100) / 200                    # CyM15_slope
        norm_data[13] = norm_data[13]                     # LagN_blue
        norm_data[14] = norm_data[14]                     # LagN_purple
        norm_data[15] = (norm_data[15] + 1) / 2                    # LagN_yellow
        norm_data[16] = (norm_data[16] + 1) / 2                    # LagN_lime
        norm_data[17] = (norm_data[17] + 1) / 2                    # LagN_orange
        norm_data[18] = (norm_data[18] + 1) / 2                    # Last_LagN_yellow
        norm_data[19] = (norm_data[19] + 1) / 2                    # Last_LagN_lime
        norm_data[20] = (norm_data[20] + 1) / 2                    # Last_LagN_orange
        norm_data[21] = (norm_data[21] + 1) / 2                    # LagN_yellow_extreme
        norm_data[22] = (norm_data[22] + 1) / 2                    # LagN_lime_extreme
        norm_data[23] = norm_data[23] / 10                    # EMA_blue_M15
        norm_data[24] = norm_data[24] / 10                    # EMA_red_M15
        norm_data[25] = (norm_data[25] + 50) / 100                    # EMA_red_M15_slope
        norm_data[26] = (norm_data[26] + 1) / 2                    # EMA_trend_M15
        norm_data[27] = norm_data[27] / 3000                    # EMAage
        norm_data[28] = norm_data[28] / 300                    # EMAdelta
        norm_data[29] = norm_data[29] / 500                    # Market_to_EMA_blue
        norm_data[30] = norm_data[30] / 10                    # CurrentMarketLevel
        norm_data[31] = norm_data[31] / 10                    # EMA_red_H1
        norm_data[32] = norm_data[32] / 10                    # EMA_blue_H1
        norm_data[33] = (norm_data[33] + 1) / 2                    # EMA_trend_H1
        norm_data[34] = norm_data[34] / 300                    # EMAdelta_H1
        norm_data[35] = norm_data[35] / 6000                    # EMAage_H1
        norm_data[36] = (norm_data[36] + 20) / 40                    # EMA_red_H1_slope
        norm_data[37] = norm_data[37] / 500                    # Market_to_EMA_blue_H1
        norm_data[38] = norm_data[38] / 2                    # active_trade
        norm_data[39] = norm_data[39] / 1000                    # profit

        return norm_data

    def _close_trade(self):
        if self.active_trade == 2:
            self.profit = (self.trade_open_price - self.CurrentMarketLevel) * 10000
        if self.active_trade == 1:
            self.profit = (self.CurrentMarketLevel - self.trade_open_price) * 10000
        self.close_profit = self.profit
        self.account_balance = self.account_balance + self.profit
        self.profit = 0
        self.active_trade = 0
        self.trade_open_price = 0

    def _take_action(self, action):
        action_type = action

        if action_type == 1 and self.active_trade != 1:       # Buy trade action
            if self.active_trade == 2:
                _close_trade()
            self.active_trade = 1
            self.trade_open_price = self.CurrentMarketLevel

        elif action_type == 2 and self.active_trade == 0:     # Sell trade action
            if self.active_trade == 1:
                _close_trade()
            self.active_trade = 2
            self.trade_open_price == self.CurrentMarketLevel

        elif action_type == 3 and self.active_trade != 0:      # Close trade action
            _close_trade()

        elif action_type == 0:                  # Hold trade action
            self.account_balance = self.account_balance + self.profit

        print(f'action_type = {action} and active_trade = {self.active_trade}')

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Cycle_12_14_H4_1'].values):
            done = true

        done = self.account_balance <= 0
        obs = self._get_current_step_data()
        info = 0

        if self.active_trade != 0:
            self.reward += 0.001

        if self.close_profit > 5:
            self.reward += 5
            self.close_profit = 0
        elif self.close_profit < 0:
            self.reward -= 5
            self.close_profit = 0

        if self.active_trade == 0:
            self.reward -= 0.005

        return obs, self.reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.account_balance = INITIAL_ACCOUNT_BALANCE
        self.profit = 0
        self.trade_open_price = 0
        self.active_trade = 0
        self.close_profit = 0
        self.reward = 0

        # Set the current step to a random point within the data frame
        self.current_step = 0

        return self._get_current_step_data()

    def render(self, mode='human', close=False):

        print(f'Step: {self.current_step}, active trade: {self.active_trade}, profit: {self.profit}, acc balance: {self.account_balance}, trade_open_price: {self.trade_open_price}')
