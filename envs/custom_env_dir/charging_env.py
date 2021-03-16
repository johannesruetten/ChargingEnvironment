import gym
import numpy as np
import itertools
import pandas as pd
import scipy.stats as stats
import random
from datetime import timedelta

class ChargingEnv(gym.Env):

    # Method initializes the Environment
    def __init__(self, game_collection, battery_capacity, charging_rate, penalty_coefficient, obs):
        
        self.step_length = 1
        self.parking = 0
        self.game_collection = game_collection
        self.battery_capacity = battery_capacity     
        self.charging_rate = charging_rate
        self.penalty_coefficient = penalty_coefficient
        self.pen_dec = 0.0001
        self.pen_min = 1
        self.discounted_action = 0
        self.obs = obs
        
        # Defines action space of the MDP
        self.action_space = {0: +self.charging_rate,
                             1: -self.charging_rate,
                             2: 0
                           }
        
        # In case you want to print the meaning of actions somewhere
        self.action_map = {0: 'Charging',
                           1: 'Discharging',
                           2: 'None'
                          }

        print('Env initialized!')

    # Executes a step while the vehicle is not parked at home
    def non_parking_step(self):
        reward = 0
        self.t_step += 1
        
        # Check if vehicle arrives in the next episode
        if self.t_step == self.start_ind:
            self.parking = 1
        
        # No data available for 25th step
        if self.t_step == 24:
            observation_ = []
        else:            
            ''' ---------- Set new observation ---------- '''
            if self.obs == 'obs2(t_sin,t_cos)':
                observation_ = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]]
            elif self.obs == 'obs25(soc,p-23-p)':
                observation_ = [self.soc] + self.hourly_prices['Spot'].tolist()[self.t_step+145:self.t_step+169]
            elif self.obs == 'obs26(t_step,soc,p-23-p)':
                observation_ = [self.t_step] + [self.soc] + self.hourly_prices['Spot'].tolist()[self.t_step+145:self.t_step+169]
            elif self.obs == 'obs4(t_sin,t_cos,daycat,temp0)':
                observation_ = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]] + [self.hourly_prices['day_cat'][self.t_step+168]] + [self.hourly_prices['temp'][self.t_step+168]]
            elif self.obs == 'obs3(t_sin,t_cos,is_we)':
                observation_ = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]] + [self.is_we]
            elif self.obs == 'obs2(t_step,daycat)':
                observation_ = [self.t_step] + [self.hourly_prices['day_cat'][self.t_step+168]]
            else:
                print('INPUT FEATURE COMBINATION NOT SPECIFIED IN THE ENVIRONMENT!')
                
        
        return observation_, reward
        
    # Executes a step and calculates associated reward
    def step(self, action):

        # If t_step+1 is the last time step -> include penalty term if vehicle is not fully charged
        if self.t_step+1 == self.end_ind:

            # Vehicle can not be discharged below 0% or charged above 100% -> Calculate discounted charging rate
            if (self.soc + (self.action_space[action]/self.battery_capacity)) < 0 or \
            (self.soc + (self.action_space[action]/self.battery_capacity)) > 1:
                
                # Calculate proportion of the hour until vehicle completely charged/discharged
                if self.action_space[action]<0:
                    discount = self.soc/(self.charging_rate/self.battery_capacity)
                elif self.action_space[action]>0:
                    discount = (1-self.soc)/(self.charging_rate/self.battery_capacity)

                # Calculate new SOC and respective reward
                self.soc = round(self.soc + (discount*self.action_space[action]/self.battery_capacity), 2)
                reward = discount * (-1) * self.step_length * self.action_space[action] \
                            * self.hourly_prices['Spot'][self.t_step+168]/10 \
                            - self.penalty_coefficient * ((1-self.soc)*self.battery_capacity)**2
                
                # Store that discounted action to save it
                self.discounted_action = discount * self.action_space[action]

            # Standard case with full charging/discharging actions possible
            else:
                # Calculate new SOC and respective reward
                self.soc = round(self.soc + (self.action_space[action]/self.battery_capacity), 2)
                reward = (-1) * self.step_length * self.action_space[action] * self.hourly_prices['Spot'][self.t_step+168]/10 \
                            - self.penalty_coefficient * ((1-self.soc)*self.battery_capacity)**2
                
                # Store that discounted action to save it
                self.discounted_action = self.action_space[action]

        # Standard case, when t_step is NOT the final time step
        else:
            # Vehicle can not be discharged below 0% or charged above 100% -> Calculate proportion
            if (self.soc + (self.action_space[action]/self.battery_capacity)) < 0 or \
            (self.soc + (self.action_space[action]/self.battery_capacity)) > 1:
            
                # Calculate proportion of the hour until vehicle completely charged/discharged
                if self.action_space[action]<0:
                    discount = self.soc/(self.charging_rate/self.battery_capacity)
                elif self.action_space[action]>0:
                    discount = (1-self.soc)/(self.charging_rate/self.battery_capacity)

                # Calculate new SOC and respective reward
                self.soc = round(self.soc + (discount*self.action_space[action]/self.battery_capacity), 2)
                reward = discount * (-1) * self.step_length * self.action_space[action] \
                            * self.hourly_prices['Spot'][self.t_step+168]/10
                
                self.discounted_action = discount * self.action_space[action]

            # Standard case with full charging/discharging actions possible
            else:
                # Calculate new SOC and respective reward
                self.soc = round(self.soc + (self.action_space[action]/self.battery_capacity), 2)
                reward = (-1) * self.step_length * self.action_space[action] * self.hourly_prices['Spot'][self.t_step+168]/10
                
                # Store that discounted action to save it
                self.discounted_action = self.action_space[action]

        self.t_step += 1
        
        # Get day category from respective column in the dataset
        self.day_cat = self.hourly_prices['day_cat'][self.t_step+168]
        if self.day_cat > 1:
            self.is_we = 1
        elif self.day_cat == 1:
            self.is_we = 0

        # Check if vehicle leaves in the next episode
        if self.t_step == self.end_ind:
            self.parking = 0
        
        ''' ---------- Set new observation ---------- '''
        if self.obs == 'obs2(t_sin,t_cos)':
            observation_ = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]]
        elif self.obs == 'obs25(soc,p-23-p)':
            observation_ = [self.soc] + self.hourly_prices['Spot'].tolist()[self.t_step+145:self.t_step+169]
        elif self.obs == 'obs26(t_step,soc,p-23-p)':
            observation_ = [self.t_step] + [self.soc] + self.hourly_prices['Spot'].tolist()[self.t_step+145:self.t_step+169]
        elif self.obs == 'obs4(t_sin,t_cos,daycat,temp0)':
            observation_ = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]] + [self.hourly_prices['day_cat'][self.t_step+168]] + [self.hourly_prices['temp'][self.t_step+168]]
        elif self.obs == 'obs3(t_sin,t_cos,is_we)':
            observation_ = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]] + [self.is_we]
        elif self.obs == 'obs2(t_step,daycat)':
            observation_ = [self.t_step] + [self.hourly_prices['day_cat'][self.t_step+168]]
        else:
            print('INPUT FEATURE COMBINATION NOT SPECIFIED IN THE ENVIRONMENT!')
        
        return observation_, reward
    
    # Returns initial observation for each game
    def reset(self, test, i):
        
        # Set initial time step to 0
        self.t_step = 0
        
        if test:
            # Iterate through all test days once
            self.hourly_prices = self.game_collection[i//10]
            # Ensure comparability of test results by adding a seed to fix sampled arr & dep times
            np.random.seed(seed=1+i*11)
        else:
            # Randomly select days
            self.hourly_prices = random.choice(self.game_collection) 
        
        # Store date
        self.game_date = self.hourly_prices.iloc[168,0].date()
        
        # Get day category from respective column in the dataset
        self.day_cat = self.hourly_prices['day_cat'][self.t_step+168]
        if self.day_cat > 1:
            self.is_we = 1
        elif self.day_cat == 1:
            self.is_we = 0
        
        
        # Sample different arrival time and soc depending on this day's category
        if self.day_cat > 1:
            self.start_time, self.start_ind = self.arr_pattern_h_we()
            self.soc = self.soc_we()
            if self.start_ind == 0:
                self.parking = 1
        else:
            self.start_time, self.start_ind = self.arr_pattern_h_wd()
            self.soc = self.soc_wd()
        
        # Sample different departure time depending on the next day's category
        if self.hourly_prices['day_cat'][self.t_step+191] > 1:
            self.end_time, self.end_ind = self.dep_pattern_h_we()
        else:
            self.end_time, self.end_ind = self.dep_pattern_h_wd()
        
        self.n_steps = self.end_ind - self.start_ind
        
        ''' ---------- Set initial observation ---------- '''
        if self.obs == 'obs2(t_sin,t_cos)':
            observation = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]]
        elif self.obs == 'obs25(soc,p-23-p)':
            observation = [self.soc] + self.hourly_prices['Spot'].tolist()[self.t_step+145:self.t_step+169]
        elif self.obs == 'obs26(t_step,soc,p-23-p)':
            observation = [self.t_step] + [self.soc] + self.hourly_prices['Spot'].tolist()[self.t_step+145:self.t_step+169]
        elif self.obs == 'obs4(t_sin,t_cos,daycat,temp0)':
            observation = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]] + [self.hourly_prices['day_cat'][self.t_step+168]] + [self.hourly_prices['temp'][self.t_step+168]]
        elif self.obs == 'obs3(t_sin,t_cos,is_we)':
            observation = [self.hourly_prices['t_sin'][self.t_step+168]] + [self.hourly_prices['t_cos'][self.t_step+168]] + [self.is_we]
        elif self.obs == 'obs2(t_step,daycat)':
            observation = [self.t_step] + [self.hourly_prices['day_cat'][self.t_step+168]]
        elif self.obs == 'benchmark':
            observation = []
        else:
            observation = []
            print('INPUT FEATURE COMBINATION NOT SPECIFIED IN THE ENVIRONMENT!')
        
        return observation
    
    # Allows you to use a declining penalty term
    def decrement_penalty(self, pen_dec, pen_min):
        self.pen_dec = pen_dec
        self.pen_min = pen_min
        self.penalty_coefficient = self.penalty_coefficient - self.pen_dec \
                           if self.penalty_coefficient > self.pen_min else self.pen_min
    

    # Method samples random WEEKDAY HOURLY ARRIVAL time from truncated gaussian distribution
    def arr_pattern_h_wd(self):
        
        # Initialize list of discrete time steps in 60min intervals starting with index 0 at noon (12:00 p.m.)
        self.time_range = pd.date_range(start='12:00:00', periods=24, freq='1h').strftime('%H:%M')
        
        # Sample index for arrival time distribution: low-15:00, high-21:00, center=18:00, dev=1h
        a_lower, a_upper = 3, 9
        a_mu, a_sigma = 6, 1
        random_arrival_index = int(round(stats.truncnorm.rvs((a_lower - a_mu) / a_sigma, (a_upper - a_mu) \
                                                             / a_sigma, loc=a_mu, scale=a_sigma)))
        start_time = self.time_range[random_arrival_index]
        
        return start_time, random_arrival_index
    
    # Method samples random WEEKDAY HOURLY DEPARTURE time from truncated gaussian distribution
    def dep_pattern_h_wd(self):
        
        # Initialize list of discrete time steps in 60min intervals starting with index 0 at noon (12:00 p.m.)
        self.time_range = pd.date_range(start='12:00:00', periods=24, freq='1h').strftime('%H:%M')
        
        # Sample index for departure time distribution: low-06:00, high-11:00, center=08:00, dev=1h
        d_lower, d_upper = 18, 23
        d_mu, d_sigma = 20, 1
        random_departure_index = int(round(stats.truncnorm.rvs((d_lower - d_mu) / d_sigma, (d_upper - d_mu) \
                                                               / d_sigma, loc=d_mu, scale=d_sigma)))
        end_time = self.time_range[random_departure_index]
        
        return end_time, random_departure_index
    
    # Method samples random WEEKEND HOURLY SOC on arrival from truncated gaussian distribution
    def soc_wd(self):
        
        # Sample index for state of charge (SOC) distribution
        soc_lower, soc_upper = 20, 80
        soc_mu, soc_sigma = 50, 10
        # Discretize SOC range to .01 steps
        self.soc_discrete = np.linspace(0,1,101,endpoint=True)
        random_soc_index = int(round(stats.truncnorm.rvs((soc_lower - soc_mu) / soc_sigma, (soc_upper - soc_mu) \
                                                         / soc_sigma, loc=soc_mu, scale=soc_sigma),2))
        soc = self.soc_discrete[random_soc_index]
        
        return soc
    
    # Method samples random WEEKEND HOURLY ARRIVAL time from truncated gaussian distribution
    def arr_pattern_h_we(self):
        
        # Initialize list of discrete time steps in 60min intervals starting with index 0 at noon (12:00 p.m.)
        self.time_range = pd.date_range(start='12:00:00', periods=24, freq='1h').strftime('%H:%M')
        
        # Sample index for arrival time distribution: low-13:00, high-19:00, center=16:00, dev=1h
        a_lower, a_upper = 1, 7
        a_mu, a_sigma = 4, 1
        random_arrival_index = int(round(stats.truncnorm.rvs((a_lower - a_mu) / a_sigma, (a_upper - a_mu) \
                                                             / a_sigma, loc=a_mu, scale=a_sigma)))
        start_time = self.time_range[random_arrival_index]
        
        return start_time, random_arrival_index
    
    # Method samples random WEEKEND HOURLY DEPARTURE time from truncated gaussian distribution
    def dep_pattern_h_we(self):
        
        # Initialize list of discrete time steps in 60min intervals starting with index 0 at noon (12:00 p.m.)
        self.time_range = pd.date_range(start='12:00:00', periods=24, freq='1h').strftime('%H:%M')
        
        # Sample index for departure time distribution: low-09:00, high-11:00, center=10:00, dev=1h
        d_lower, d_upper = 21, 23
        d_mu, d_sigma = 22, 1
        random_departure_index = int(round(stats.truncnorm.rvs((d_lower - d_mu) / d_sigma, (d_upper - d_mu) \
                                                               / d_sigma, loc=d_mu, scale=d_sigma)))
        end_time = self.time_range[random_departure_index]
        
        return end_time, random_departure_index
    
    # Method samples random WEEKEND HOURLY SOC on arrival from truncated gaussian distribution
    def soc_we(self):
        
        # Sample index for state of charge (SOC) distribution
        soc_lower, soc_upper = 20, 80
        soc_mu, soc_sigma = 50, 10
        # Discretize SOC range to .01 steps
        self.soc_discrete = np.linspace(0,1,101,endpoint=True)
        random_soc_index = int(round(stats.truncnorm.rvs((soc_lower - soc_mu) / soc_sigma, (soc_upper - soc_mu) \
                                                         / soc_sigma, loc=soc_mu, scale=soc_sigma),2))
        soc = self.soc_discrete[random_soc_index]
        
        return soc