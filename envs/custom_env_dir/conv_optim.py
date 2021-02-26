import cvxpy as cvx
import numpy as np
import mosek
from envs.custom_env_dir.data_handler import DataHandler
import gym
import os

''' CALCULATE THEORETICAL OPTIMUM BY MEANS OF CONVEX OPTIMIZATION ASSUMING COMPLETE KNOWLEDGE OF FUTURE DATA '''
class ConvOptim():  

    def run_optimizer(self, store_dir, benchmark, supervised_training_set, game_collection, supervised_test_set, development):
        
        obs = ''
        # Initialize charging environment with given EV data
        env = gym.make('ChargingEnv-v0', game_collection=game_collection,
                        battery_capacity=24, charging_rate=6,
                        penalty_coefficient=12, obs)
    
        # Sample each day 10 times for benchmark and test set
        if benchmark:
            n_games = len(game_collection)*10
            test= True
            if development:
                filename = 'Theoretical_limit_DEV'
            else:
                filename = 'Theoretical_limit_TEST'
        elif supervised_test_set:
            test = True
            n_games = len(game_collection)*10
            filename = 'Supervised_test'
        elif supervised_training_set:
            n_games = 50000 #für imitation learning dataset hochgesetzt
            filename = 'Supervised_train'

        # Create lists to store optimization results
        price_list, input_price_list, arr_list, soc_list, action_list, dates, day_cats, \
        nextday_cats, starts, ends, scores, avg_scores, eps_history, pen_history, steps_array, \
        final_soc, t_step, t_sin, t_cos, t_saw1, t_saw2 , t0, t1, t2, t3, t4, t5, t6, t7, t8, \
        t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, d1, d2, d3, d4 \
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(n_games):

            # Create lists to store optimization results for each episode/game
            episode_prices, episode_input_prices, episode_soc, episode_actions, episode_day_cats, \
            episode_nextday_cats, calc_actions, episode_t_step, episode_t_sin, episode_t_cos, \
            episode_t_saw1, episode_t_saw2 , episode_t0, episode_t1, episode_t2, episode_t3, \
            episode_t4, episode_t5, episode_t6, episode_t7, episode_t8, episode_t9, episode_t10, \
            episode_t11, episode_t12, episode_t13, episode_t14, episode_t15, episode_t16, episode_t17, \
            episode_t18, episode_t19, episode_t20, episode_t21, episode_t22, episode_t23, \
            episode_d1, episode_d2, episode_d3, episode_d4 \
            = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            
            # Get initial data from the environment
            reward=0
            observation = env.reset(test, i)
            score = 0
            initial_soc = int(round(env.soc*100,0))
            steps = env.n_steps
            prices = env.hourly_prices['Spot'][env.start_ind+168:env.end_ind+168].to_list()

            # Define objective function
            action = cvx.Variable(steps, integer=True)
            soc = cvx.Variable(steps, integer=True)
            objective = cvx.Minimize(prices@action)

            # Define constraints
            constraints = [action >=-25,
                           action <=25,
                           soc == initial_soc + cvx.cumsum(action),
                           soc>=0,
                           soc<=100,
                           soc[steps-1]==100]

            # Define problem
            prob = cvx.Problem(objective, constraints)

            # Solve problem with the selected solver
            prob.solve(solver=cvx.MOSEK)
            
            

            '''
            _____________________________________________________
            
            Store all necessary data in a CSV file for evaluation
            _____________________________________________________

            '''
            episode_prices = env.hourly_prices['Spot'][168:192].to_list()
            episode_day_cats = env.hourly_prices['day_cat'][168]
            episode_nextday_cats = env.hourly_prices['day_cat'][191]

            episode_input_prices = env.hourly_prices['Spot'].to_list()
            episode_soc.append(initial_soc/100)
            
            episode_t_step = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
            episode_t_sin = env.hourly_prices['t_sin'][168:192].to_list()
            episode_t_cos = env.hourly_prices['t_cos'][168:192].to_list()
            episode_t_saw1 = env.hourly_prices['t_saw1'][168:192].to_list()
            episode_t_saw2 = env.hourly_prices['t_saw2'][168:192].to_list()
            episode_t0 = env.hourly_prices['t0'][168:192].to_list()
            episode_t1 = env.hourly_prices['t1'][168:192].to_list()
            episode_t2 = env.hourly_prices['t2'][168:192].to_list()
            episode_t3 = env.hourly_prices['t3'][168:192].to_list()
            episode_t4 = env.hourly_prices['t4'][168:192].to_list()
            episode_t5 = env.hourly_prices['t5'][168:192].to_list()
            episode_t6 = env.hourly_prices['t6'][168:192].to_list()
            episode_t7 = env.hourly_prices['t7'][168:192].to_list()
            episode_t8 = env.hourly_prices['t8'][168:192].to_list() 
            episode_t9 = env.hourly_prices['t9'][168:192].to_list()
            episode_t10 = env.hourly_prices['t10'][168:192].to_list()
            episode_t11 = env.hourly_prices['t11'][168:192].to_list()
            episode_t12 = env.hourly_prices['t12'][168:192].to_list()
            episode_t13 = env.hourly_prices['t13'][168:192].to_list()
            episode_t14 = env.hourly_prices['t14'][168:192].to_list()
            episode_t15 = env.hourly_prices['t15'][168:192].to_list()
            episode_t16 = env.hourly_prices['t16'][168:192].to_list()
            episode_t17 = env.hourly_prices['t17'][168:192].to_list()
            episode_t18 = env.hourly_prices['t18'][168:192].to_list()
            episode_t19 = env.hourly_prices['t19'][168:192].to_list()
            episode_t20 = env.hourly_prices['t20'][168:192].to_list()
            episode_t21 = env.hourly_prices['t21'][168:192].to_list()
            episode_t22 = env.hourly_prices['t22'][168:192].to_list()
            episode_t23 = env.hourly_prices['t23'][168:192].to_list()
            episode_d1 = env.hourly_prices['d1'][168:192].to_list()
            episode_d2 = env.hourly_prices['d2'][168:192].to_list()
            episode_d3 = env.hourly_prices['d3'][168:192].to_list()
            episode_d4 = env.hourly_prices['d4'][168:192].to_list()

            for j in range(0,env.start_ind):
                episode_soc.append(initial_soc/100)
                episode_actions.append('-')
                calc_actions.append(0)

            for j in range(0,(env.end_ind-env.start_ind)):
                episode_soc.append(soc[j].value/100)
                episode_actions.append((action[j].value/100)*24)
                calc_actions.append((action[j].value/100)*24)

            for j in range(24-env.end_ind):
                episode_soc.append(soc[(env.end_ind-env.start_ind-1)].value/100)
                episode_actions.append('-')
                calc_actions.append(0)

            price_array = np.array(episode_prices)/10
            action_array = np.array(calc_actions)
            episode_rewards = price_array*action_array*(-1)
            
            if i%100==0:
                print('episode: ', i, ' reward: ', sum(episode_rewards))

            price_list.append(episode_prices)
            arr_list.append(env.start_ind)
            input_price_list.append(episode_input_prices)
            soc_list.append(episode_soc)
            action_list.append(episode_actions)
            dates.append(env.game_date)
            day_cats.append(episode_day_cats)
            nextday_cats.append(episode_nextday_cats)
            starts.append(env.start_time)
            ends.append(env.end_time)
            final_soc.append(1)
            scores.append(sum(episode_rewards))
            eps_history.append('-')
            pen_history.append('-')
            avg_scores.append('-')
            
            t_step.append(episode_t_step)
            t_sin.append(episode_t_sin)
            t_cos.append(episode_t_cos)
            t_saw1.append(episode_t_saw1)
            t_saw2.append(episode_t_saw2)
            t0.append(episode_t0)
            t1.append(episode_t1)
            t2.append(episode_t2)
            t3.append(episode_t3)
            t4.append(episode_t4)
            t5.append(episode_t5)
            t6.append(episode_t6)
            t7.append(episode_t7)
            t8.append(episode_t8)
            t9.append(episode_t9)
            t10.append(episode_t10)
            t11.append(episode_t11)
            t12.append(episode_t12)
            t13.append(episode_t13)
            t14.append(episode_t14)
            t15.append(episode_t15)
            t16.append(episode_t16)
            t17.append(episode_t17)
            t18.append(episode_t18)
            t19.append(episode_t19)
            t20.append(episode_t20)
            t21.append(episode_t21)
            t22.append(episode_t22)
            t23.append(episode_t23)
            d1.append(episode_d1)
            d2.append(episode_d2)
            d3.append(episode_d3)
            d4.append(episode_d4)

        # benchark is stored with one entire episode summarized in one row
        if benchmark:
            DataHandler().store_benchmark_results(price_list, soc_list, action_list, dates, day_cats, starts, ends, scores, avg_scores, \
                                    final_soc, eps_history, pen_history, filename, store_dir)
        
        # labeled dataset is stored with single charging/discarging decisions in one row
        elif supervised_training_set or supervised_test_set:
            DataHandler().store_supervised_dataset(price_list, input_price_list, arr_list, soc_list, action_list, dates, \
                                    day_cats, nextday_cats, starts, ends, scores, \
                                    avg_scores, final_soc, eps_history, pen_history, filename, store_dir, \
                                    t_step, t_sin, t_cos, t_saw1, t_saw2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, \
                                    t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, d1, d2, d3, d4)
        else:
            print('nothing stored...')