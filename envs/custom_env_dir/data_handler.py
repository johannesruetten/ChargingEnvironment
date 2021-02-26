import numpy as np
import pandas as pd
import collections
from datetime import datetime
from datetime import timedelta
import os 

''' THIS CLASS HAS MULTIPLE FUNCTIONS FOR DATA LOADING AND STORING '''
class DataHandler(object):
    
    ''' This function splits the data in train/test/dev sets and slices it into "game collections" '''
    def get_data_7d_3split(self, include_weekends, n_episodes, start_year, start_month, start_day):

            n_episodes = n_episodes
            total_game_count = 0
            train_count = 0
            dev_count = 0
            test_count = 0
            c = 0
            start_date = datetime(start_year, start_month, start_day, 12, 0, 0)
            end_date = start_date + timedelta(days=8) + timedelta(minutes=-1)
            # Collection to save sliced data for each game
            train_collection = {}
            dev_collection = {} 
            test_collection = {}
            full_collection = {}
    
            ''' Data is loaded from multiple csv files, since unfortunately there are some issues with certain columns in certain files '''
            df = pd.read_csv('data/data_prices_daycat_2.csv', sep=None, decimal='.', engine='python')
            df.iloc[:, 0] = df.iloc[:, 0].astype(str)
            #df.iloc[:, 0] = df.iloc[:, 0].str.split('.').str[0]
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

            # Get sine and cosine encoding
            df2 = pd.read_csv('data/data_prices_daycat_sincostime_2.csv', sep=None, decimal=',', engine='python')
            df['t_sin'] = df2['sine_h']
            df['t_cos'] = df2['cosine_h']

            # Get weather data (wind speed, minutes of sunshine and temperature)
            df3 = pd.read_csv('data/data_prices_daycat_2_wind.csv', sep=None, decimal='.', engine='python')
            df['wind'] = df3['wind']
            df['sun'] = df3['sun']
            df['temp'] = df3['temp']
            
            # Get time and daycat dummy encoding
            df4 = pd.read_csv('data/data_prices_daycat_2_tencode.csv', sep=None, decimal='.', engine='python')
            df['t0'] = df4['t0']
            df['t1'] = df4['t1']
            df['t2'] = df4['t2']
            df['t3'] = df4['t3']
            df['t4'] = df4['t4']
            df['t5'] = df4['t5']
            df['t6'] = df4['t6']
            df['t7'] = df4['t7']
            df['t8'] = df4['t8']
            df['t9'] = df4['t9']
            df['t10'] = df4['t10']
            df['t11'] = df4['t11']
            df['t12'] = df4['t12']
            df['t13'] = df4['t13']
            df['t14'] = df4['t14']
            df['t15'] = df4['t15']
            df['t16'] = df4['t16']
            df['t17'] = df4['t17']
            df['t18'] = df4['t18']
            df['t19'] = df4['t19']
            df['t20'] = df4['t20']
            df['t21'] = df4['t21']
            df['t22'] = df4['t22']
            df['t23'] = df4['t23']
            df['d1'] = df4['d1']
            df['d2'] = df4['d2']
            df['d3'] = df4['d3']
            df['d4'] = df4['d4']
            df['t_saw1'] = df4['t_saw1']
            df['t_saw2'] = df4['t_saw2']

            # Loop to slice entire dataset into game-sets consisting of 192 hours
            # Thereby the prices from the previous week can be used as input feature
            done = False
            i = 0

            while not done:
                if not include_weekends:
                    # Skip weekends in training and test data
                    if (start_date + timedelta(days=1)).isoweekday() == 6:
                        start_date = start_date + timedelta(days=2)
                        end_date = end_date + timedelta(days=2)

                    if (start_date + timedelta(days=1)).isoweekday() == 7:
                        start_date = start_date + timedelta(days=1)
                        end_date = end_date + timedelta(days=1)

                # Create mask to slice data into 192 consecutive hourly prices
                mask = (df.iloc[:, 0] >= start_date) & (df.iloc[:, 0] <= end_date)

                if df.loc[mask].empty:
                    done = True

                if len(df.loc[mask].index) == 192:
                    new_game = pd.DataFrame(df.loc[mask]).set_index([pd.Index(range(0,192))])

                    # Split data into TRAIN/DEV/TEST data --> in every 30 days: 20 train, 5 dev, 5 test
                    if (total_game_count >= 25 + 30 * c) and (total_game_count < 30 + 30 * c):
                        test_collection[test_count] = new_game
                        test_count += 1
                    elif (total_game_count >= 20 + 30 * c) and (total_game_count < 25 + 30 * c):
                        dev_collection[dev_count] = new_game
                        dev_count += 1
                    else:
                        train_collection[train_count] = new_game
                        train_count += 1

                    # Compile one full collection with all games
                    full_collection[total_game_count] = new_game
                    total_game_count = test_count + train_count + dev_count

                    if total_game_count % 30 == 0:
                        c += 1
                        
                start_date = start_date + timedelta(days=1)
                end_date = end_date + timedelta(days=1)

                # Stop when sufficient games created
                if total_game_count == n_episodes:
                    done = True
                i += 1

            print(total_game_count, ' total games created.')
            print(train_count, ' training games created.')
            print(dev_count, ' development games created.')
            print(test_count, ' test games created.')

            return train_collection, dev_collection, test_collection, train_count, dev_count, test_count, full_collection


    ''' This function stores all the results from training/testing in a CSV file '''
    def store_results(self, price_list, soc_list, action_list, dates, day_cats, \
                      starts, ends, scores, avg_scores, final_soc, eps_history, \
                      pen_history, filename, optimizer, gamma, lr, replace, \
                      store_dir, discounted_action_list,temp_list):
        
        results = pd.DataFrame(
            {'date': dates,
             'day_category': day_cats,
             'start': starts,
             'end': ends,
             'score': scores,
             'avg_score': avg_scores,
             'final_soc': final_soc,
             'eps_history': eps_history,
             'penalty_history': pen_history,
             'price list': price_list,
             'soc list': soc_list,
             'p1':"", 'p2':"", 'p3':"", 'p4':"", 'p5':"", 'p6':"", 
             'p7':"", 'p8':"", 'p9':"", 'p10':"", 'p11':"", 'p12':"",
             'p13':"", 'p14':"", 'p15':"", 'p16':"", 'p17':"", 'p18':"", 
             'p19':"", 'p20':"", 'p21':"", 'p22':"", 'p23':"", 'p24':"",
             'soc1':"", 'soc2':"", 'soc3':"", 'soc4':"", 'soc5':"", 'soc6':"", 
             'soc7':"", 'soc8':"", 'soc9':"", 'soc10':"", 'soc11':"", 'soc12':"",
             'soc13':"", 'soc14':"", 'soc15':"", 'soc16':"", 'soc17':"", 'soc18':"", 
             'soc19':"", 'soc20':"", 'soc21':"", 'soc22':"", 'soc23':"", 'soc24':"",
             'action1':"", 'action2':"", 'action3':"", 'action4':"", 'action5':"", 'action6':"", 
             'action7':"", 'action8':"", 'action9':"", 'action10':"", 'action11':"", 'action12':"",
             'action13':"", 'action14':"", 'action15':"", 'action16':"", 'action17':"", 'action18':"", 
             'action19':"", 'action20':"", 'action21':"", 'action22':"", 'action23':"", 'action24':"",
             'real_a1':"", 'real_a2':"", 'real_a3':"", 'real_a4':"", 'real_a5':"", 'real_a6':"", 
             'real_a7':"", 'real_a8':"", 'real_a9':"", 'real_a10':"", 'real_a11':"", 'real_a12':"",
             'real_a13':"", 'real_a14':"", 'real_a15':"", 'real_a16':"", 'real_a17':"", 'real_a18':"", 
             'real_a19':"", 'real_a20':"", 'real_a21':"", 'real_a22':"", 'real_a23':"", 'real_a24':"",
             'temp1':"", 'temp2':"", 'temp3':"", 'temp4':"", 'temp5':"", 'temp6':"", 
             'temp7':"", 'temp8':"", 'temp9':"", 'temp10':"", 'temp11':"", 'temp12':"",
             'temp13':"", 'temp14':"", 'temp15':"", 'temp16':"", 'temp17':"", 'temp18':"", 
             'temp19':"", 'temp20':"", 'temp21':"", 'temp22':"", 'temp23':"", 'temp24':""
            })
        
        price_array = np.array(price_list)
        soc_array = np.array(soc_list)
        action_array = np.array(action_list)
        discounted_action_array = np.array(discounted_action_list)
        temp_array = np.array(temp_list)
        
        # Store all hourly prices and hourly soc in csv columns
        for i in range(len(price_list[0])):
            results['p'+str(i+1)] = price_array[:,i]
            results['soc'+str(i+1)] = soc_array[:,i]
            results['action'+str(i+1)] = action_array[:,i]
            results['real_a'+str(i+1)] = discounted_action_array[:,i]
            results['temp'+str(i+1)] = temp_array[:,i]

        # datetime object containing current date and time
        now = datetime.now().strftime('%Y%m%d_%H%M')
        cwd = os.getcwd()
        filepath = store_dir + '/' + now + '_' + filename + '_results.csv'
        results.to_csv(filepath, mode='a', index=True)
 
    ''' This function stores benchmark results in a CSV file '''
    def store_benchmark_results(self, price_list, soc_list, action_list, dates, day_cats, \
                                starts, ends, scores, avg_scores, final_soc, eps_history, \
                                pen_history, filename, store_dir):

            results = pd.DataFrame(
                {'date': dates,
                 'day_category': day_cats,
                 'start': starts,
                 'end': ends,
                 'score': scores,
                 'avg_score': avg_scores,
                 'final_soc': final_soc,
                 'eps_history': eps_history,
                 'penalty_history': pen_history,
                 'price list': price_list,
                 'soc list': soc_list,
                 'p1':"", 'p2':"", 'p3':"", 'p4':"", 'p5':"", 'p6':"", 
                 'p7':"", 'p8':"", 'p9':"", 'p10':"", 'p11':"", 'p12':"",
                 'p13':"", 'p14':"", 'p15':"", 'p16':"", 'p17':"", 'p18':"", 
                 'p19':"", 'p20':"", 'p21':"", 'p22':"", 'p23':"", 'p24':"",
                 'soc1':"", 'soc2':"", 'soc3':"", 'soc4':"", 'soc5':"", 'soc6':"", 
                 'soc7':"", 'soc8':"", 'soc9':"", 'soc10':"", 'soc11':"", 'soc12':"",
                 'soc13':"", 'soc14':"", 'soc15':"", 'soc16':"", 'soc17':"", 'soc18':"", 
                 'soc19':"", 'soc20':"", 'soc21':"", 'soc22':"", 'soc23':"", 'soc24':"",
                 'action1':"", 'action2':"", 'action3':"", 'action4':"", 'action5':"", 'action6':"", 
                 'action7':"", 'action8':"", 'action9':"", 'action10':"", 'action11':"", 'action12':"",
                 'action13':"", 'action14':"", 'action15':"", 'action16':"", 'action17':"", 'action18':"", 
                 'action19':"", 'action20':"", 'action21':"", 'action22':"", 'action23':"", 'action24':""
                })

            price_array = np.array(price_list)
            soc_array = np.array(soc_list)
            action_array = np.array(action_list)

            # Store all hourly prices and hourly soc in csv columns
            for i in range(len(price_list[0])):
                results['p'+str(i+1)] = price_array[:,i]
                results['soc'+str(i+1)] = soc_array[:,i]
                results['action'+str(i+1)] = action_array[:,i]

            # datetime object containing current date and time
            now = datetime.now().strftime('%Y%m%d_%H%M')
            cwd = os.getcwd()
            filepath = store_dir + '/' + now + '_' + filename + '_results.csv'
            results.to_csv(filepath, mode='a', index=True)

            print('results stored!')
            
    ''' This function stores a labeled dataset as CSV with only one decision (hour) in each row'''
    def store_supervised_dataset(self, price_list, input_price_list, arr_list, \
                                 soc_list, action_list, dates, day_cats, nextday_cats, \
                                 starts, ends, scores, avg_scores, final_soc, eps_history, \
                                 pen_history, filename, store_dir, t_step, t_sin, t_cos, \
                                 t_saw1, t_saw2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, \
                                 t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, d1, d2, d3, d4):
            
            action_counter = 0
            
            price_array = np.array(price_list)
            input_price_array = np.array(input_price_list)
            soc_array = np.array(soc_list)
            action_array = np.array(action_list)
            
            t_step_array = np.array(t_step)
            t_sin_array = np.array(t_sin)
            t_cos_array = np.array(t_cos)
            t_saw1_array = np.array(t_saw1)
            t_saw2_array = np.array(t_saw2)
            t0_array = np.array(t0)
            t1_array = np.array(t1)
            t2_array = np.array(t2)
            t3_array = np.array(t3)
            t4_array = np.array(t4)
            t5_array = np.array(t5)
            t6_array = np.array(t6)
            t7_array = np.array(t7)
            t8_array = np.array(t8)
            t9_array = np.array(t9)
            t10_array = np.array(t10)
            t11_array = np.array(t11)
            t12_array = np.array(t12)
            t13_array = np.array(t13)
            t14_array = np.array(t14)
            t15_array = np.array(t15)
            t16_array = np.array(t16)
            t17_array = np.array(t17)
            t18_array = np.array(t18)
            t19_array = np.array(t19)
            t20_array = np.array(t20)
            t21_array = np.array(t21)
            t22_array = np.array(t22)
            t23_array = np.array(t23)
            d1_array = np.array(d1)
            d2_array = np.array(d2)
            d3_array = np.array(d3)
            d4_array = np.array(d4)
            
            
            results = []
            
            for d in range(len(dates)):
                h = 0
                t = 12
                t2 = -1
                
                for p in range(len(price_list[0])):
                    
                    if action_array[d,p] == '-':
                        h += 1
                        t += 1
                        continue
                    else:
                        t2 += 1

                    temp = []
                    temp.append(dates[d])
                    temp.append(day_cats[d])
                    temp.append(nextday_cats[d])
                    temp.append(starts[d])
                    temp.append(ends[d])
                    temp.append(t)
                    temp.append(soc_array[d,p])
                    temp.append(p)
                    temp.append(input_price_array[d,arr_list[d]+167+t2])
                    temp.append(input_price_array[d,arr_list[d]+166+t2])
                    temp.append(input_price_array[d,arr_list[d]+165+t2])
                    temp.append(input_price_array[d,arr_list[d]+164+t2])
                    temp.append(input_price_array[d,arr_list[d]+163+t2])
                    temp.append(input_price_array[d,arr_list[d]+162+t2])
                    temp.append(input_price_array[d,arr_list[d]+161+t2])
                    temp.append(input_price_array[d,arr_list[d]+160+t2])
                    temp.append(input_price_array[d,arr_list[d]+159+t2])
                    temp.append(input_price_array[d,arr_list[d]+158+t2])
                    temp.append(input_price_array[d,arr_list[d]+157+t2])
                    temp.append(input_price_array[d,arr_list[d]+156+t2])
                    temp.append(input_price_array[d,arr_list[d]+155+t2])
                    temp.append(input_price_array[d,arr_list[d]+154+t2])
                    temp.append(input_price_array[d,arr_list[d]+153+t2])
                    temp.append(input_price_array[d,arr_list[d]+152+t2])
                    temp.append(input_price_array[d,arr_list[d]+151+t2])
                    temp.append(input_price_array[d,arr_list[d]+150+t2])
                    temp.append(input_price_array[d,arr_list[d]+149+t2])
                    temp.append(input_price_array[d,arr_list[d]+148+t2])
                    temp.append(input_price_array[d,arr_list[d]+147+t2])
                    temp.append(input_price_array[d,arr_list[d]+146+t2])
                    temp.append(input_price_array[d,arr_list[d]+145+t2])
                    temp.append(input_price_array[d,arr_list[d]+144+t2])
                    temp.append(input_price_array[d,arr_list[d]+120+t2])
                    temp.append(input_price_array[d,arr_list[d]+23+t2])
                    temp.append(input_price_array[d,arr_list[d]+22+t2])
                    temp.append(input_price_array[d,arr_list[d]+21+t2])
                    temp.append(input_price_array[d,arr_list[d]+20+t2])
                    temp.append(input_price_array[d,arr_list[d]+19+t2])
                    temp.append(input_price_array[d,arr_list[d]+18+t2])
                    temp.append(input_price_array[d,arr_list[d]+17+t2])
                    temp.append(input_price_array[d,arr_list[d]+16+t2])
                    temp.append(input_price_array[d,arr_list[d]+15+t2])
                    temp.append(input_price_array[d,arr_list[d]+14+t2])
                    temp.append(input_price_array[d,arr_list[d]+13+t2])
                    temp.append(input_price_array[d,arr_list[d]+12+t2])
                    temp.append(input_price_array[d,arr_list[d]+11+t2])
                    temp.append(input_price_array[d,arr_list[d]+10+t2])
                    temp.append(input_price_array[d,arr_list[d]+9+t2])
                    temp.append(input_price_array[d,arr_list[d]+8+t2])
                    temp.append(input_price_array[d,arr_list[d]+7+t2])
                    temp.append(input_price_array[d,arr_list[d]+6+t2])
                    temp.append(input_price_array[d,arr_list[d]+5+t2])
                    temp.append(input_price_array[d,arr_list[d]+4+t2])
                    temp.append(input_price_array[d,arr_list[d]+3+t2])
                    temp.append(input_price_array[d,arr_list[d]+2+t2])
                    temp.append(input_price_array[d,arr_list[d]+1+t2])
                    temp.append(input_price_array[d,arr_list[d]+t2])

                    temp.append(action_array[d,p])
                    
                    if float(action_array[d,p]) > 0:
                        temp.append(1)
                        temp.append(0)
                    elif float(action_array[d,p]) < 0:
                        temp.append(-1)
                        temp.append(1)
                    else:
                        temp.append(0)
                        temp.append(2)
                        
                    temp.append(soc_array[d,p+1])
                    
                    temp.append(t_step_array[d,p])
                    temp.append(t_sin_array[d,p])
                    temp.append(t_cos_array[d,p])
                    temp.append(t_saw1_array[d,p])
                    temp.append(t_saw2_array[d,p])
                    temp.append(t0_array[d,p])
                    temp.append(t1_array[d,p])
                    temp.append(t2_array[d,p])
                    temp.append(t3_array[d,p])
                    temp.append(t4_array[d,p])
                    temp.append(t5_array[d,p])
                    temp.append(t6_array[d,p])
                    temp.append(t7_array[d,p])
                    temp.append(t8_array[d,p])
                    temp.append(t9_array[d,p])
                    temp.append(t10_array[d,p])
                    temp.append(t11_array[d,p])
                    temp.append(t12_array[d,p])
                    temp.append(t13_array[d,p])
                    temp.append(t14_array[d,p])
                    temp.append(t15_array[d,p])
                    temp.append(t16_array[d,p])
                    temp.append(t17_array[d,p])
                    temp.append(t18_array[d,p])
                    temp.append(t19_array[d,p])
                    temp.append(t20_array[d,p])
                    temp.append(t21_array[d,p])
                    temp.append(t22_array[d,p])
                    temp.append(t23_array[d,p])
                    temp.append(d1_array[d,p])
                    temp.append(d2_array[d,p])
                    temp.append(d3_array[d,p])
                    temp.append(d4_array[d,p])
                        
                    results.append(temp)
                    
                    t += 1                   
                    action_counter =+ 1
                    
                    if t == 24:
                        t = 0
            
            results = np.array(results)
            results_df = pd.DataFrame(results, columns=['date','day_cat','nextday_cat','start','end',\
                                                        'hour','soc_in','step','p-1','p-2','p-3','p-4',\
                                                        'p-5','p-6','p-7','p-8','p-9','p-10','p-11','p-12', \
                                                        'p-13','p-14','p-15','p-16','p-17','p-18','p-19','p-20',\
                                                        'p-21','p-22','p-23','p-24','p-48','p-145','p-146','p-147',\
                                                        'p-148','p-149','p-150','p-151','p-152','p-153','p-154',\
                                                        'p-155','p-156','p-157','p-158','p-159','p-160','p-161',\
                                                        'p-162','p-163','p-164','p-165','p-166','p-167','p-168',\
                                                        'action','vis_action','env_action','soc_out','t_step','t_sin',\
                                                        't_cos','t_saw1','t_saw2','t0','t1','t2','t3','t4',\
                                                        't5','t6','t7','t8','t9','t10','t11','t12','t13','t14',\
                                                        't15','t16','t17','t18','t19','t20','t21','t22','t23','d1',\
                                                        'd2','d3','d4',])
            
            # datetime object containing current date and time
            now = datetime.now().strftime('%Y%m%d_%H%M')
            cwd = os.getcwd()
            filepath = store_dir + '/' + now + '_' + filename + '_dataset.csv'

            results_df.to_csv(filepath, mode='a', index=True)

            print('Labeled dataset stored!')