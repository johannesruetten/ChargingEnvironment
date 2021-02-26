import numpy as np
from numpy import genfromtxt
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Pandas FutureWarning caused by pd.DataFrame


class SupModel():

    def load_data(self, train_data, test_data, in_col, out_col):
        cwd = os.getcwd()
        #First, the data is loaded from the CSV file
        x_train = pd.read_csv(train_data, delimiter=";")[in_col].to_numpy()
        y_train = np.ravel(pd.read_csv(train_data, delimiter=";")[out_col].to_numpy())
        
        x_test = pd.read_csv(test_data, delimiter=";")[in_col].to_numpy()
        y_test = np.ravel(pd.read_csv(test_data, delimiter=";")[out_col].to_numpy())
        
        print('Data loaded...')        
        return x_train, y_train, x_test, y_test

    def scale_data(self, store_dir, x_train, x_test):
        #The input data is normalized by the StandardScaler
        #Fitted only to the training data, since the distribution of the test data should remain unknown to the system
        self.scaler = StandardScaler()
        self.scaler.fit(x_train)

        #In a second step, the normalization is applied to both the test and training data
        x_train = self.scaler.transform(x_train)
        x_test = self.scaler.transform(x_test)
        filepath2 = store_dir + '/scaler.sav'
        joblib.dump(self.scaler, filepath2)
        print('Data scaled...')
        return x_train, x_test
        
    def train_model_kneighbors(self, store_dir, k, x_train, y_train, x_test, y_test):
        # create folder to store model and scaler
        os.makedirs(store_dir)
        # Scale data and store scaler
        x_train, x_test = self.scale_data(store_dir, x_train, x_test)
        
        # K-nearest Neighbors Classifier
        model = KNeighborsClassifier(n_neighbors=k) 
        model.fit(x_train,y_train.ravel())

        # save the model to disk
        model_name = 'KNeighbors_' #type(model).__name__
        model_param = 'k(' + str(k) + ')' 
        filename = model_name + model_param 
        filepath = store_dir + '/' + filename + '_finalized_model.sav'
        joblib.dump(model, filepath)
        print('K-nearest neighbors classifier stored...')
        
    def load_model_kneighbors(self, store_dir, k):
        # load the model from disk
        model_name = 'KNeighbors_'
        model_param = 'k(' + str(k) + ')'  
        filename = model_name + model_param 
        filepath = store_dir + '/' + filename + '_finalized_model.sav'
        model = joblib.load(filepath)
        print('K-nearest neighbors classifier loaded...')
        return model
        
    def train_model_mlp(self, store_dir, hl, af, sl, x_train, y_train, x_test, y_test):
        # create folder to store model and scaler
        os.makedirs(store_dir)
        # Scale data and store scaler
        x_train, x_test = self.scale_data(store_dir, x_train, x_test)

        #The MLP is trained using the training data via backward propagation
        #The hyperparameters are passed and the random seed is kept constant for reproducible results
        # MLP Classifier
        model = MLPClassifier(hidden_layer_sizes=hl, activation=af, solver=sl, max_iter=100, verbose=False, random_state=2)  
        model.fit(x_train,y_train.ravel())
        
        # save the model to disk
        model_name = 'MLP_' #type(model).__name__
        model_param = 'hl(' + str(hl) + ')_af(' + str(af) + ')_sl(' + str(sl) + ')'
        filename = model_name + model_param 
        filepath = store_dir + '/' + filename + '_finalized_model.sav'
        joblib.dump(model, filepath)
        filepath2 = store_dir + '/scaler.sav'
        joblib.dump(self.scaler, filepath2)
        print('MLP classifier stored...')

    def load_model_mlp(self, store_dir, hl, af, sl):
        # load the model from disk
        model_name = 'MLP_'
        model_param = 'hl(' + str(hl) + ')_af(' + str(af) + ')_sl(' + str(sl) + ')'  
        filename = model_name + model_param
        filepath = store_dir + '/' + filename + '_finalized_model.sav'
        model = joblib.load(filepath)
        print('MLP classifier loaded...')     
        return model

    
    def load_scaler(self, store_dir):   
        # load the scaler from disk
        filepath = store_dir + '/scaler.sav'
        scaler = joblib.load(filepath)
        print('Scaler loaded...')
        return scaler

        
    def test_model(self, model, store_dir, x_test, y_test):
        model = model
        x_test = self.load_scaler(store_dir).transform(x_test)
        
        #The KNN provides an output (prediction) to the test data
        predictions = model.predict(x_test)

        #R^2 is already used as an internal scoring function of the MLPRegressor class
        #other quality measures (MAPE and MAE) are calculated here
        avg_absolute_error = np.empty
        avg_r2 = np.empty
        avg_absolute_percentage_error = np.empty
        avg_r2 = np.append(avg_r2, model.score(x_test,y_test))
        avg_absolute_error = np.append(avg_absolute_error, mean_absolute_error(y_test, predictions))
        print('MAE: ', mean_absolute_error(y_test, predictions),'R2: ', model.score(x_test,y_test))