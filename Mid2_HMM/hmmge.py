#https://waterprogramming.wordpress.com/2018/07/03/fitting-hidden-markov-models-part-ii-sample-python-script/
#https://rdrr.io/cran/seqHMM/man/plot.hmm.html
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from math import log10

def load_csv(file):
    return pd.read_csv(file)

def plotTimeSerie(time, visible, hidden):
    sns.lineplot(data=hidden, color="coral", label="hidden")
    ax2 = plt.twinx()
    sns.scatterplot(data=visible, color="blue", label="visible", ax=ax2)
#    plt.xticks(np.arange(len(time)), time)
#    plt.xticks(rotation=180)
    plt.show()

def split_train_test(data, ratio):
    split_index = int(len(data)*ratio)
    print('split_index :'+str(split_index))
    return data[0:split_index], data[split_index+1::]

def main():
    data = load_csv('./dataset/energydata_complete.csv')
    ratio = 20/21 
    sample_size = 400
    hidden_states_count = 4

    #Reshaping dat
    train_lights, test_lights = split_train_test(data['lights'].values.reshape(-1, 1), ratio)
    train_appliances, test_appliances = split_train_test(data['Appliances'].values.reshape(-1, 1), ratio)
    train_time, test_time = split_train_test(data['date'].values.reshape(-1, 1), ratio)

#    print("Ciaooo: "+str(train_appliances)+str(train_appliances.size))

    #Create an HMM and fit it to data
    model = GaussianHMM(algorithm='viterbi', n_components=hidden_states_count, covariance_type='full', n_iter=10000)
    model.fit(train_lights)

    #Decode the optimal sequence of internal hidden state (Viterbi)
    hidden_states = model.predict(test_lights)

    #Generate new sample (visible, hidden)
    X, Z = model.sample(sample_size)
    print("Model Score:", model.score_samples(X))
    print(model.transmat_)

    plotTimeSerie(test_time, test_lights, hidden_states)

main()
