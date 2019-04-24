#Osservazione sulla media e varianza degli hidden state!! Fare il confronto frafico tirando fuori le gaussiane da entrambi predict/sample
#Capire cosa è sto ground truth
#rappresentare il grafo degli stati
#plottare bitcoin
#https://waterprogramming.wordpress.com/2018/07/03/fitting-hidden-markov-models-part-ii-sample-python-script/
#https://turing.ml/tutorials/4-bayeshmm/
#https://rdrr.io/cran/seqHMM/man/plot.hmm.html
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def load_csv(file):
    return pd.read_csv(file)


def split_train_test(data, ratio):
    split_index = int(len(data)*ratio)
    return data[0:split_index], data[split_index+1::]


def plot_time_series(visible, hidden, time=None, title=None):
    sns.set_style("whitegrid")
    if time is None:
        time = np.arange(len(visible))
    else:
        time = [x[5:len(x)-3] for x in time[:, 0].tolist()]
    df = pd.DataFrame(dict(visible=visible[:, 0].tolist(), hidden=hidden.tolist(), time=time))
    sns.lineplot(data=visible, legend=None)
    ax = sns.scatterplot(data=df, x='time', y='visible', hue='hidden', zorder=10)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xticks(rotation=90)
    plt.show()

def hidden_markov_model(hidden_states_count, train, test, time, sample_size, data_name):
    #Create an HMM and fit it to data
    model = GaussianHMM(algorithm='viterbi', n_components=hidden_states_count, covariance_type='diag', n_iter=10000)
    model.fit(train)

    #Decode the optimal sequence of internal hidden state (Viterbi)
    hidden_states = model.predict(test)
    plot_time_series(test, hidden_states, time, data_name+' - Predict')

    prob_next_step = model.transmat_[hidden_states[-1], :]
    print('Next Step '+str(prob_next_step))

    print('\nMeans and Variances of hidden states:')
    for i in range (model.n_components):
        print('\nHidden state', i+1)
        print('Mean = ', round(model.means_[i][0], 3))
        print('Variance = ', round(np.diag(model.covars_[i])[0], 3))

    #Generate new sample (visible, hidden)
    X, Z = model.sample(sample_size)
    plot_time_series(X, Z, title=data_name+' - Sample')
    print("\nModel Score:", model.score_samples(X))
    print(model.transmat_)


def main():
    data = load_csv('./dataset/energydata_complete.csv')
    ratio = 4/5 
    sample_size = 100
    hidden_states_count = 3 

    #Reshaping data
    train_lights, test_lights = split_train_test(data['lights'].values.reshape(-1, 1), ratio)
    train_appliances, test_appliances = split_train_test(data['Appliances'].values.reshape(-1, 1), ratio)
    train_time, test_time = split_train_test(data['date'].values.reshape(-1, 1), ratio)


#    #Reshape v2
#    data_lights = np.column_stack([data['lights']])
#    data_appliaces = np.column_stack(data['Appliances'])


    hidden_markov_model(hidden_states_count, train_lights, test_lights, test_time, sample_size, 'Lights')
#    hidden_markov_model(hidden_states_count, train_appliances, test_appliances, test_time, sample_size, 'Appliances')

main()
