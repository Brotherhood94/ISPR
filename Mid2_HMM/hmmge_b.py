#https://waterprogramming.wordpress.com/2018/07/03/fitting-hidden-markov-models-part-ii-sample-python-script/
#https://turing.ml/tutorials/4-bayeshmm/
#https://rdrr.io/cran/seqHMM/man/plot.hmm.html
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.stats as stats
import math
import os, sys

flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]


def load_csv(file):
    return pd.read_csv(file)

def split_train_test(data, ratio):
    split_index = int(len(data)*ratio)
    return data[0:split_index], data[split_index+1::]

def get_points(model):
    values = []
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(np.array(model.transmat_)))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    for i in range(model.n_components):
        mean = model.means_[i][0]
        sigma = math.sqrt(np.diag(model.covars_[i])[0])
        x = np.linspace(mean - 3*sigma, mean + 3*sigma, 1000)
        values.append({'no': i, 'mean':mean, 'sigma':sigma, 'x':x, 'color':flatui[i], 'pi':pi[i]})
    return values

def plot_gaussians(data, points, states, title):
    sns.set()
    fig, ax = plt.subplots()
    fig.set_size_inches(19.20, 10.80)
    ax.set_title(title)
    ax.hist(data, bins=60, density=True, color='black', alpha=0.3, linewidth=2)
    for point in points:
        ax.plot(point['x'], point['pi']*stats.norm.pdf(point['x'], point['mean'], point['sigma']), color=point['color'], label='hidden state '+str(point['no']))
    plt.legend()
    plt.savefig('./Results_B/'+str(states)+'/'+title+'.png', dpi=300)                                                                                                                                                                                    
#    plt.show()    


def plot_time_series(visible, hidden, states, time=None, title=None):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(19.20,10.80))
    if time is None:
        time = np.arange(len(visible))
    else:
        time = [x[5:len(x)-3] for x in time[:, 0].tolist()]
    df = pd.DataFrame(dict(visible=visible[:, 0].tolist(), hidden=hidden.tolist(), time=time))
    sns.lineplot(data=visible, palette='PuBuGn_d', legend=None)
    ax = sns.scatterplot(data=df, x='time', y='visible', hue='hidden', palette='cubehelix', zorder=10)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    plt.xticks(rotation=90)
    plt.savefig('./Results_B/'+str(states)+'/'+title+'.png', dpi=300)
#    plt.show()

def hidden_markov_model(hidden_states_count, train, test, time, sample_size, data_name, f):
    #Create an HMM and fit it to data
    model = GaussianHMM(algorithm='map', n_components=hidden_states_count, covariance_type='diag', n_iter=10000)
    model.fit(train)
    
    #Decode the optimal sequence of internal hidden state (Viterbi)
    hidden_states = model.predict(test)

    #Prob next step
    prob_next_step = model.transmat_[hidden_states[-1], :]

    #Generate new sample (visible, hidden)
    X, Z = model.sample(sample_size)
   
    #Plot Data
    plot_time_series(test, hidden_states, hidden_states_count, None, data_name+' - Predict')
    points = get_points(model)
    plot_gaussians(train, points, hidden_states_count, data_name+' - Gaussian Predict')
#    plot_time_series(X, Z, hidden_states_count, None, title=data_name+' - Sample')
   
    #Write Data
    f.write('\n'+data_name+'\n')
    f.write('Transition Matrix:\n'+str(model.transmat_)+'\n')
    f.write('\nNext Step '+str(prob_next_step)+'\n')
    for point in points:
        f.write('\nHidden Variable NO° '+str(point['no'])+'\n\tMean: '+str(point['mean'])+'\n\tSigma: '+str(point['sigma'])+'\n')
    f.write('\n#######################################\n')
#    f.write("\nModel Score:"+str(model.score_samples(X)))

def main():
    data = load_csv('./dataset/BTC-USD.csv')
    ratio = 4/5 
    sample_size = 100
    hidden_states_count = 3 
    
    os.makedirs('./Results_B/'+str(hidden_states_count))
    f = open('./Results_B/'+str(hidden_states_count)+'/data.txt', 'w+')

    #Reshaping data
    train_open, test_open = split_train_test(data['Open'].values.reshape(-1, 1), ratio)
    train_close, test_close = split_train_test(data['Close'].values.reshape(-1, 1), ratio)
    train_time, test_time = split_train_test(data['Date'].values.reshape(-1, 1), ratio)

    hidden_markov_model(hidden_states_count, np.log10(train_open), np.log10(test_open), test_time, sample_size, 'Open', f)
    hidden_markov_model(hidden_states_count, np.log10(train_close), np.log10(test_close), test_time, sample_size, 'Close', f)
    
    f.close()

main()
