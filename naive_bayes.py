#!/usr/bin/env python

from collections import Counter, defaultdict
import numpy as np

def probability(list1):
    number_of_group = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key]/float(number_of_group)
    return prob

if __name__ == "__main__":

    dataset = np.array([['Rainy','Hot','High','False'],['Rainy','Hot','High','True'],
                        ['Overcast','Hot','High','False'],['Sunny','Mild','High','False'],
                        ['Sunny','Cool','Normal','False'],['Sunny','Cool','Normal','True'],
                        ['Overcast','Cool','Normal','True'],['Rainy','Mild','High','False'],
                        ['Rainy','Cool','Normal','False'],['Sunny','Mild','Normal','False'],
                        ['Rainy','Mild','Normal','True'],['Overcast','Mild','High','True'],
                        ['Overcast','Hot','Normal','False'],['Sunny','Mild','High','True']])
    labels = np.array(['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No'])
    test_data = np.array(['Sunny','Hot','Normal','False'])

    priors = probability(labels)
    print(priors)
    unique_labels = np.unique(labels)
    likelihoods = dict()
    for one_label in unique_labels:
        likelihoods[one_label] = defaultdict(list)
        idx = np.where(labels == one_label)[0]
        subset = dataset[idx, :]
        print(subset)
        feature_numbers = subset.shape[1]
        for i in range(dataset.shape[1]):
            likelihoods[one_label][i] = probability(list(subset[:, i]))

    # calculates posteriors and compares them
    posteriors = priors
    for one_label in unique_labels:
        for i in range(len(test_data)):
            posteriors[one_label] *= likelihoods[one_label][i][test_data[i]]
    print(posteriors)
    pre_label = 0
    max_bayes = 0
    for key in posteriors.keys():
        if posteriors[key] > max_bayes:
            pre_label = key
            max_bayes = posteriors[key]
    print(pre_label)
