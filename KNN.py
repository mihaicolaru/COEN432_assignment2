# KNN implementation
from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy as np

import matplotlib.pyplot as plt


# read training data from file     
f = open("A2_15000.txt", "r")

# read input file line by line
lines = f.read().splitlines()
f.close()
del f

input_features = []
target = []

# parse file into usable data
line_index = 0
limit = 100
for line in lines:
    components = line.split(";")  

    # extract target
    target.append(float(components[3]))
    
    # extract numeric probabilities as list from input matrix
    pairing_probs_str = components[2]
    individual_probs_list = []
    i = 0
    for item in pairing_probs_str.split("'"):
        if(i % 2 == 0):
            pass
        else:
            individual_probs_list.append(float(item))
        i += 1
    
    # extract upper right corner of probability matrix
    individual_probs_matrix = []
    i = 0
    j = 1
    for idx in range(0, 60):
        individual_probs_matrix.append(individual_probs_list[i + j: i + 60])
        i += 60
        j += 1
    
    # extract gene position/probability pairs
    feature_set = []
    for i in range(0, 59):
        pair_encode_i = []
        # encode 1st gene position
        for k in range(0, i):
            pair_encode_i.append(0)
        pair_encode_i.append(1)
        # print(f"pair encode i: {pair_encode_i}, length: {len(pair_encode_i)}")
        
        for j in range(0, 60-(i+1)):
            # encode 2nd gene position
            pair_encode_j = []
            for k in range(0, j):
                pair_encode_j.append(0)
            pair_encode_j .append(1)
            # print(f"pair encode j: {pair_encode_j}, length: {len(pair_encode_j)}")

            full_pair = pair_encode_i + pair_encode_j
            for k in range(0, 60-len(full_pair)):
                full_pair.append(0)
            # print(f"pair encode full: {full_pair}, length: {len(full_pair)}")
            
            full_pair.append(individual_probs_matrix[i][j])
            feature_set.append(full_pair)
            # feature_set.append(individual_probs_matrix[i][j])

    input_features.append(feature_set)

    line_index += 1
    if(line_index >= limit):
        break
    # break


Y = np.array(target)
del target
del lines

print("done parsing, flattening")

x_flat = [np.reshape(instance, -1).astype(np.float32) for instance in input_features]
del input_features

print("done flattening, create array")

X = np.array(x_flat)
del x_flat

print("start testing")

# for i in range(1, 1000):
#     clf = DecisionTreeRegressor(max_depth=i)
#     scores = cross_val_score(clf, fitted, target, cv=10)
#     print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

track_neighbors = []
track_mean = []
track_std_dev = []
for i in range(1, int(limit/2), 2):
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='ball_tree')

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    neg_scores = cross_val_score(knn, X, Y, cv=kf, scoring='neg_mean_squared_error')
    scores = [abs(score) for score in neg_scores]
    mean = abs(neg_scores.mean())
    std_dev = abs(neg_scores.std())
    track_neighbors.append(i)
    track_mean.append(mean)
    track_std_dev.append(std_dev)

    print(f"neighbors: {i}, mean MSE: {mean}, mean std dev: {std_dev}\nscores: {scores}\n")


fig, ax1 = plt.subplots()

ax1.set_xlabel('neighbors')
ax1.set_ylabel('mean MSE', color='tab:red')
ax1.plot(track_neighbors, track_mean, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('std dev', color='tab:blue')  # we already handled the x-label with ax1
ax2.plot(track_neighbors, track_std_dev, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.suptitle(f"Samples used: {limit}")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig('Ass2Lifetime.png')