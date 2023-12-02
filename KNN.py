# KNN implementation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict

import numpy as np

import matplotlib.pyplot as plt

# ==================== read input file ====================
# read training data from file     
f = open("A2_15000.txt", "r")

# read input file line by line
lines = f.read().splitlines()
f.close()

input_features = []
target = []

# parse file into usable data
line_index = 0
limit = 5000
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
        jdx = i + 1
        for j in range(0, 60-(i+1)):
            loc_prob_pair = [i, jdx, individual_probs_matrix[i][j]]
            feature_set.append(loc_prob_pair)
            jdx += 1

    # add pos/prob pairs to input
    input_features.append(feature_set)

    # check for number of lines reached
    line_index += 1
    if(line_index >= limit):
        break
    

print(f"done parsing {line_index} or {len(input_features)}, flattening input features")

x_flat = [np.reshape(instance, -1).astype(np.float32) for instance in input_features]

print("done flattening, creating feature/target arrays")

X = np.array(x_flat)
Y = np.array(target)


print("start testing")

# for i in range(1, 1000):
#     clf = DecisionTreeRegressor(max_depth=i)
#     scores = cross_val_score(clf, fitted, target, cv=10)
#     print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# loop to find optimal number of neighbors
track_neighbors = []
track_mean = []
track_std_dev = []

# loop to find optimal number of neighbors for KNN
try:
    for i in range(1, int(limit/2), int(limit/100)):
        # define KNN regressor parameters
        knn = KNeighborsRegressor(n_neighbors=i, weights='distance', algorithm='ball_tree', n_jobs=1)

        # define K-fold cross validation parameters
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        # K-fold KNN, extract scores (MSE), mean, standard deviation
        neg_scores = cross_val_score(knn, X, Y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        scores = [abs(score) for score in neg_scores]
        mean = abs(neg_scores.mean())
        std_dev = abs(neg_scores.std())
        track_neighbors.append(i)
        track_mean.append(mean)
        track_std_dev.append(std_dev)

        print(f"neighbors: {i}, mean MSE (RSS/number of samples): {mean}, mean std dev: {std_dev}\nscores: {scores}\n")

except KeyboardInterrupt:
    print("ctrl+c pressed, printing plots...")


# plot loop results
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
fig.savefig(f'{limit} samples.png')


# find best neighbor
i = 0
top_score = 1000
top_neighbors = 0
for score in track_mean:
    if(score < top_score):
        top_score = score
        top_neighbors = track_neighbors[i]
    i += 1

print(f"top score: {top_score}, number of neighbors: {top_neighbors}\n")

print(f"testing {top_neighbors} neighbors with k=10 K-fold cross validation:\n")

# perform knn on the data once again at optimal neighbor
knn = KNeighborsRegressor(n_neighbors=top_neighbors, weights='distance', algorithm='ball_tree')

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# K-fold KNN, extract scores (MSE), mean, standard deviation and predictions
neg_scores = cross_val_score(knn, X, Y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
predictions = cross_val_predict(knn, X, Y, cv=kf, n_jobs=-1)
scores = [abs(score) for score in neg_scores]
mean = abs(neg_scores.mean())
std_dev = abs(neg_scores.std())

print(f"mean MSE (RSS/number of samples): {mean}, mean RSS = {limit * mean}, mean std dev: {std_dev}\nscores: {scores}\n")

print(f"predictions:\n")

i = 0
for prediction in predictions:
    print(f"prediction {i}: {prediction:0.6f}, actual {i}: {target[i]:0.6f}")
    i += 1