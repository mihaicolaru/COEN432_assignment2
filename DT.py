from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# read training data from file     
f = open("A2_15000.txt", "r")

# read input file line by line
lines = f.read().splitlines()

input_features = []
target = []

# parse file into usable data
for line in lines:

    components = line.split(";")   
    
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
    
    # extract features from instance

    # nucleotide frequency (g)
    sequence = components[1]
    num_a = 0
    num_u = 0
    num_g = 0
    num_c = 0
    for gene in sequence:
        if(gene == "a"):
            num_g += 1
        if(gene == "u"):
            num_c += 1
        if(gene == "g"):
            num_g += 1
        if(gene == "c"):
            num_c += 1

    # average pairing probability
    # number of pairs over threshold (0.1)
    num_pairs = 0
    total_prob = 0
    count = 0
    for i in range(0, 60):
        for j in range(0, 60-(i+1)):
            # print(f"i: {i}, j: {j}")
            total_prob += individual_probs_matrix[i][j]
            if(individual_probs_matrix[i][j] > 0.1):
                num_pairs += 1
            count += 1
    # print(f"count: {count}")
    average_prob = total_prob/count

    input_features.append([num_g, total_prob])
    target.append(float(components[3]))


    # break

    # # extract corresponding pairs and probabilities
    # pairs = []
    # probabilities = []

    # sequence = components[1]

    # # probability index
    # i = 0
    # j = 0
    # # pairs index

    # for idx in range(0, len(sequence)):
    #     i = idx
    #     j = 0
    #     for jdx in range(idx + 1, len(sequence)):
    #         # print(f"idx: {i}\njdx: {j}")
    #         pairs.append(sequence[idx] + sequence[jdx])
    #         probabilities.append(individual_probs_matrix[i][j])
    #         j += 1


    # for i in range(0, len(pairs)):
    #     print(f"pair: {pairs[i]}\nprobability: {probabilities[i]}")




# depth = []
# for i in range(3,20):
#     clf = tree.DecisionTreeClassifier(max_depth=i)
#     # Perform 7-fold cross validation 
#     scores = cross_val_score(estimator=clf, X=x, y=y, cv=7, n_jobs=4)
#     depth.append((i,scores.mean()))
# print(depth)




# kf = KFold(n_splits=10)
# kf.get_n_splits(input_features)

# for train, test in kf.split(input_features):
#     print(f"train: {train}")
#     print(f"test: {test}")

for i in range(1, 1000):
    clf = DecisionTreeRegressor(max_depth=i)
    scores = cross_val_score(clf, input_features, target, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# X = input_features[0:500]
# Y = target[0:500]
# X_test = input_features[500:1000]

# regr_1 = DecisionTreeRegressor(max_depth=10)
# regr_2 = DecisionTreeRegressor(max_depth=20)

# regr_1.fit(X, Y)
# regr_2.fit(X, Y)

# Y1_pred = regr_1.predict(X_test)
# Y2_pred = regr_2.predict(X_test)

# Y_actual = target[500:1000]


# for i in range(0, len(X_test)):
#     print(f"predicted 1,2: {Y1_pred[i]}, {Y2_pred[i]}\nactual: {Y_actual[i]}")