# KNN implementation
from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# read training data from file     
f = open("A2_15000.txt", "r")

# read input file line by line
lines = f.read().splitlines()

input_features = []
target = []

# parse file into usable data
lim = 0
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

    lim += 1
    if(lim >= 10):
        break
    # break

# print(input_features)






# enc = OneHotEncoder(handle_unknown='ignore')

# enc.fit(input_features)
# fitted = enc.transform(input_features).toarray()




# for i in range(1, 1000):
#     clf = DecisionTreeRegressor(max_depth=i)
#     scores = cross_val_score(clf, fitted, target, cv=10)
#     print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



knn = KNeighborsRegressor(n_neighbors=40)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(knn, input_features, target, cv=kf, scoring='r2')

print(f"score: {scores}, mr2: {scores.mean()}")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))