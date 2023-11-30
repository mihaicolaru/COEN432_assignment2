

# # define instance class, holds training data unit
# class instance:
#     def __init__(self, number, sequence, pairing_probs, activation):        
#         self.number = number
#         self.sequence = sequence
#         self.pairing_probs = pairing_probs
#         self.activation = activation

#     def __str__(self):
#         return f"instance number: {self.number}\nsequence: {self.sequence}\npairing probability matrix: {str(self.pairing_probs)}\nactivation: {self.activation}"
    
#     def display(self):
#         print("instance number:", self.number, "\nsequence:", self.sequence, "activation:", self.activation, "\n")


from sklearn.tree import DecisionTreeRegressor


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
    num_g = 0
    for gene in sequence:
        if(gene == "g"):
            num_g += 1

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

    input_features.append([num_pairs, total_prob])
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


X = input_features[0:500]
Y = target[0:500]
X_test = input_features[500:1000]

regr_1 = DecisionTreeRegressor(max_depth=10)
regr_2 = DecisionTreeRegressor(max_depth=20)

regr_1.fit(X, Y)
regr_2.fit(X, Y)

Y1_pred = regr_1.predict(X_test)
Y2_pred = regr_2.predict(X_test)

Y_actual = target[500:1000]


for i in range(0, len(X_test)):
    print(f"predicted 1,2: {Y1_pred[i]}, {Y2_pred[i]}\nactual: {Y_actual[i]}")