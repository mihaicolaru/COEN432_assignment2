
class instance:
    def __init__(self, number, sequence, pairing_probs, activation):        
        self.number = number
        self.sequence = sequence
        self.pairing_probs = pairing_probs
        self.activation = activation

    def __str__(self):
        return f"instance number: {self.number}\nsequence: {self.sequence}\npairing probability matrix: {str(self.pairing_probs)}\nactivation: {self.activation}"
    
    def display(self):
        print("instance number:", self.number, "\nsequence:", self.sequence, "activation:", self.activation, "\n")




        
f = open("A2_15000.txt", "r")

lines = f.read().splitlines()

instances = []

for line in lines:
    components = line.split(";")
    
    pairing_probs_str = components[2]
    individual_probs_list = []
    individual_probs_matrix = []
    
    i = 0
    for item in pairing_probs_str.split("'"):
        if(i % 2 == 0):
            pass
        else:
            individual_probs_list.append(float(item))
        i += 1
    
    i = 0
    j = 1
    for idx in range(0, 60):
        individual_probs_matrix.append(individual_probs_list[i + j: i + 60])
        i += 60
        j += 1
    
    # for item in individual_probs_matrix:
    #     print(len(item))

    # print(str(individual_probs_matrix))

    inst = instance(components[0], components[1], individual_probs_matrix, components[3])

    instances.append(inst)

    # print(inst)

    # print(len(individual_probs))

    # for i in individual_probs:
    #     print(i)

    

    # print(index, sequence, pairing_probs, activation)
    # break

for inst in instances:
    print(inst)