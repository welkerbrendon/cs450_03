class KNNClassifier:
    data = []
    targets = []
    possible_targets = set()
    k = 1
    def __init__(self, n_neighbors):
        self.k = n_neighbors
    def fit(self, data_train, data_target):
        self.data = data_train
        self.targets = data_target
        return self
    def predict(self, data_test):
        all_results = []
        for test_line in data_test:
            k_results = []
            for data_line,target in zip(self.data,self.targets):
                distance = 0
                i = 0
                while (i < len(test_line) - 1 and i < len(data_line) - 1):
                    distance += (test_line[i] - data_line[i])**2
                    i += 1
                if(len(k_results) < self.k):
                    k_results.append([distance, target])
                else:
                    i = 0
                    while (i < self.k):
                        if(k_results[i][0] > distance):
                            k_results[i] = [distance, target]
                            i = self.k
                        i += 1
            i = 0
            k_results = sorted(k_results, key=lambda x: x[0])
            closest_neighbors = []
            while(i < self.k):
                closest_neighbors.append(k_results[i][1])
                i += 1
            prediction = [-1, closest_neighbors[0]]
            self.possible_targets = set(self.targets)
            for possibility in self.possible_targets:
                count = closest_neighbors.count(possibility)
                if(count > prediction[0]):
                    prediction[0] = count
                    prediction[1] = possibility

            all_results.append(prediction[1])

        return all_results
