import difflib

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from KNNClassifier import KNNClassifier


def csv_file_loader(file_name):
    full_data_array = np.genfromtxt(file_name, delimiter=',')
    data = []
    targets = []
    for row in full_data_array:
        row = row.tolist()
        targets.append(row.pop())
        data.append(row)
    return data, targets

def space_deliminated_file_loader(file_name):
    full_data_array = pd.read_csv(file_name, delim_whitespace=True).values
    data = []
    targets = []
    for row in full_data_array:
        row = row.tolist()
        targets.append(row.pop(0))
        row.pop()
        data.append(row)
    for row in data:
        if(row[2] == '?'):
            row[2] = 0
        else:
            row[2] = float(row[2])
    return data, targets

def non_numeric_data(file_name):
    full_data_array = np.genfromtxt(file_name, delimiter=',')
    data = []
    targets = []
    for row in full_data_array:
        row = row.tolist()
        targets.append(row.pop())
        data.append(row)
    for status in targets:
        if(status == "vgood"):
            status = 4
        elif(status == "good"):
            status = 3
        elif(status == "acc"):
            status == 1
        elif(status == "unacc"):
            status == 0
    for row in data:
        i = 0
        while(i < len(row)):
            str(row[i])
            if(row[i] == "low" or row[i] == "small" or row[i] == "2"):
                row[i] = 0
            elif(i == len(row) - 1):
                if(row[i] == "med"):
                    row[i] = 2
                else:
                    row[i] = 4
            else:
                if(row[i] == "med" or row[i] == "3"):
                    row[i] = 1
                elif(row[i] == "high" or row[i] == "4" or row[i] == "big"):
                    row[i] = 2
                elif(row[i] == "vhigh" or row[i] == "5more" or row[i] == "more"):
                    row[i] = 3
            i += 1
    return data, targets



response = int(input("Please select which data set you would like to use:\n"
                 "0 = Pima Indian Diabetes\n"
                 "1 = Automobile MPG\n"
                 "2 = UCI: Car Evaluation\n "))
data_array = []
data_target = []

if(response == 0):
    data_array, data_target = csv_file_loader("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
elif(response == 1):
    data_array, data_target = space_deliminated_file_loader("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
elif(response == 2):
    data_array, data_target = csv_file_loader("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")

data_train, data_test, target_train, target_test = train_test_split(data_array, data_target, test_size=.3)

classifier = KNNClassifier(n_neighbors=int(input("Please enter the K value you would like to run with: ")))
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

if(response == 0):
    similarity_amount = difflib.SequenceMatcher(None, targets_predicted, target_test)
    print("My prediction: \n", similarity_amount.ratio())

    classifier2 = KNeighborsClassifier(n_neighbors=3)
    model2 = classifier2.fit(data_train, target_train)
    targets_predicted2 = model2.predict(data_test)
    similarity_amount2 = difflib.SequenceMatcher(None, targets_predicted2, target_test)
    print("Built-in KNeightborsClassier prediction: \n", similarity_amount2.ratio())
elif(response == 1):
    ammount_correct = 0
    total_amount = len(targets_predicted)
    for prediction, target in zip(targets_predicted, target_test):
        if(target - 4 <= prediction <= target + 4):
            ammount_correct += 1
    print("My prediction: \n", ammount_correct/total_amount)
else:
    similarity_amount = difflib.SequenceMatcher(None, targets_predicted, target_test)
    print("My prediction: \n", similarity_amount.ratio())
