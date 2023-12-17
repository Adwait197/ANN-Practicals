import numpy as np

j = int(input("Enter a Number (0-9): "))
step_function = lambda x: 1 if x >= 0 else 0

training_data = [
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
]

weights = np.array([0, 0, 0, 0, 0, 1])

for data in training_data:
    input = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input, weights))
    error = label - output
    weights += input * error

input = np.array([int(x) for x in list('{0:06b}'.format(j))])
output = "odd" if step_function(np.dot(input, weights)) == 0 else "even"
print(j, " is ", output)

