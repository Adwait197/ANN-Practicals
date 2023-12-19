import numpy as np

def initialize_weights(input_dim, category):
    weights = np.random.uniform(size=(input_dim,))
    weights /= np.sum(weights)
    return weights

def calculate_similarity(input_pattern, weights):
    return np.minimum(input_pattern, weights).sum()

def update_weights(input_pattern, weights, vigilance):
    while True:
        activation = calculate_similarity(input_pattern, weights)
        if activation >= vigilance:
            return weights
        else:
            weights[np.argmax(input_pattern)] += 1
            weights /= np.sum(weights)

def ART_neural_network(input_patterns, vigilance):
    num_patterns, input_dim = input_patterns.shape
    categories = []

    for pattern in input_patterns:
        matched_category = None
        for category in categories:
            if calculate_similarity(pattern, category["weights"]) >= vigilance:
                matched_category = category
                break

        if matched_category is None:
            weights = initialize_weights(input_dim, len(categories))
            matched_category = {"weights": weights, "patterns": []}
            categories.append(matched_category)

        matched_category["patterns"].append(pattern)
        matched_category["weights"] = update_weights(pattern, matched_category["weights"], vigilance)

    return categories

# Example usage
input_patterns = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]])
vigilance = 0.5

categories = ART_neural_network(input_patterns, vigilance)

# Print the learned categories
for i, category in enumerate(categories):
    print(f"Category {i+1}:")
    print("Patterns:")
    [print(pattern) for pattern in category["patterns"]]
    print("Weights:")
    print(category["weights"])
    print()
