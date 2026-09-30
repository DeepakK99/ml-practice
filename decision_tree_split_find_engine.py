def gini(y):
    n = len(y)
    class_counts = {}

    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1

    return 1 - sum(
        (count / n) ** 2
        for count in class_counts.values()
    )

def weighted_gini(left, right):
    gini_left = gini(left)
    gini_right = gini(right)

    total = len(left) + len(right)

    return (
        len(left) / total * gini_left
        + len(right) / total * gini_right
    )

def find_best_split(X, y):
    n = len(X) 
    parent_gini = gini(y)
    thersholds = []
    for i in range(1, len(X)):
        thersholds.append((X[i-1] + X[i])/2)

    max_gain = float("-inf")
    best_threshold = None
    best_split = []
    
    for t in thersholds:
        i = 0
        left = []
        right = []

        while X[i] <= t:
            left.append(y[i])
            i += 1
        right = y[i:]

        w_gini = weighted_gini(left, right)
        gain = parent_gini - w_gini
        if gain > max_gain:
            max_gain = gain
            best_threshold = t
            best_split = [left, right] 

    return (max_gain, best_threshold, best_split)

def find_best_feature_split(X, y):
    best_feature_gain = float("-inf")
    best_feature_threshold = None
    best_feature_split = []
    best_feature = None
    for feature_num in range(len(X[0])):
        x_tup = [(data_point[feature_num], i) for i, data_point in enumerate(X)]
        x_tup.sort()
        order = [tup[1] for tup in x_tup]
        curr_X = [tup[0] for tup in x_tup]
        curr_y = [y[i] for i in order]
        max_gain, best_threshold, best_split = find_best_split(curr_X, curr_y)
        if max_gain > best_feature_gain:
            best_feature_gain = max_gain
            best_feature_threshold = best_threshold
            best_feature_split = best_split
            best_feature = feature_num
    
    return (best_feature, best_feature_gain, best_feature_threshold, best_feature_split)

X = [
    [20, 100],
    [25, 200],
    [30, 150],
    [35, 300],
    [40, 250],
    [45, 400]
]

y = [0, 0, 1, 1, 1, 1]

print(find_best_feature_split(X, y))