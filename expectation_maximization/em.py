import random
import numpy as np
import matplotlib.pyplot as plt

initial_CPTs = {
    'D': { 'none': 0.5, 'mild': 0.25, 'severe': 0.25 },

    'S': { ('P', 'none', 'T_P'): 0.05, ('A', 'none', 'T_P'): 0.95,
           ('P', 'mild', 'T_P'): 0.05, ('A', 'mild', 'T_P'): 0.95,
           ('P', 'severe', 'T_P'): 0.05, ('A', 'severe', 'T_P'): 0.95,
           ('P', 'none', 'T_A'): 0.1, ('A', 'none', 'T_A'): 0.9,
           ('P', 'mild', 'T_A'): 0.7, ('A', 'mild', 'T_A'): 0.3,
           ('P', 'severe', 'T_A'): 0.7, ('A', 'severe', 'T_A'): 0.3, },

    'F': { ('P', 'none'): 0.1,  ('A', 'none'): 0.9,
           ('P', 'mild'): 0.6,   ('A', 'mild'): 0.4,
           ('P', 'severe'): 0.3, ('A', 'severe'): 0.7 },

    'G': { ('P', 'none'): 0.1,  ('A', 'none'): 0.9,
           ('P', 'mild'): 0.3,   ('A', 'mild'): 0.7,
           ('P', 'severe'): 0.6, ('A', 'severe'): 0.4 },

    'T': { 'P': 0.1, 'A': 0.9 }
}

def expectation_step(training_data, CPTs):
    weights = []
    total_sum_joint_probs = 0  # Initialize sum of joint probabilities
    
    for data_point in training_data:
        S, F, G, T = data_point['S'], data_point['F'], data_point['G'], data_point['T']
        D_known = data_point['D'] != 'NA'
        D_status = data_point['D'] if D_known else None

        if D_known:
            # If D's status is known, set weight to 1 for that status, and 0 for others
            weight = {status: 1.0 if status == D_status else 0.0 for status in ['none', 'mild', 'severe']}
            # add the normalization factor as 1 for this data point
            total_sum_joint_probs += 1
        else:
            D_statuses = ['none', 'mild', 'severe']
            joint_probs = {}
            for D in D_statuses:
                joint_probs[D] = (
                    CPTs['D'][D] *
                    CPTs['S'][(S, D, T)] *
                    CPTs['F'][(F, D)] *
                    CPTs['G'][(G, D)]
                )

            # Sum of joint probabilities for normalization factor
            normalization_factor = sum(joint_probs.values())
            total_sum_joint_probs += normalization_factor

            # Normalizing joint probabilities to get weights for this data point
            weight = {D: joint_probs[D] / normalization_factor for D in D_statuses}

        weights.append(weight)
    
    return weights, total_sum_joint_probs

def maximization_step(CPTs, weights, training_data):
    new_CPTs = {key: {} for key in CPTs.keys()}
    
    sum_weights_D = {status: 0 for status in ['none', 'mild', 'severe']}
    for weight in weights:
        for D, w in weight.items():
            sum_weights_D[D] += w  # Sum of weights for each Dunetts status

    # Update P(D)
    for D in ['none', 'mild', 'severe']:
        new_CPTs['D'][D] = sum_weights_D[D] / len(training_data) if len(training_data) > 0 else 0

    # Initialize sums for symptoms given D (for normalization)
    sum_symptoms_given_D = {
        symptom: {D: {'P': 0, 'A': 0} for D in ['none', 'mild', 'severe']} for symptom in ['S', 'F', 'G']
    }

    # Accumulate weights for symptoms given D
    for i, data_point in enumerate(training_data):
        S, F, G, T = data_point['S'], data_point['F'], data_point['G'], data_point['T']
        for D in ['none', 'mild', 'severe']:
            sum_symptoms_given_D['S'][D][S] += weights[i][D]
            sum_symptoms_given_D['F'][D][F] += weights[i][D]
            sum_symptoms_given_D['G'][D][G] += weights[i][D]

    # Normalize and update CPTs for symptoms given D (all combinations)
    for symptom in ['S', 'F', 'G']:
        for D in ['none', 'mild', 'severe']:
            for state in ['P', 'A']:
                if symptom == 'S':  # Include gene status for S
                    for T in ['T_P', 'T_A']:
                        key = (state, D, T)
                        total = sum_weights_D[D]
                        new_CPTs[symptom][key] = sum_symptoms_given_D[symptom][D][state] / total if total > 0 else 0
                else:
                    key = (state, D)
                    total = sum_weights_D[D]
                    new_CPTs[symptom][key] = sum_symptoms_given_D[symptom][D][state] / total if total > 0 else 0

    return new_CPTs

def modify_CPTs(CPTs, delta_upperbound):
    # Durette has 3 outcomes, so handle separately
    deltas_D = [random.uniform(0, delta_upperbound) for _ in range(3)]
    total_D = sum(deltas_D) + 1
    for key, delta in zip(CPTs['D'], deltas_D):
        CPTs['D'][key] = (CPTs['D'][key] + delta) / total_D

    # S, F and G have binary outcomes
    for condition in ['none', 'mild', 'severe']:
        for gene_status in ['T_P', 'T_A']:
            p_key = ('P', condition, gene_status)
            a_key = ('A', condition, gene_status)

            delta_P = random.uniform(0, delta_upperbound)
            delta_A = random.uniform(0, delta_upperbound)
            total = delta_P + delta_A + 1
            CPTs['S'][p_key] = (CPTs['S'][p_key] + delta_P) / total
            CPTs['S'][a_key] = (CPTs['S'][a_key] + delta_A) / total

    for symptom in ['F', 'G']:
        for condition in ['none', 'mild', 'severe']:
            p_key = ('P', condition)
            a_key = ('A', condition)
            
            delta_P = random.uniform(0, delta_upperbound)
            delta_A = random.uniform(0, delta_upperbound)
            total = delta_P + delta_A + 1
            CPTs[symptom][p_key] = (CPTs[symptom][p_key] + delta_P) / total
            CPTs[symptom][a_key] = (CPTs[symptom][a_key] + delta_A) / total
    
def perform_em(training_dataset, CPTs, max_iterations=100000, convergence_threshold=0.01):
    old_sum_joint_probs = -float('inf')
    
    for iteration in range(max_iterations):
        weights, new_sum_joint_probs = expectation_step(training_dataset, CPTs)
        new_CPTs = maximization_step(CPTs, weights, training_dataset)
        CPTs.update(new_CPTs)  # Update CPTs with the new values
        
        # Check for convergence using the sum of joint probabilities
        if abs(new_sum_joint_probs - old_sum_joint_probs) < convergence_threshold:
            print(f"Convergence reached after {iteration} iterations.")
            break
        
        old_sum_joint_probs = new_sum_joint_probs
    
    return CPTs

# Prediction using test dataset
def predict_dunetts(symptoms, gene_status, CPTs):
    gene_key = 'T_P' if gene_status == 1 else 'T_A'
    categories = ['none', 'mild', 'severe']
    max_likelihood = -1
    prediction = -1

    for i, category in enumerate(categories):
        likelihood = CPTs['D'][category]  # P(D=mild/severe/none)
        for j, symptom_key in enumerate(['S', 'F', 'G']):
            symptom_status = 'P' if symptoms[j] == 1 else 'A'
            if symptom_key == 'S':  # Sloepnea depends on gene status
                likelihood *= CPTs[symptom_key][(symptom_status, category, gene_key)]
            else:
                likelihood *= CPTs[symptom_key][(symptom_status, category)]
        
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            prediction = i  # 0 for none, 1 for mild, 2 for severe
    
    return prediction

def evaluate_predictions(test_file_path, CPTs):

    test_data = []
    with open(test_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            numbers = [int(part) for part in parts]
            test_data.append(numbers)

    correct_predictions = 0
    total_predictions = len(test_data)

    for data_point in test_data:
        symptoms = data_point[:3]
        gene_status = data_point[3]
        actual_category = data_point[4]
        predicted_category = predict_dunetts(symptoms[:], gene_status, CPTs)
        
        if predicted_category == actual_category:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

# Parsing dataset
def parse_line(line):
    """
    Symptoms and gene are represented as 0 (absent) or 1 (present).
    Dunetts Syndrome status is represented as -1 (not available), 0 (none), 1 (mild), or 2 (severe).
    """
    parts = line.strip().split()
    numbers = [int(part) for part in parts]

    # features
    symptoms = ['S', 'F', 'G']
    gene = 'T'
    dunetts = 'D'

    # values of features
    symptom_values = ['A' if num == 0 else 'P' for num in numbers[:3]]
    gene_status = 'T_A' if numbers[3] == 0 else 'T_P'

    dunetts_status = {0: 'none', 1: 'mild', 2: 'severe', -1: 'NA'}[numbers[4]]

    # each datapoint is a dictionary with keys 'S', 'F', 'G' (symptoms), 'T' (gene) and 'D'
    # and values being P/A for symptoms with P_T and A_T for gene
    data_point = {symptoms[i]: symptom_values[i] for i in range(3)}
    data_point[gene] = gene_status
    data_point[dunetts] = dunetts_status

    return data_point

def load_data(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            data_point = parse_line(line)
            dataset.append(data_point)
    return dataset

def output_parsed_dataset(filePath, dataset):
    with open(filePath, 'w') as file:
        for data_point in dataset:
            data_point_str = str(data_point)
            file.write(data_point_str + '\n')

def plotData(training_dataset):
    delta_values = np.arange(0, 4, 0.2)
    num_trials = 20
    accuracies_before_em = {delta: [] for delta in delta_values}
    accuracies_after_em = {delta: [] for delta in delta_values}

    for delta in delta_values:
        for _ in range(num_trials):
            print("\n")
            CPTs = initial_CPTs.copy()
            modify_CPTs(CPTs, delta)
            accuracy_before = evaluate_predictions("./em-data/testdata.txt", CPTs)
            print(f"delta: {delta}, accuracy before: {accuracy_before}")
            accuracies_before_em[delta].append(accuracy_before)

            final_CPTs = perform_em(training_dataset, CPTs).copy()
            
            accuracy_after = evaluate_predictions("./em-data/testdata.txt", final_CPTs)
            print(f"delta: {delta}, accuracy after: {accuracy_after}")
            accuracies_after_em[delta].append(accuracy_after)

    mean_accuracies_before = [np.mean(accuracies_before_em[delta]) for delta in delta_values]
    std_accuracies_before = [np.std(accuracies_before_em[delta]) for delta in delta_values]

    mean_accuracies_after = [np.mean(accuracies_after_em[delta]) for delta in delta_values]
    std_accuracies_after = [np.std(accuracies_after_em[delta]) for delta in delta_values]

    plt.figure(figsize=(10, 6))
    plt.errorbar(delta_values, mean_accuracies_before, yerr=std_accuracies_before, label='Before EM', fmt='-o')
    plt.errorbar(delta_values, mean_accuracies_after, yerr=std_accuracies_after, label='After EM', fmt='-o')

    plt.title('Prediction Accuracy vs. Delta')
    plt.xlabel('Delta')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    training_dataset = load_data('em-data/traindata.txt')
    output_parsed_dataset('parsed_train_data.txt', training_dataset)

    # for delta in np.arange(0, 4, 0.2):
    #     CPTs = initial_CPTs.copy()
    #     modify_CPTs(CPTs, delta)
    #     final_CPTs = perform_em(training_dataset, CPTs)
    #     print(evaluate_predictions("./em-data/testdata.txt", final_CPTs))

    plotData(training_dataset)

