from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import pickle
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Expression, Constraint, value
import math
from pyomo.environ import Piecewise
import torch
# Load classifier learning results
file_path = '.../learning_results.pkl'  

with open(file_path, 'rb') as f:
    results_dict = pickle.load(f)
optimization_time = results_dict['optimization_time']
accuracy = results_dict['accuracy']
test_result = results_dict['test_results']
train_result = results_dict['train_results']
optimal_parameters_list = results_dict['optimal_parameters_list']
# load the training set
file_path = '.../train32.xlsx'
data = pd.read_excel(file_path)

print(data.head())

labels = data.iloc[:, 0].to_numpy()  
features_df = data.iloc[:, 1:-2] 
print("Extracted features:")

feature_min_max = [(features_df[col].min(), features_df[col].max()) for col in features_df.columns]
# Define the Pyomo optimization model
model = ConcreteModel()
# Define transformations between piecewise vectors and constants (used in objective function)
def vector_to_constant(vector):
    n = len(vector)  
    constant = sum(vector) / n
    return constant
def constant_to_vector(constant, num_intervals):
    vector = torch.zeros(num_intervals)  
    remaining_constant = constant * num_intervals  
    for i in range(num_intervals):
        if remaining_constant >= 1:
            vector[i] = 1
            remaining_constant -= 1
        else:
            vector[i] = remaining_constant
            break
    return vector.tolist() 

# Select sample for counterfactual analysis
sample_id = 750
target_label = 2
sample_index = test_result['sample_ids'].tolist().index(sample_id)
sample_piecewise_vectors = [test_result['piecewise_vectors'][i][sample_index, :] for i in range(len(test_result['piecewise_vectors']))]
sample_piecewise_vectors = np.array(sample_piecewise_vectors, dtype=object)
x = [vector_to_constant(vec) for vec in sample_piecewise_vectors]
print(x)
train_scoreten = train_result['train_scores']
labels_train = train_result['labels_train']
train_scores = train_scoreten.tolist()

num_features = 12

M = 1000
A = 0.5
num_intervals_list = [6] * 12
num_intervals_list[1:5] = [2, 2, 2, 2]
# actionability constraints
fixed_features = [0, 1, 2, 3, 4, 9, 11] 
directions = {
    7: "increase",  
    10: "increase"  
}
non_fixed_features = [i for i in range(num_features) if i not in fixed_features]
sub_feature_bounds = {
    5: (60, 90),    
    6: (60, 100),    
    7: (110, 160),  
    8: (7.35, 7.45),
    10: (35, 55)   
}


# Create decision variables and their bounds

model.x0 = Var(non_fixed_features, bounds=(0, 1))
for i in non_fixed_features:
    min_val, max_val = feature_min_max[i]
    lb = 0.0
    ub = 1.0
    x_normalized = x[i]
    if i in sub_feature_bounds:
        real_lb, real_ub = sub_feature_bounds[i]
        sub_lb = (real_lb - min_val) / (max_val - min_val)
        sub_ub = (real_ub - min_val) / (max_val - min_val)     
        lb = min(sub_lb, x_normalized)
        ub = max(sub_ub, x_normalized)
    else:
        lb = 0.0
        ub = 1.0
    if i in directions:
        if directions[i] == "increase":
            lb = max(lb, x[i])
        elif directions[i] == "decrease":
            ub = min(ub, x[i])
    model.x0[i].setlb(lb)
    model.x0[i].setub(ub)
            
model.s = Var(
    [(i, j) for i in non_fixed_features for j in range(num_intervals_list[i] - 1)], 
    bounds=(0, 1)
)
model.ξ = Var(non_fixed_features, within=Binary)
model.a = Var(
    [(i, j) for i in non_fixed_features for j in range(num_intervals_list[i] - 1)], 
    within=Binary
)
model.b = Var(
    [(i, j) for i in non_fixed_features for j in range(num_intervals_list[i] - 1)], 
    within=Binary
)

optimal_parameters = optimal_parameters_list[0]
# load optimal parameters
lambda_opt_data = optimal_parameters['lambda']
lambda_opt = float(lambda_opt_data)
delta_u_opt = [[float(du) for du in du_list] for du_list in optimal_parameters['delta_u']]
v_opt = [torch.tensor(v, dtype=torch.float32) for v in optimal_parameters['v']]
# Create piecewise vectors for decision variables
epsilon = 1e-6
for i in non_fixed_features:
    for j in range(num_intervals_list[i]-1):
        model.add_component(f'constraint_a_b_{i}_{j}_1', Constraint(expr=model.s[i, j] <= model.a[i, j]))
        model.add_component(f'constraint_a_b_{i}_{j}_2', Constraint(expr=model.s[i, j] >= epsilon * model.a[i, j]))
        model.add_component(f'constraint_a_b_{i}_{j}_3', Constraint(expr=model.s[i, j] >= model.b[i, j]))
        model.add_component(f'constraint_a_b_{i}_{j}_4', Constraint(expr=model.s[i, j] <= model.b[i, j] + (1 - model.b[i, j]) * (1 - epsilon)))
        
        if j < num_intervals_list[i] - 2:  
            model.add_component(f'constraint_b_{i}_{j}_5', Constraint(expr=model.s[i, j+1] <= model.b[i, j]))
            model.add_component(f'constraint_ab_order_{i}_{j}_6', Constraint(expr=model.a[i, j] >= model.a[i, j+1]))
            model.add_component(f'constraint_ab_order_{i}_{j}_7', Constraint(expr=model.b[i, j] >= model.b[i, j+1]))
for i in non_fixed_features:
    model.add_component(f'constraint_x0_{i}', Constraint(expr=model.x0[i] == sum(model.s[i, j] for j in range(num_intervals_list[i]-1)) / (num_intervals_list[i] - 1)))
s_complete = {}
for i in range(num_features):
    if i in non_fixed_features:
        for j in range(num_intervals_list[i] - 1):
            s_complete[(i, j)] = model.s[i, j]
    else:
        s_fixed = constant_to_vector(float(x[i]), num_intervals_list[i] - 1) 
        for j in range(num_intervals_list[i] - 1):
            s_complete[(i, j)] = s_fixed[j] 
# Calculate new_score
marginal_scores = []
for i in range(num_features):
    dot_product_expr = sum(delta_u_opt[i][j] * s_complete[(i, j)] for j in range(num_intervals_list[i] - 1)) + 20
    marginal_scores.append(dot_product_expr)
p = lambda_opt
if p == 0:
    new_score = (np.prod(marginal_scores)) ** (1 / len(marginal_scores))
else:
    new_score = (sum(score**p for score in marginal_scores) / len(marginal_scores)) ** (1 / p) 

for i in range(num_features):
    for j in range(i + 1, num_features):
        interaction_term = sum(
            torch.dot(v_opt[i][k], v_opt[j][l]).item() * s_complete[(i, k)] * s_complete[(j, l)]
            for k in range(num_intervals_list[i] - 1)
            for l in range(num_intervals_list[j] - 1)
        )
        new_score += interaction_term
for i in non_fixed_features:
    model.add_component(
        f'constraint_ξ_{i}_lower', 
        Constraint(expr=-M * model.ξ[i] <= x[i] - model.x0[i])
    )
    model.add_component(
        f'constraint_ξ_{i}_upper', 
        Constraint(expr=x[i] - model.x0[i] <= M * model.ξ[i])
    )
    
unique_labels = np.unique(labels_train)
sorted_scores = sorted(train_scores)
intervals = []
intervals.append((-float('inf'), sorted_scores[0]))
for i in range(len(sorted_scores)-1):
    intervals.append((sorted_scores[i], sorted_scores[i+1]))
intervals.append((sorted_scores[-1], float('inf')))
def compute_train_reliability(train_scores, labels_train):
    train_reliability = []
    for i in range(len(train_scores)):
        num, denom = 0.0, 0.0
        for j in range(len(train_scores)):
            if i == j:
                continue
            if labels_train[j] != labels_train[i]:
                denom += 1.0
                if ((train_scores[j] > train_scores[i] and labels_train[j] > labels_train[i]) or
                    (train_scores[j] < train_scores[i] and labels_train[j] < labels_train[i])):
                    num += 1.0
        rel = num / denom if denom != 0 else 0.0
        train_reliability.append(rel)
    return train_reliability
train_reliability = compute_train_reliability(train_scores, labels_train)
def predict_label_at_score(test_score_value):
    label_confidences = {}
    for h in unique_labels:
        conf_sum = 0.0
        denom_sum = 0.0
        for idx, train_label in enumerate(labels_train):
            if train_label != h:
                denom_sum += 1
                if train_label > h and (train_scores[idx] > test_score_value):
                    conf_sum += train_reliability[idx]
                elif train_label < h and (train_scores[idx] < test_score_value):
                    conf_sum += train_reliability[idx]
        conf_h = conf_sum / denom_sum if denom_sum != 0 else 0.0
        label_confidences[h] = conf_h

    pred_label = max(label_confidences, key=label_confidences.get)
    return pred_label
# For each interval, conduct "endpoints + midpoint" testing; extract single-point intervals if endpoint predictions differ from midpoint
intervals_ex = [] 
for (low, high) in intervals:
    import math
    if low == -float('inf'):
        test_mid = sorted_scores[0] - 1 
    elif high == float('inf'):
        test_mid = sorted_scores[-1] + 1
    else:
        test_mid = 0.5 * (low + high)
    mid_label = predict_label_at_score(test_mid)
    if math.isinf(low) and low < 0:
        test_low = sorted_scores[0] - 1
    else:
        test_low = low

    if math.isinf(high) and high > 0:
        test_high = sorted_scores[-1] + 1
    else:
        test_high = high

    low_label  = predict_label_at_score(test_low)
    high_label = predict_label_at_score(test_high)
    eps = 1e-9
    current_start = low
    if not math.isinf(low):
        if low_label != mid_label:
            intervals_ex.append((low, low, low_label))
            current_start = low + eps  
    if not math.isinf(high):
        if mid_label != high_label:
            sub_high = high - eps
            if sub_high < current_start:
                sub_high = current_start
            intervals_ex.append((current_start, sub_high, mid_label))
            intervals_ex.append((high, high, high_label))
        else:
            intervals_ex.append((current_start, high, mid_label))
    else:
        intervals_ex.append((current_start, high, mid_label))
intervals = []
interval_labels = []
for (low, high, lbl) in intervals_ex:
    intervals.append((low, high))
    interval_labels.append(lbl)

# Merge adjacent intervals with the same label
merged_intervals = []
merged_labels = []
if len(intervals) > 0:
    combined = sorted(zip(intervals, interval_labels), key=lambda x: x[0][0])
    current_low, current_high = combined[0][0]
    current_label = combined[0][1]
    for i in range(1, len(combined)):
        (low, high) = combined[i][0]
        lbl = combined[i][1]
        if math.isclose(low, current_high, abs_tol=1e-9) and lbl == current_label:
            current_high = high
        elif (low <= current_high) and lbl == current_label:
            if high > current_high:
                current_high = high
        else:
            merged_intervals.append((current_low, current_high))
            merged_labels.append(current_label)
            current_low, current_high, current_label = low, high, lbl

    merged_intervals.append((current_low, current_high))
    merged_labels.append(current_label)

# Find intervals corresponding to the target_label
target_intervals = [(low, high) for (low, high), lbl in zip(merged_intervals, merged_labels)
                    if lbl == target_label]
if len(target_intervals) == 0:
    model.add_component('no_target_interval_constr', Constraint(expr=(new_score <= -M)))
else:
# Use "OR" logic to constrain new_score to lie within any one of the target intervals
    model.target_interval_vars = Var(range(len(target_intervals)), within=Binary)
    for i, (low, high) in enumerate(target_intervals):
        model.add_component(
            f'target_low_constr_{i}',
            Constraint(expr=new_score >= low - M * (1 - model.target_interval_vars[i]))
        )
        model.add_component(
            f'target_high_constr_{i}',
            Constraint(expr=new_score <= high + M * (1 - model.target_interval_vars[i]))
        )

    model.add_component(
        'target_or_logic_constr',
        Constraint(expr=sum(model.target_interval_vars[i] for i in range(len(target_intervals))) == 1)
    )
print("Before merging:")
for (low, high), lbl in zip(intervals, interval_labels):
    print(f"Interval [{low}, {high}) -> Label = {lbl}")
print("\nAfter merging:")
for (low, high), lbl in zip(merged_intervals, merged_labels):
    print(f"Interval [{low}, {high}) -> Label = {lbl}")
# Define the objective function
model.objective = Objective(
    expr=sum((x[i] - model.x0[i]) ** 2 for i in non_fixed_features) 
        + A * sum(model.ξ[i] for i in non_fixed_features),
    sense=minimize
)

solver = SolverFactory('scip')
solver.options['limits/gap'] = 0.0    #global optimal
solver.options['limits/time'] = 12000
solver.options['display/verblevel'] = 4    
solver.options['constraints/nonlinear/linearize'] = True
solver.options['constraints/nonlinear/convexify'] = True
solver.options['presolving/maxrounds'] = 0
solver.options['heuristics/undercover/freq'] = -1
solver.options['separating/convexproj/freq'] = 10
total_vars = 0
total_binary_vars = 0
total_continuous_vars = 0
for v in model.component_objects(Var, active=True):
    for idx in v:
        total_vars += 1
        variable = v[idx]
        if variable.domain is Binary:
            total_binary_vars += 1
        elif variable.domain is Reals:
            total_continuous_vars += 1
print(f"Total variables: {total_vars}")
print(f"Total binary variables: {total_binary_vars}")
print(f"Total continuous variables: {total_continuous_vars}")
total_constrs = 0
for c in model.component_objects(Constraint, active=True):
    for _ in c:
        total_constrs += 1
print(f"Total constraints: {total_constrs}")


results = solver.solve(model, tee=True)
feature_names = [
    "Age",
    "Heart failure 0-1",
    "Interstitial lung disease 0-1",
    "Pneumonia 0-1",
    "Altered consciousness 0-1",
    "Diastolic blood pressure",
    "Heart rate",
    "Hemoglobin",
    "PH",
    "PaCO2",
    "Serum albumin",
    "Blood urea nitrogen"
]
# Output results
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found.")
    for i in non_fixed_features:
        print(f"x0[{i}] =", model.x0[i].value)
# Convert decision variables to piecewise vectors and back to real feature values
    x0_optimal_vectors = []
    
    for i in range(num_features):
        if i in non_fixed_features:
            x0_optimal_vectors.append(constant_to_vector(model.x0[i].value, num_intervals_list[i] - 1))
        else:
            x0_optimal_vectors.append(constant_to_vector(float(x[i]), num_intervals_list[i] - 1))
    for i in range(num_features):
        min_val, max_val = feature_min_max[i]
        interval_size = (max_val - min_val) / (num_intervals_list[i] - 1)
        x0_real_value = sum(x0_optimal_vectors[i][j] for j in range(len(x0_optimal_vectors[i]))) * interval_size + min_val
# Extract original feature values
        original_real_value = sum(sample_piecewise_vectors[i][j].item() for j in range(len(sample_piecewise_vectors[i]))) * interval_size + min_val
        feature_name = feature_names[i]        
        print(f"Feature ({feature_name}): Original value = {original_real_value}, Optimal value = {x0_real_value}, Change = {x0_real_value - original_real_value}")

else:
    print("Optimization was unsuccessful.")
