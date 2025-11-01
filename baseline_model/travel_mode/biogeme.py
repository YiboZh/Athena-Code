import pandas as pd
import numpy as np
import biogeme.biogeme as bio
from biogeme import models
from biogeme.data.swissmetro import (
    read_data,
    PURPOSE,
    CHOICE,
    GA,
    TRAIN_CO,
    SM_CO,
    SM_AV,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    CAR_AV_SP,
    GROUP,
)
from biogeme.expressions import Beta, Variable, bioDraws, exp, log, MonteCarlo

# Read and filter the data
database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

# Define the parameters for the utility functions
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Compute the costs depending on the ownership of a GA
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Define utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Part 1: Model Estimation
def estimate_model(database):
    """Estimate the model parameters using maximum likelihood estimation"""
    logprob = models.loglogit(V, av, CHOICE)
    weight = 8.890991e-01 * (1.0 * (GROUP == 2) + 1.2 * (GROUP == 3))
    formulas = {'log_like': logprob, 'weight': weight}
    biogeme = bio.BIOGEME(database, formulas, parameters=None)
    biogeme.save_iterations = False
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    results = biogeme.estimate()
    print(f"Log likelihood at convergence: {results.data.logLike:.3f}")
    return results

# Part 2: Prediction - Fixed to handle parameter names correctly
def predict_modes(database, results):
    """Predict mode choices based on the estimated model"""
    # Create the dictionary of beta parameters
    betas = results.get_beta_values()
    print("Available parameters in the model:", betas.keys())

    # Calculate the utilities for each alternative using the estimated parameters
    # Use the exact parameter names as they appear in the results
    # u1 = betas['ASC_TRAIN'] + betas['B_TIME'] * TRAIN_TT_SCALED.values + betas['B_COST'] * TRAIN_COST_SCALED.values
    # u2 = betas['ASC_SM'] + betas['B_TIME'] * SM_TT_SCALED.values + betas['B_COST'] * SM_COST_SCALED.values
    # u3 = betas['ASC_CAR'] + betas['B_TIME'] * CAR_TT_SCALED.values + betas['B_COST'] * CAR_CO_SCALED.values
    u1 = betas['ASC_TRAIN'] + betas['B_TIME'] * TRAIN_TT_SCALED.get_value_c(database=database, prepare_ids=True) + betas['B_COST'] * TRAIN_COST_SCALED.get_value_c(database=database, prepare_ids=True)
    u2 = betas['ASC_SM'] + betas['B_TIME'] * SM_TT_SCALED.get_value_c(database=database, prepare_ids=True) + betas['B_COST'] * SM_COST_SCALED.get_value_c(database=database, prepare_ids=True)
    u3 = betas['ASC_CAR'] + betas['B_TIME'] * CAR_TT_SCALED.get_value_c(database=database, prepare_ids=True) + betas['B_COST'] * CAR_CO_SCALED.get_value_c(database=database, prepare_ids=True)

    # Create a DataFrame to store the utilities and probabilities
    pred_df = database.data.copy()
    # pred_df = pd.DataFrame(pred_df)
    # u1_values = u1.get_value_c(database=database, prepare_ids=True)

    print(type(u1), u1.shape)
    # Add utilities to DataFrame
    pred_df['U_TRAIN'] = u1
    pred_df['U_SM'] = u2
    pred_df['U_CAR'] = u3

    # Calculate exponential of utilities
    pred_df['EXP_U_TRAIN'] = np.exp(pred_df['U_TRAIN'])
    pred_df['EXP_U_SM'] = np.exp(pred_df['U_SM'])
    pred_df['EXP_U_CAR'] = np.exp(pred_df['U_CAR'])

    # Account for availability
    pred_df['EXP_U_TRAIN'] = pred_df['EXP_U_TRAIN'] * pred_df['TRAIN_AV_SP']
    pred_df['EXP_U_SM'] = pred_df['EXP_U_SM'] * pred_df['SM_AV']
    pred_df['EXP_U_CAR'] = pred_df['EXP_U_CAR'] * pred_df['CAR_AV_SP']

    # Calculate sum of exp(utility) * availability
    pred_df['SUM_EXP_U'] = pred_df['EXP_U_TRAIN'] + pred_df['EXP_U_SM'] + pred_df['EXP_U_CAR']

    # Calculate probabilities
    pred_df['P_TRAIN'] = pred_df['EXP_U_TRAIN'] / pred_df['SUM_EXP_U']
    pred_df['P_SM'] = pred_df['EXP_U_SM'] / pred_df['SUM_EXP_U']
    pred_df['P_CAR'] = pred_df['EXP_U_CAR'] / pred_df['SUM_EXP_U']

    # Determine the predicted choice (highest probability)
    pred_df['PREDICTED_CHOICE'] = pred_df[['P_TRAIN', 'P_SM', 'P_CAR']].idxmax(axis=1)
    pred_df['PREDICTED_CHOICE'] = pred_df['PREDICTED_CHOICE'].map({'P_TRAIN': 1, 'P_SM': 2, 'P_CAR': 3})

    return pred_df

# Part 3: Model Evaluation
def evaluate_model(predicted_df, actual_choice_column='CHOICE'):
    """Evaluate the model performance"""
    # Calculate accuracy
    correct_predictions = (predicted_df['PREDICTED_CHOICE'] == predicted_df[actual_choice_column]).sum()
    total_observations = len(predicted_df)
    accuracy = correct_predictions / total_observations

    # Confusion matrix
    confusion_matrix = pd.crosstab(
        predicted_df[actual_choice_column],
        predicted_df['PREDICTED_CHOICE'],
        rownames=['Actual'],
        colnames=['Predicted']
    )

    # Mode-specific metrics
    mode_metrics = {}
    for mode in [1, 2, 3]:
        true_pos = ((predicted_df[actual_choice_column] == mode) &
                     (predicted_df['PREDICTED_CHOICE'] == mode)).sum()
        false_pos = ((predicted_df[actual_choice_column] != mode) &
                      (predicted_df['PREDICTED_CHOICE'] == mode)).sum()
        false_neg = ((predicted_df[actual_choice_column] == mode) &
                      (predicted_df['PREDICTED_CHOICE'] != mode)).sum()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        mode_name = {1: 'TRAIN', 2: 'SWISSMETRO', 3: 'CAR'}[mode]
        mode_metrics[mode_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'mode_metrics': mode_metrics
    }

# Part 4: Scenario Analysis
def scenario_analysis(database, results, scenarios):
    """Perform scenario analysis by modifying input variables"""
    scenario_results = {}

    for scenario_name, modifications in scenarios.items():
        # Create a copy of the database for this scenario
        from copy import deepcopy
        scenario_db = deepcopy(database)

        # Apply modifications to the database
        for var_name, modification in modifications.items():
            # The modification can be a scalar or a function
            if callable(modification):
                scenario_db.data[var_name] = modification(scenario_db.data[var_name])
            else:
                scenario_db.data[var_name] = modification

        # Predict mode choices for this scenario
        predicted_df = predict_modes(scenario_db, results)

        # Calculate mode shares
        mode_shares = {
            'TRAIN': (predicted_df['PREDICTED_CHOICE'] == 1).mean(),
            'SWISSMETRO': (predicted_df['PREDICTED_CHOICE'] == 2).mean(),
            'CAR': (predicted_df['PREDICTED_CHOICE'] == 3).mean()
        }

        scenario_results[scenario_name] = mode_shares

    return scenario_results

# Function to check parameter names
def check_parameters(results):
    """Print the parameter names from the estimation results"""
    betas = results.get_beta_values()
    print("Model parameters:")
    for param_name, value in betas.items():
        print(f"  {param_name}: {value:.4f}")
    return betas.keys()

# Main execution flow
    # try:
# 1. Estimate the model
results = estimate_model(database)
print("\nEstimated parameters:")
param_names = check_parameters(results)

# 2. Check if we have all required parameters before prediction
required_params = ['ASC_TRAIN', 'ASC_SM', 'ASC_CAR', 'B_TIME', 'B_COST']
missing_params = [p for p in required_params if p not in param_names]

if missing_params:
    print(f"WARNING: Missing parameters: {missing_params}")
    print("Please rename parameters or update the prediction function.")
else:
    # 3. Predict mode choices
    predicted_df = predict_modes(database, results)

    # 4. Evaluate the model
    evaluation = evaluate_model(predicted_df)
    print(f"\nModel accuracy: {evaluation['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(evaluation['confusion_matrix'])
    print("\nMode-specific metrics:")
    for mode, metrics in evaluation['mode_metrics'].items():
        print(f"{mode}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

    # 5. Perform scenario analysis
    scenarios = {
        'baseline': {},  # No changes
        'increase_train_cost': {
            'TRAIN_COST_SCALED': lambda x: x * 1.2  # 20% increase in train cost
        },
        'decrease_sm_travel_time': {
            'SM_TT_SCALED': lambda x: x * 0.8  # 20% decrease in Swissmetro travel time
        },
        'increase_car_cost': {
            'CAR_CO_SCALED': lambda x: x * 1.5  # 50% increase in car cost (e.g., due to carbon tax)
        }
    }

    scenario_results = scenario_analysis(database, results, scenarios)
    print("\nScenario Analysis - Mode Shares:")
    for scenario, mode_shares in scenario_results.items():
        print(f"\n{scenario}:")
        for mode, share in mode_shares.items():
            print(f"  {mode}: {share:.4f}")
