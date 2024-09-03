import os
import sys
import time
from IPython.display import display
project_path = os.getcwd()
sys.path.insert(0, project_path)

# Set the path for objective function
objective_path = os.path.join(project_path, '9in_10out')
sys.path.insert(0, objective_path)

import numpy as np
import pandas as pd
import torch
from nextorch import plotting, bo, doe, utils, io, parameter

def biomarker_inflammatory_response(X):
    """
    Computes the inflammatory response biomarker based on all input variables.
    Realistically models how inflammatory response might be influenced by various chemical properties.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.5 * np.log1p(molecular_weight) +
            0.3 * logp +
            0.2 * h_donors -
            0.4 * h_acceptors +
            0.1 * np.sqrt(tpsa) -
            0.2 * mol_refractivity +
            0.1 * np.log1p(rotatable_bonds) +
            0.05 * aromatic_rings -
            0.03 * heavy_atoms)


def biomarker_oxidative_stress(X):
    """
    Computes the oxidative stress biomarker based on all input variables.
    Models the impact of chemical properties on oxidative stress more realistically.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.4 * np.sin(logp) +
            0.3 * np.log1p(molecular_weight) +
            0.2 * np.tanh(h_donors / 10) +
            0.1 * np.exp(-h_acceptors / 50) -
            0.2 * mol_refractivity +
            0.1 * np.sqrt(rotatable_bonds + 1) +
            0.05 * aromatic_rings -
            0.03 * heavy_atoms)


def biomarker_neuroprotection(X):
    """
    Computes the neuroprotection biomarker based on all input variables.
    Incorporates factors that could realistically impact neuroprotection.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.5 * np.tanh(molecular_weight / 200) +
            0.3 * np.log1p(h_donors) +
            0.2 * np.sqrt(h_acceptors) -
            0.1 * np.exp(-tpsa / 20) +
            0.05 * mol_refractivity -
            0.03 * rotatable_bonds +
            0.02 * aromatic_rings +
            0.01 * heavy_atoms)


def biomarker_mitochondrial_function(X):
    """
    Computes the mitochondrial function biomarker based on all input variables.
    Reflects how these properties might influence mitochondrial activity.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.4 * np.sqrt(molecular_weight) / (tpsa + 1) +
            0.3 * np.sin(logp / 5) -
            0.2 * np.log1p(h_donors) +
            0.1 * np.exp(-h_acceptors / 100) +
            0.05 * mol_refractivity / (rotatable_bonds + 1) +
            0.02 * aromatic_rings -
            0.01 * heavy_atoms)


def biomarker_synaptic_function(X):
    """
    Computes the synaptic function biomarker based on all input variables.
    Provides a realistic model of how chemical properties could affect synaptic function.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.5 * np.log1p(h_donors) * np.exp(-logp / 20) +
            0.3 * np.sin(molecular_weight / 50) +
            0.2 * np.sqrt(tpsa) -
            0.1 * np.exp(-rotatable_bonds / 10) +
            0.05 * aromatic_rings +
            0.02 * heavy_atoms)


def biomarker_motor_function(X):
    """
    Computes the motor function biomarker based on all input variables.
    Models realistic relationships between chemical properties and motor function.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.4 * np.sin(molecular_weight / 25) * np.sqrt(heavy_atoms + 1) +
            0.3 * np.sign(logp) * np.log1p(np.abs(logp)) -
            0.2 * np.exp(-h_donors / 30) +
            0.1 * np.tanh(h_acceptors / 50) +
            0.05 * mol_refractivity / (rotatable_bonds + 1))


def biomarker_cognitive_function(X):
    """
    Computes the cognitive function biomarker based on all input variables.
    Realistically incorporates how chemical properties might affect cognitive function.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.5 * np.cos(logp / 5) * np.sqrt(tpsa + 1) -
            0.3 * np.exp(-molecular_weight / 100) +
            0.2 * np.log1p(h_donors) +
            0.1 * np.tanh(h_acceptors / 10) +
            0.05 * mol_refractivity / (rotatable_bonds + 1) -
            0.02 * aromatic_rings +
            0.01 * heavy_atoms)


def biomarker_cardiac_function(X):
    """
    Computes the cardiac function biomarker based on all input variables.
    Models how various chemical properties can realistically influence cardiac function.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.4 * np.exp(-tpsa / 100) +
            0.3 * np.tanh(h_donors / 20) +
            0.2 * np.log1p(molecular_weight) -
            0.1 * np.sin(logp / 10) +
            0.05 * mol_refractivity -
            0.02 * rotatable_bonds +
            0.01 * aromatic_rings +
            0.01 * heavy_atoms)


def biomarker_liver_function(X):
    """
    Computes the liver function biomarker based on all input variables.
    Reflects how liver function might be influenced by chemical properties.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.4 * np.log1p(h_donors) +
            0.3 * np.exp(-mol_refractivity / 100) +
            0.2 * np.sin(tpsa / 10) -
            0.1 * np.log1p(rotatable_bonds) +
            0.05 * aromatic_rings -
            0.02 * heavy_atoms +
            0.01 * molecular_weight)


def biomarker_kidney_function(X):
    """
    Computes the kidney function biomarker based on all input variables.
    Models realistic relationships between chemical properties and kidney function.
    """
    molecular_weight = X[:, 0]
    logp = X[:, 1]
    h_donors = X[:, 2]
    h_acceptors = X[:, 3]
    tpsa = X[:, 4]
    mol_refractivity = X[:, 5]
    rotatable_bonds = X[:, 6]
    aromatic_rings = X[:, 7]
    heavy_atoms = X[:, 8]

    return (0.4 * np.sin(molecular_weight / 50) +
            0.3 * np.sign(logp) * np.log1p(np.abs(logp)) +
            0.2 * np.tanh(h_donors / 10) -
            0.1 * np.exp(-h_acceptors / 20) +
            0.05 * mol_refractivity / (rotatable_bonds + 1) +
            0.02 * aromatic_rings -
            0.01 * heavy_atoms)


def compute_biomarkers(X):
    """
    Computes all biomarkers based on the input variables and combines them into a single array.
    """
    X = X.astype(float)
    inflammatory_response = -biomarker_inflammatory_response(X)
    oxidative_stress = -biomarker_oxidative_stress(X)
    neuroprotection = biomarker_neuroprotection(X)
    mitochondrial_function = biomarker_mitochondrial_function(X)
    synaptic_function = biomarker_synaptic_function(X)
    motor_function = biomarker_motor_function(X)
    cognitive_function = biomarker_cognitive_function(X)
    cardiac_function = biomarker_cardiac_function(X)
    liver_function = biomarker_liver_function(X)
    kidney_function = biomarker_kidney_function(X)

    # Combine all the biomarker outputs into a single numpy array
    responses = np.column_stack((
        inflammatory_response,
        oxidative_stress,
        neuroprotection,
        mitochondrial_function,
        synaptic_function,
        motor_function,
        cognitive_function,
        cardiac_function,
        liver_function,
        kidney_function
    ))

    return responses

def run_experiment(n_init, n_trials):
    ##%% Initialize a multi-objective Experiment object
    # Set its name, the files will be saved under the folder with the same name
    Exp_9_10 = bo.EHVIMOOExperiment('mice_9_10_open')

    # Set the type and range for each parameter
    par_mw = parameter.Parameter(x_type='continuous', x_range=[20, 1000])
    par_logP = parameter.Parameter(x_type='continuous', x_range=[-10, 10])
    par_hbd = parameter.Parameter(x_type='ordinal', x_range=[0, 15], interval=1)
    par_hba = parameter.Parameter(x_type='ordinal', x_range=[0, 20], interval=1)
    par_tpsa = parameter.Parameter(x_type='continuous', x_range=[0, 3000])
    par_mr = parameter.Parameter(x_type='continuous', x_range=[20, 150])
    par_rb = parameter.Parameter(x_type='ordinal', x_range=[0, 15], interval=1)
    par_ar = parameter.Parameter(x_type='ordinal', x_range=[0, 10], interval=1)
    par_ha = parameter.Parameter(x_type='ordinal', x_range=[1, 100], interval=1)

    parameters = [par_mw, par_logP, par_hbd, par_hba, par_tpsa, par_mr, par_rb, par_ar, par_ha]
    Exp_9_10.define_space(parameters)

    import biomarker_functions as bi

    # Choose whether to standardize output (using data from LHC sample)
    stand_flag = False

    if stand_flag:
        X_unit = doe.latin_hypercube(n_dim=9, n_points=500, seed=1)
        X_real = utils.encode_to_real_ParameterSpace(X_unit, Exp_9_10.parameter_space)
        Y = bi.compute_biomarkers(X_real)
        bio_mean = pd.DataFrame(Y).mean().to_list()
        bio_std = pd.DataFrame(Y).std().to_list()
        objective_func = lambda X: (bi.compute_biomarkers(X) - bio_mean) / bio_std
    else:
        objective_func = bi.compute_biomarkers

    # Define the design space
    X_name_list = [
        "Molecular Weight",
        "LogP",
        "Hydrogen Bond Donors",
        "Hydrogen Bond Acceptors",
        "Topological Polar Surface Area",
        "Molar Refractivity",
        "Rotatable Bonds",
        "Aromatic Rings",
        "Heavy Atoms"
    ]

    Y_name_list = [
        "Inflammatory Response",
        "Oxidative Stress",
        "Neuroprotection",
        "Mitochondrial Function",
        "Synaptic Function",
        "Motor Function",
        "Cognitive Function",
        "Cardiac Function",
        "Liver Function",
        "Kidney Function"
    ]

    var_names = X_name_list + Y_name_list

    # Get the information of the design space
    n_dim = len(X_name_list)  # the dimension of inputs
    n_objective = len(Y_name_list)  # the dimension of outputs

    ##%% Initial Sampling
    # Latin hypercube design
    X_init = doe.latin_hypercube(n_dim=n_dim, n_points=n_init)
    # Get the initial responses
    Y_init = bo.eval_objective_func_encoding(X_init, Exp_9_10.parameter_space, objective_func)

    # Import the initial data
    Exp_9_10.input_data(X_init,
                        Y_init,
                        X_names=X_name_list,
                        Y_names=Y_name_list,
                        unit_flag=True,
                        standardized=stand_flag)

    # Set the optimization specifications
    if stand_flag:
        ref_point = [-3.0] * 10
    else:
        ref_point = [5, 5, 3, 0, 0, 0, 1, 3, 1, 0.5]

    Exp_9_10.set_ref_point(ref_point)
    Exp_9_10.set_optim_specs(objective_func=objective_func,
                             maximize=True)

    for i in range(n_trials):
        if i % 20 == 0:
            print("{} trials completed".format(i))
        # Generate the next experiment point
        X_new, X_new_real, acq_func = Exp_9_10.generate_next_point(n_candidates=1)
        # Get the reponse at this point
        Y_new_real = objective_func(X_new_real)
        # Retrain the model by input the next point into Exp object
        Exp_9_10.run_trial(X_new, X_new_real, Y_new_real)

    # Results:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    sample_size = Exp_9_10.Y_real.shape[0]
    front_size = Exp_9_10.get_optim()[0].shape[0]
    final_hypervolume_obj = Hypervolume(torch.tensor(ref_point))
    final_hypervolume = final_hypervolume_obj.compute(torch.from_numpy(Exp_9_10.get_optim()[0]))
    return front_size / sample_size, final_hypervolume
