import numpy as np

# Define the biomarker functions with more realistic and complex relationships

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