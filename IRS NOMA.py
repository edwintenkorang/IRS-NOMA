import numpy as np
from scipy.optimize import minimize

# Constants and parameters
Ko = 3 #rician constant
N = 30 #Number of IRS elements
pl_edge_bs = 3.5 #path loss from base station to cell-edge user
pl_irs_bs = 3 #path loss from base station to IRS
pl_centre_bs = 2.5 #path loss from cell-centre to base station
pl_irs_edge = 2 #path loss from cell-edge user to IRS
Rb = 500 #distance of base station to cell-edge
Rc = 200 #distance of cell-centre
Ri = 50 #radius of IRS
AWGN_dB = -99  # gaussian noise
SNR_threshold = 1 #sinr_threshold

# Generate random channel values
tC_i = np.random.randn() + 1j * np.random.randn() 
tE_i = np.random.randn() + 1j * np.random.randn()
tI_i = np.random.randn(N) + 1j * np.random.randn(N)
tE_I = np.random.randn(N) + 1j * np.random.randn()
to = np.ones(N)
randomPhases = 2 * np.pi * np.random.rand(N)
theta = np.diag(randomPhases)
covariance_matrix = np.zeros((N, N))

# Calculate the covariance matrix
for i in range(N):
    for l in range(N):
        covariance_matrix[i, l] = 0.5**abs(i - l)

# Define the optimization objective function
def objective(x):
    bc, be = x
    
    # Define the channels hc and he
    hc = (Rc**(-pl_centre_bs/2)) * tC_i #direct cell-centre - bs channel
    he = (Rb**(-pl_edge_bs/2)) * tE_i #direct cell-edge - bs channel
    
    phi = np.diag(2 * np.pi * np.random.rand(N))
    heI = (Ri**(-pl_edge_bs/2)) * (np.sqrt(Ko / (Ko + 1)) * to + np.sqrt(1 / (Ko + 1)) * np.dot(covariance_matrix, tE_I)) #cascaded cell-center IS channel
    
    SINR_edge = (abs(hc)**2 * bc**2) / ((abs(he + np.dot(np.conjugate(tI_i).T, np.dot(phi, heI)))**2 * be**2) + 10**(AWGN_dB/10))
    SINR_center = ((abs(he + np.dot(np.conjugate(tI_i).T, np.dot(phi, heI)))**2 * be**2) + 10**(AWGN_dB/10)) / (10**(AWGN_dB/10))
    
    lambda_val = (10**(AWGN_dB/10) + 10**(SNR_threshold/10) + 10**(SNR_threshold/10)) / abs(hc)
    sly = (10**(AWGN_dB/10) * 10**(SNR_threshold/10)) / abs(hc)
    
    P = (lambda_val * abs(he + np.dot(np.conjugate(tI_i).T, np.dot(phi, heI)))**2 + 10**(SNR_threshold/10) * (10**(AWGN_dB/10))**2) / (abs(he + np.dot(np.conjugate(tI_i).T, np.dot(phi, heI)))**2) + sly
    return P

# Initial guess for bc and be
x0 = np.random.rand(2)

# Bounds for bc and be
bounds = [(0, 1), (0, 1)]

# Perform optimization
result = minimize(objective, x0, bounds=bounds)

# Extract optimal values and minimum power
optimal_bc, optimal_be = result.x
minimum_power = result.fun

print(f'Optimal Power Allocation:')
print(f'bc: {optimal_bc}')
print(f'be: {optimal_be}')
print(f'Minimum Power (P): {minimum_power}')