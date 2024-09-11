

# Import libraries
import numpy as np


def MonteCarlo(asset,val,weight,divyield,corrs,vol,sims,T,K,r):

    # Create diagnonal matrix of volatility values
    vsdiag = np.diag(vol)
    # Compute covariance matrix
    cov = vsdiag @ (corrs @ vsdiag)

    #Cholesky decomposition
    M = np.linalg.cholesky(cov)

    # Set up arrays of zeros
    mu = np.zeros(asset)
    multiplier = np.sqrt(T)*M
    Z = np.zeros(asset)
    logS = np.zeros(asset)

    # For loop to simulate and get payoffs - do not need dt because using
    # exact solution to GBM
    for i in range(asset):
        mu[i] = np.log(val[i]) + (r - divyield[i] - 0.5 * cov[i,i]) * T
   
    # Initialize sumval
    sumval = 0

    for h in range(sims):
        # Initialize basket value
        basketval = 0
    
        # Get random variables
        Z = np.random.normal(0, 1, asset)
            
        for i in range(asset):
            logS[i] = mu[i]
            
            for j in range(nAssets):
                logS[i] = logS[i] + multiplier[i,j] * Z[j]
            
            basketval = basketval + weights[i] * np.exp(logS[i])

        # option value and sum
        optionval = max(basketval - K,0)
        sumval = sumval + optionval

    print(np.exp(-r * T) * (sumval/nSims)) 
def MonteCarloWithAnti(asset,val,weight,divyield,corrs,vol,sims,T,K,r):

    # Create diagonal matrix of volatility values
    vsdiag = np.diag(vol)
    # Compute covariance matrix
    cov = vsdiag @ (corrs @ vsdiag)

    # Cholesky decomposition
    M = np.linalg.cholesky(cov)

    # Set up arrays of zeros
    mu = np.zeros(asset)
    multiplier = np.sqrt(T) * M

    # For loop to compute the drift component for each asset
    for i in range(asset):
        mu[i] = np.log(val[i]) + (r - divyield[i] - 0.5 * cov[i, i]) * T

    # Initialize sumval
    sumval = 0

    for h in range(sims // 2):
        # Initialize basket values
        basketval1 = 0
        basketval2 = 0

        # Get random variables and their antithetic counterparts
        Z = np.random.normal(0, 1, asset)
        Z_antithetic = -Z

        logS1 = np.zeros(asset)
        logS2 = np.zeros(asset)

        for i in range(asset):
            logS1[i] = mu[i]
            logS2[i] = mu[i]
            for j in range(asset):
                logS1[i] += multiplier[i, j] * Z[j]
                logS2[i] += multiplier[i, j] * Z_antithetic[j]

            basketval1 += weights[i] * np.exp(logS1[i])
            basketval2 += weights[i] * np.exp(logS2[i])

        # Option values and sum
        optionval1 = max(basketval1 - K, 0)
        optionval2 = max(basketval2 - K, 0)
        sumval += optionval1 + optionval2

    # Compute and print the discounted average option value
    print(np.exp(-r * T) * (sumval / sims))
# Parameters for valuation:
nAssets = 7
Svals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
weights = [0.10, 0.15, 0.15, 0.05, 0.20, 0.10, 0.25]
divs = [0.0169, 0.0239, 0.0136, 0.0192, 0.0081, 0.0362, 0.0166]
vs = [0.1155, 0.2068, 0.1453, 0.1799, 0.1559, 0.1462, 0.1568]
nSims = 50000
T1 = 10    
K1 = 1.0
r1 = 0.063


corr = np.array([[1.00, 0.35, 0.10, 0.27, 0.04, 0.17, 0.71],\
                [0.35, 1.00, 0.39, 0.27, 0.50, -0.08, 0.15],\
                [0.10, 0.39, 1.00, 0.53, 0.70, -0.23, 0.09],\
                [0.27, 0.27, 0.53, 1.00, 0.46, -0.22, 0.32],\
                [0.04, 0.50, 0.70, 0.46, 1.00, -0.29, 0.13],\
                [0.17, -0.08, -0.23, -0.22, -0.29, 1.00, -0.03],\
                [0.71, 0.15, 0.09, 0.32, 0.13, -0.03, 1.00]])
MonteCarlo(nAssets,Svals,weights,divs,corr,vs,nSims,T1,K1,r1)
MonteCarloWithAnti(nAssets,Svals,weights,divs,corr,vs,nSims,T1,K1,r1)
