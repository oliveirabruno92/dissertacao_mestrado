import pymc3 as pm
import numpy as np
import theano as T


def bayesian_lr(features: np.array, target: np.array) -> pm.model.Model:
    
    X_t = T.shared(features)
    n_features = features.shape[1]
    
    with pm.Model() as bayes_lr:
        
        # prior for intercept 
        βₒ = pm.Normal('intercept', mu=0, sigma=10)
        
        # priors for coefficients
        β = pm.Normal('beta', mu=0, sigma=10, shape=n_features) 
                            
        # prior for standard deviation
        σ = pm.HalfNormal('sigma', sigma=1)
        
        # observations
        μ = X_t.dot(β) + βₒ 
        y_obs = pm.Normal('obs', mu=μ, sigma=σ, observed=target)
    
    return bayes_lr


