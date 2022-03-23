import pymc3 as pm
import numpy as np
import theano as T
# from aesara import tensor as at


def bayesian_lr_with_regularized_horseshoe(features: np.array, target: np.array, nonzero_features: int) -> pm.model.Model:
    
    X_t = T.shared(features)
    n_samples, n_features = features.shape
    
    with pm.Model() as reg_horseshoe:
        
        # sigma prior
        σ = pm.HalfNormal('sigma', 2.5)
    
        # tau prior
        τ = pm.HalfStudentT('tau', nu=2, sigma=(nonzero_features / (n_features - nonzero_features)) * σ / np.sqrt(n_samples))
        
        # lambda prior
        λ = pm.HalfStudentT('lambda', nu=5, shape=n_features)
        
        # c^2 prior
        c2 = pm.InverseGamma('c2', alpha=1, beta=1)
        
        # defining lambda_tilde
        λ_ = λ * np.sqrt(c2 / (c2 + (τ**2) * (λ**2)))
        
        # Beta prior
        z = pm.Normal('z', 0., 1., shape=n_features)
        β = pm.Deterministic('beta', z * τ * λ_)
        
        # Intercept prior
        βₒ = pm.Normal('beta_0', mu=0, sigma=10)
        
        μ = X_t.dot(β) + βₒ 
        
        # observations
        y_obs = pm.Normal('obs', mu=μ, sigma=σ, observed=target)
        
        return reg_horseshoe