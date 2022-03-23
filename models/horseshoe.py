import pymc3 as pm
import numpy as np
import theano as T


def bayesian_lr_with_horseshoe_prior(features: np.array, target: np.array) -> pm.model.Model:
    
    X_t = T.shared(features)
    n_features = features.shape[1]
    
    with pm.Model() as horseshoe:
        
        # sigma prior
        σ = pm.HalfNormal('sigma', 2.5)
    
        # lambda prior
        λ = pm.HalfCauchy('lambda', beta=1, shape=n_features)
        
        # tau prior
        τ = pm.HalfCauchy('tau', beta=1)
        
        # Beta prior
        z = pm.Normal('z', mu=0., sigma=1., shape=n_features)
        β = pm.Deterministic('beta', z * (τ ** 2) * (λ ** 2))
        
        # Intercept prior
        βₒ = pm.Normal('beta_0', mu=0, sigma=10)
        
        μ = X_t.dot(β) + βₒ 
        
        # observations
        y_obs = pm.Normal('obs', mu=μ, sigma=σ, observed=target)
        
        return horseshoe