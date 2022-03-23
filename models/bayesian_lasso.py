import pymc3 as pm
import numpy as np
import theano as T


def bayesian_lasso_regression(features: np.array, target: np.array) -> pm.model.Model:
    
    X_t = T.shared(features)
    n_samples, n_features = features.shape
    
    with pm.Model() as bayesian_lasso:
        
        # sigma prior
        σ = pm.HalfNormal('sigma', 10)
        
        # lambda prior
        λ = pm.HalfCauchy('lambda', beta=1)
    
        # Beta prior
        β = pm.Laplace('beta', mu=0, b=σ / λ, shape=n_features)
        
        # Intercept prior
        βₒ = pm.Normal('beta_0', mu=0, sigma=10)
        
        μ = X_t.dot(β) + βₒ 
        
        # observations
        y_obs = pm.Normal('obs', mu=μ, sigma=σ, observed=target)
        
        return bayesian_lasso