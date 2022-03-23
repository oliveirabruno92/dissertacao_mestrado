import numpy as np 
import matplotlib.pyplot as plt 
import argparse 

class RegressionData:
    
    def __init__(self, args):
        self.n_samples = args.get('n_samples') 
        self.n_features = args.get('n_features')
        self.nonzero_features = args.get('nonzero_features')
        self.seed = args.get('seed') 
        self.rng = np.random.default_rng(self.seed)
        
    def _generate_coeficients(self, mean: float = 5., std: float = 1.) -> np.array:
        coefs = np.zeros(self.n_features)
        coefs[:self.nonzero_features] = (
            self.rng.choice(a=[-1,1], size=self.nonzero_features) * 
            self.rng.normal(mean, std, size=self.nonzero_features)
        )
        return coefs

    def _noise(self):
        return self.rng.normal(0, 1, size=self.n_samples)
    
    def _intercept(self):
        return self.rng.uniform(-3,3)
    
    def _generate_features(self):
        return self.rng.normal(size=(self.n_samples, self.n_features))
    
    def _generate_target(self, features, coefs):
        coefs = self._generate_coeficients()
        noise = self._noise()
        intercept = self._intercept()
        return intercept + features.dot(coefs) + noise  
    
    def generate(self):
        coefs = self._generate_coeficients()
        X = self._generate_features()
        y = self._generate_target(features=X, coefs=coefs)
        return X, y, coefs
    
    def plot(self, X, y, coefs):
        
        fig, axes = plt.subplots(
            nrows=2, 
            ncols=self.nonzero_features,
            sharex=True, 
            sharey=True,
            figsize=(16, 6),
            tight_layout=True
        )
        
        zero_coef_ix = self.rng.choice(
            range(self.nonzero_features, self.n_features), 
            replace=False, 
            size=self.nonzero_features
        )
        zero_coef_ix.sort()

        for i, (ax, coef) in enumerate(zip(axes[0], coefs)):
            ax.scatter(X[:, i], y, alpha=0.75)
            ax.set_xlabel(f"$x_{{ {i} }}$")
            ax.set_title(f"$\\beta_{{ {i} }} \\approx {coef:.2f}$")

        for ax, i in zip(axes[1], zero_coef_ix):
            ax.scatter(X[:, i], y, alpha=0.75)
            
            ax.set_xlabel(f"$x_{{ {i} }}$")
            ax.set_title(f"$\\beta_{{ {i} }} = 0$")
            
        axes[0, 0].set_ylabel("$y$")
        axes[1, 0].set_ylabel("$y$")
        plt.show();
        
        return fig 
    
def main(args):
    
    data = RegressionData(args)
    X, y, coefs = data.generate()
    if args['plot']:
        _ = data.plot(X, y, coefs)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data params.')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--n_features', type=int, default=50)
    parser.add_argument('--nonzero_features', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot', type=bool, default=True)
    
    args = parser.parse_args().__dict__
    main(args)

    