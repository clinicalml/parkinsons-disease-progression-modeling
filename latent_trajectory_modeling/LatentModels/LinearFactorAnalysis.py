from factor_analyzer import FactorAnalyzer
import numpy as np

class LinearFactorAnalysis(object):
    
    def fit(self, data):
        self.fa = FactorAnalyzer()
        self.fa.fit(data, 25)
        # Check Eigenvalues
        ev, v = self.fa.get_eigenvalues()
        n_fac = sum(ev>2) # can manually tune this 1 factor if >5; 5 factors if >2; 15 if >1
        self.fa = FactorAnalyzer(n_factors =  n_fac, rotation="varimax")
        self.fa.fit(data)
        
    def decode(self, latents):
        # must be called after fa
        assert self.fa.loadings_.shape[1] == latents.shape[1]
        return np.matmul(latents, self.fa.loadings_.T)
        
    def predict(self, data):
        # returns latents + predictioins
        latents = self.fa.transform(data)
        return latents, np.matmul(latents, self.fa.loadings_.T)