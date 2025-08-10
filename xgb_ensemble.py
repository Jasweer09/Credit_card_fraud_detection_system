import numpy as np
class XGBEnsemble:
    def __init__(self, model_borderline, model_native, threshold=0.3):
        self.model_borderline = model_borderline
        self.model_native = model_native
        self.threshold = threshold

    def predict_proba(self, X):
        proba_borderline = self.model_borderline.predict_proba(X)[:, 1]
        proba_native = self.model_native.predict_proba(X)[:, 1]
        proba_avg = (proba_borderline + proba_native) / 2
        return np.vstack([1 - proba_avg, proba_avg]).T

    def predict(self, X):
        
        return (self.predict_proba(X)[:, 1] >= 0.55).astype(int)