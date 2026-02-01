from sklearn.base import TransformerMixin

class BaseObservable(TransformerMixin):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        """Fit the observable to the data X."""
        raise NotImplementedError("fit method not implemented.")
    
    def transform(self, X):
        """Transform the data X into the observable space."""
        raise NotImplementedError("transform method not implemented.")

    def get_output_names(self) -> int:
        """Return the number of output features."""
        raise NotImplementedError("get_output_names method not implemented.")

