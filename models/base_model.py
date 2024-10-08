
class BaseModel(object):
    """Base class for mislabel detection models.
    
    X: np.ndarray (n_samples, d) where d is input feature dimension
    y: np.ndarray (n_samples,) where y is the label of each sample
    """
    def __init__(self, X, y, **kwargs):
        super().__init__()
        self.X = X
        self.y = y

    
    def detect_label_errors(self):
        """Find label errors in the dataset.
        
        Returns:
        
        y_pred: np.ndarray (n_samples,) where y_pred is the predicted (denoised) label of each sample
        mistakenness: np.ndarray (n_samples,) where mistakenness is a score of each sample being mislabeled
        threshold (None): float, threshold used to determine label errors
        """
        raise NotImplementedError()

    