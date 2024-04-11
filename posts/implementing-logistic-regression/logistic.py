import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        # your computation here: compute the vector of scores s
        # s = w1 x (loan interest rate) + w2 x (load percent income)
        # return X@self.w <- also works
    
        return X@self.w.T

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)

        """
        
        S = self.score(X)
        return torch.where(S > 0, 1.0, 0.0)
    
class LogisticRegression(LinearModel):
    def sigmoid(x):
        return 1.0 / (1.0 + torch.exp(-x))
        
    def loss(self, X, y):
        score = LinearModel.score(X)
        n = X.size()[0]
        loss = (1/n) * torch.einsum(self.littleLoss(y, score))
        return loss
    
    def litteLoss(self, y, s):
        loss = -y * torch.log(self.sigmoid(s)) - (1 - y) * torch.log(1 - self.sigmoid(s));
        return loss

    def grad(self, X, y):
        n = X.size()[0]
        grad = (1/n) * torch.einsum((self.sigmoid(self.score(X)) - y) * X)
        return grad   



