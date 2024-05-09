import torch

class LinearModel:

    def __init__(self):
        self.w = None 
        self.w_prev = None

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
    
        return torch.matmul(X, self.w)

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

    def sigmoid(self, s):
        """
        Computes the sigmoid function on s

        ARGUMENTS:
            s, torch.Tensor: a feature matrix of size (n, p) where n is the number of data points
            and p is the number of features.
            
        """
        return 1.0 / (1.0 + torch.exp(-s))
        
    def loss(self, X, y):
        """
        Compute the misclassification rate. 

        ARGUMENTS:
            X, torch.Tensor: a feature matrix of size (n, p) where n is the number of data points
            and p is the number of features.

            y, torch.Tensor: the target vector. size (n, ), values are either {0, 1}
        
        RETURNS:
            L(w), torch.Tensor. size (1), The results of the logistic loss function, from Professor Chodrows notes: https://middlebury-csci-0451.github.io/CSCI-0451-s24/assignments/blog-posts/blog-post-optimization.html#implement-linearmodel-and-logisticregression
        """
        s = self.score(X)
        sig_s = self.sigmoid(s)
        loss = torch.negative(y) * torch.log(sig_s) - ((1 - y) * torch.log(1 - sig_s))
        return torch.mean(loss)
    

    def grad(self, X, y):
        """
        Compute the gradient of the empirical risk L(w). 

        ARGUMENTS:
            X, torch.Tensor: a feature matrix of size (n, p) where n is the number of data points
            and p is the number of features.

            y, torch.Tensor: the target vector. size (n, ), values are either {0, 1}
        
        RETURNS:
            grad(L(w)): the Empirical Risk, based on equation 9.5 in Professor Chodrow's notes: https://www.philchodrow.prof/ml-notes/chapters/23-gradient-descent.html#gradient-of-the-empirical-risk
        
        """
        s = self.score(X)
        grad =  X * (torch.sigmoid(s) - y).unsqueeze(1) 
        grad_T = torch.transpose(grad, 0, 1)
        return torch.mean(grad_T, 1) 
    
    def hessian(self, X, y):
        """
        Computes the Hessian matrix 

        ARGUMENTS:
            X, torch.Tensor: a feature matrix of size (n, p) where n is the number of data points
            and p is the number of features.

            y, torch.Tensor: the target vector. size (n, ), values are either {0, 1}
        
        RETURNS:
            Hessian, torch.tensor: The matrix of second derivatives of the Loss function.

        """
        s = self.score(X)
        D = torch.sigmoid(s) * (1 - torch.sigmoid(s))
        D_diag = torch.diag(D) 
        return X.T@D_diag@X


    
class NewtonOptimizer():
    def __init__(self, model):
        self.model = model 


    def step(self, X, y, alpha):
        # Compute the usual gradient
        grad = self.model.grad(X, y)
        hess = self.model.hessian(X, y)
        self.model.w = self.model.w - (alpha * (torch.inverse(hess)@grad))



class GradientDescentOptimizer(LogisticRegression):
    def __init__(self, model):
        self.model = model 
        

    def step(self, X, y, alpha, beta):
        """
        Compute one step of the logistic regression update using the feature matrix X 
        and target vector y in terms of alpha, the learning rate and beta, the momentum. 

        ARGUMENTS:
            X, torch.Tensor: a feature matrix of size (n, p) where n is the number of data points
            and p is the number of features. 

            y, torch.Tensor: the target vector. size (n, ), values are either {0, 1}

            alpha, float: the learning rate

            beta, float: the momentum

        RETURNS:
            L(w), torch.Tensor: size (1), the empirical loss of the function at that step
        
        """
        if self.model.w_prev == None:
            self.model.w_prev = torch.rand((X.size()[1]))

        grad = self.model.grad(X, y)
        w_step = self.model.w - (alpha * grad) + (beta * (self.model.w - self.model.w_prev))
        self.model.w_prev = self.model.w
        self.model.w = w_step

        return self.model.loss(X, y)


