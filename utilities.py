import numpy as np #Numerical Python for fast and vector computations
import matplotlib.pyplot as plt #Making graphics and plots
from sklearn.metrics import accuracy_score #Acuuracy score metrics from sklearn

class Perceptron:
    #initialise the weights and bias
    def __init__(self,X,y,alpha=0.1,n_iter=1000):
        self.X = X
        self.y = y
        #Initialise the weights
        self.W = None
        #Initialise the bias of our model
        self.b = None
        #The parameter history of our percptron model
        self.History = {'Weight_History':[],
                        'Bias_History': [],
                        'Loss_History':[],
                        'Index':[]
                       }
        self.head_features = lambda i: print(f"{self.X[:i]}\n")
        self.head_target = lambda i: print(f"{self.y[:i]}\n")
        self.AnimateParameters = [] #This variable stores the parameters to be used for the animation
        self.alpha = alpha
        self.n_iter = n_iter
        
    #Print the details of the entered dataset
    def details(self):
        b = self.b
        W = self.W
        print("\tDATASET DETAILS\n")
        print(f"Feature size: {self.X.shape}\nTarget size: {self.y.shape}\nInitial Weights: {W}\nInitial bias: {b}\n")
        
    #Create the model   
    def Model(self,X,W,b):
        Z = (X).dot(W) + b
        A = 1/(1+np.exp(-Z))
        return A
    
    #Evaluate the cost of the model
    def Cost_function(self, A):
        epsilon = 1e-8  # Small epsilon value to avoid taking logarithm of zero or one
        cost = (-1 / ((self.X).shape[0])) * (np.sum(self.y * np.log(A + epsilon) + (1 - self.y) * np.log(1 - A + epsilon)))
        return cost
    
    #Calculate the derviatives of the cost function with resoect to the Weights and bias
    def Gradients(self,A):
        dW = (-1/(self.X.shape[0]))*(self.X.T).dot(self.y-A)
        db = (-1/(self.X.shape[0]))*np.sum(self.y-A)
        return (dW,db)
    
    #Adjust the parameters weights and bias
    def Update(self,dW,db):
        self.W = self.W - self.alpha*dW
        self.b = self.b - self.alpha*db
        return (self.W,self.b) 
    
    #Fit or train the Model
    def fit(self):
        #Initialise the weights
        self.W = np.random.randn(self.X.shape[-1],1)
        #Initialise the bias of our model
        self.b = np.random.randn(1)
        for i in range(self.n_iter):
            A = self.Model(self.X,self.W,self.b)
            L = self.Cost_function(A)
            dW,db = self.Gradients(A)
            self.W,self.b = self.Update(dW,db)
            self.History['Weight_History'].append(self.W)
            self.History['Bias_History'].append(self.b)
            self.History['Loss_History'].append(L) 
            self.History['Index'].append(i)
            self.AnimateParameters.append([self.W,self.b,self.History['Loss_History'][:i+1],i])
        #Return the parameters for later use like animations    
        return self.AnimateParameters
    
    #Use the sklearn metric to evaluate the model performance
    def Train_Report(self):
        return accuracy_score(self.y,self.Predict(self.X))
    
    #Print the learning curve
    def Learning_curve(self):
        if not len(self.History['Loss_History']):
            print("Train the model before attempting to plot the learning curve!")
            return None
        else:
            plt.plot(self.History['Loss_History'],color='green')
            plt.show()

    #Plot the final state plots for the decision line, sigmoid function and the cost function ( Learning curve )     
    def Final_plots(self):
        W = self.History['Weight_History']
        b = self.History['Bias_History']
        loss = self.History['Loss_History']
        i = self.History['Index']
        X = self.X
        y = self.y
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].clear() #Decision Line
        ax[1].clear() #Sigmoid actvation function
        ax[2].clear() #Cost function
        s = 300
        ax[0].scatter(X[:,0],X[:,1],c=y,s=s,cmap='summer',edgecolors='k',linewidths=3)
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()
        x1 = np.linspace(-3,6,100)
        #x2 = (-Model1.W[0]*x1 - Model1.b)/Model1.W[1]
        x2 = (-W[-1][0]*x1 - b[-1])/W[-1][1]
        ax[0].plot(x1,x2,c='orange',lw=4)
        ax[0].set_xlim(X[:,0].min(),X[:,0].max())
        ax[0].set_ylim(X[:,1].min(),X[:,1].max())
        ax[0].set_title('Decision Line')
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('x2')
        #Sigmoid activation function
        Z = X.dot(W[-1]) + b[-1]
        Z_new = np.linspace(Z.min(),Z.max(),100)
        A = 1/(1+np.exp(-Z_new))
        ax[1].plot(Z_new,A,c='orange',lw=4)
        ax[1].scatter(Z[y==0],np.zeros(Z[y==0].shape),c='#008066',edgecolors='k',linewidths=3,s=s)
        ax[1].scatter(Z[y==1],np.ones(Z[y==1].shape),c='#ffff66',edgecolors='k',linewidths=3,s=s)
        ax[1].set_xlim(Z.min(),Z.max())
        ax[1].set_title('Sigmoid activation function')
        ax[1].set_xlabel('Z')
        ax[1].set_ylabel('A(Z)')
        for j in range(len(A[y.flatten()==0])):
            ax[1].vlines(Z[y==0][j],ymin=0,ymax=1/(1+np.exp(-Z[y==0][j])),color='red',alpha=0.4,zorder=-1)
        for j in range(len(A[y.flatten()==1])):
            ax[1].vlines(Z[y==1][j],ymax=1,ymin=1/(1+np.exp(-Z[y==1][j])),color='red',alpha=0.4,zorder=-1) 
        #Cost function
        ax[2].plot(i,loss,color='red',lw=4)
        ax[2].set_xlim(-20,len(loss))
        ax[2].set_ylim(0,loss[0]*1.1)
        ax[2].set_title("Cost function")
        ax[2].set_xlabel('Iteration Number')
            
    #Predict for new entry
    def Predict(self,New):
        y_pred = self.Model(New,self.W,self.b)
        return y_pred>=0.5
