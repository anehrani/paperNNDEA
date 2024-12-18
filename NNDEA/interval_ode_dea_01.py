

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import torch.optim as optim
import torch.nn as nn
from nn_helper import plus


###################################################################
#   Initial settings
###################################################################
CoeffX = [
            np.array([1.50, 4.00, 3.20, 5.20, 3.50, 3.20])/5.2,
            np.array([0.20, 0.70, 1.20, 2.00, 1.20, 0.70])/2.0
          ]
CoeffY = [
            np.array([1.40, 1.40, 4.20, 2.80, 1.90, 1.40])/4.20,
            np.array([0.35, 2.10, 1.05, 4.20, 2.50, 1.50])/4.20
        ]


D = np.array( np.array([1., 0., 0., 0., 0., 0., 0.]).T )
A = np.array([[ 0., CoeffX[0][0], CoeffX[0][1], CoeffX[0][2], CoeffX[0][3], CoeffX[0][4], CoeffX[0][5]],
              [ 0., CoeffX[1][0], CoeffX[1][1], CoeffX[1][2], CoeffX[1][3], CoeffX[1][4], CoeffX[1][5]],
              [ 0.,             -CoeffY[0][0], -CoeffY[0][1], -CoeffY[0][2], -CoeffY[0][3], -CoeffY[0][4], -CoeffY[0][5]],
              [ 0.,             -CoeffY[1][0], -CoeffY[1][1], -CoeffY[1][2], -CoeffY[1][3], -CoeffY[1][4], -CoeffY[1][5]],
              [-1.0, 0., 0., 0., 0., 0., 0.],
              [0., -1., 0., 0., 0., 0., 0.],
              [0., 0., -1., 0., 0., 0., 0.],
              [0., 0., 0., -1., 0., 0., 0.],
              [0., 0., 0., 0., -1., 0., 0.],
              [0., 0., 0., 0., 0., -1., 0.],
              [0., 0., 0., 0., 0., 0., -1.],
              ])
B = np.array( np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).T )
U = np.array( np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]).T)
#######################################################################

def F( X : np.array ):
    """
        X shape: 1 x 15
        0 : theta
        1-6: Lambda
        7-10: U
        11-14 : A of under study
    """
    #B[0] = X[0] * CoeffX[0][0]
    #B[1] = X[0] * CoeffX[0][1]
    Lambda = X[:7] # primal
    U[:4] = X[7:11] # dual
    A[0, 0] = -X[11]
    A[1, 0] = -X[12]
    B[2] = -X[13]
    B[3] = -X[14]
    # update coeff matrix


    rhs = list(-(D + A.T @ np.clip( U + (A @ Lambda) - B ,a_min=0, a_max=None) ))
    rhs += list(np.clip( U +  A @ Lambda - B ,a_min=0, a_max=None) - U)
                  
    
    return np.array(rhs[:11])


def F_array_process(t, array):
    """
      batch size first

    """
    if array.ndim == 1:
       return F(array).reshape(-1)
    # creating target
    result = t[0] * F(array[0,:])
    for i in range(1, array.shape[0]):
        result = np.vstack( (result, t[i] * F(array[i,:]) ))
    return result

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.hidden_0 = nn.Linear(5, 128)
        self.hidden_1 = nn.Linear(128, 256)
        self.hidden_2 = nn.Linear(256, 512)
        self.hidden_6 = nn.Linear(512, 512)
        self.hidden_7 = nn.Linear(512, 256)
        self.hidden_8 = nn.Linear(256, 124)
        self.hidden_9 = nn.Linear(124, 64)
        self.output = nn.Linear(64, 11)
        self.activation = nn.ReLU6()
        self.relu = nn.ReLU()

    def forward(self, inx):
        x = self.activation(self.hidden_0(inx))
        x = self.activation(self.hidden_1(x))
        x = self.activation(self.hidden_2(x))
        #x = self.activation(self.hidden_3(x))
        #x = self.activation(self.hidden_4(x))
        #x = self.activation(self.hidden_5(x))
        x = self.activation(self.hidden_6(x))
        x = self.activation(self.hidden_7(x))
        x = self.activation(self.hidden_8(x))
        x = self.activation(self.hidden_9(x))
        x = self.output(x)
        for i in range(x.shape[1]):
          x[:,i] = ( 1 - torch.exp(-inx[:,0]) ) * x[:,i]

        return x
    
class data_loader:
    def __init__(self, RHSFunction: callable, t_values, batch_size:int=1) -> None:
        self.rhsF = RHSFunction
        self.batch_size = batch_size
        self.t_vals = t_values
    def get_data(self):

        Xdata = []
        #ydata = []
        # this will create a batch of data points for training
        while True:
            for _ in range(self.batch_size):
                # generate random initial conditions
                # select a random number between 0 and 10
                idx = np.random.randint(0, len(self.t_vals))
                b = 7.81 #np.random.uniform(0, 10)
                X = list(np.random.uniform(0, 1, 4))
                #y_sample = X[:11]
                #
                Xdata.append([self.t_vals[idx]] + X )
                #ydata.append(y_sample)


            yield np.array(Xdata) #, np.array(ydata)

class ModelTrainerInterval:
    def __init__(self, model_path: str, y_initial, RHSFunction: callable, device = 'cpu'):
        self.device = device
        self.model_path = model_path
        self.model = NeuralNet()
        self.F = RHSFunction
        self.y_initial = y_initial

    def get_model(self):
        return self.model
    
    def load_model(self):
        self.model.load_state_dict(torch.load(f"{self.model_path}/odeNN_model.pth"))
        self.model.eval()

    def save_model(self):
        torch.save(self.model.state_dict(),  f"{self.model_path}/odeNN_model.pth")

    def train_nn(self, t_values, batch_size = 256, num_epochs=5000, save_period=200):    
        # Train the model
        self.model.train()
        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        loss_func = nn.MSELoss()

        datagen = data_loader( self.F, t_values, batch_size = batch_size)
        avg_loss = 0
        report_period = 200 
        for epoch in range(num_epochs):
            data_x = datagen.get_data().__next__()
            
            # Convert data to tensors
            data_x = torch.tensor(data_x, dtype=torch.float32)

            # Move tensors to the appropriate device (e.g., GPU if available)
            data_x = data_x.to(self.device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(data_x)

            # Processing ydata and gt_y
            # Ensure no gradient tracking here
            with torch.no_grad():
                data_tmp = data_x.cpu().numpy()
                ydata = y_pred.cpu().numpy()  # Move to CPU if necessary
                ydata = np.hstack((ydata, data_tmp[:, 1:] ))
                gt_y = F_array_process(data_tmp[:, 0], ydata)
                gt_y = torch.tensor(gt_y, dtype=torch.float32).to(self.device)

            # Loss computation
            loss = loss_func(y_pred, gt_y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            scheduler.step()

            with torch.no_grad():
                avg_loss += loss.item()

            if epoch % report_period == 0:
                avg_loss /= report_period
                print(f'Epoch {epoch}, Loss: { avg_loss }')
                avg_loss = 0
        
            if epoch % save_period == 0 and epoch > 0:
                with torch.no_grad():
                    self.save_model()

        
    def predicts(self, input ):
        inx = torch.tensor(input, dtype=torch.float32)
        y_pred = self.model( inx ).detach().numpy()
        return y_pred



if __name__ =="__main__":

    # code for RHS interval b
    # Initial conditions for y
    y0 = np.array([0]*16) # x1, x2, x3, x4, d1, d2, d3, d4

    # Time points at which to solve the system
    t0 = 0
    t1 = 10
    t_eval = np.arange(t0, t1, 0.005)

        # ------------------
    # solving the same problem using neural network
    # ------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnlpsolver = ModelTrainerInterval("weights", y0, F_array_process, device=device )

    # use the existing train data

    nnlpsolver.train_nn(t_eval, batch_size=256, num_epochs=5100)



    # predicting and plotting the results
    nnlpsolver.load_model()



    # create prediction for specific input

    # inpx = [t_eval[-1], CoeffX[0][-1], CoeffX[1][-1], CoeffY[0][-1], CoeffY[1][-1] ]
    inpx = []
    for t in t_eval:
        inpx.append([t, CoeffX[0][-1], CoeffX[1][-1], CoeffY[0][-1], CoeffY[1][-1] ])

    model_preds = nnlpsolver.predicts( np.array(inpx) )

    print("model predictions: ", model_preds)
    print("GT predictions for theta: ", 1 )

    # plot the results
    plt.figure()

    plt.plot(t_eval, model_preds[:,0], "--", label=' theta')
    plt.plot(t_eval, model_preds[:,7], label='u 1')
    plt.plot(t_eval, model_preds[:,8], label='u 2')
    plt.plot(t_eval, model_preds[:,9], label='u 3')
    plt.plot(t_eval, model_preds[:,10], label='u 4')
    plt.plot(t_eval, model_preds[:,1], label='lambda 1')
    plt.plot(t_eval, model_preds[:,2], label='lambda 2')
    plt.plot(t_eval, model_preds[:,3], label='lambda 3')
    plt.plot(t_eval, model_preds[:,4], label='lambda 4')
    plt.plot(t_eval, model_preds[:,5], label='lambda 5')
    plt.plot(t_eval, model_preds[:,6], label='lambda 6')

    plt.legend()
    plt.show()
    plt.savefig("results/interval_ode.png")







    print("Done!")