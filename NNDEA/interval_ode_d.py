

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import torch.optim as optim
import torch.nn as nn
from nn_helper import plus




def F( X ):
   x1, x2, x3, x4, u, b, d1, d2, d3, d4 = X.squeeze()
   rhs = np.array([
        [-(-d1 + 3.18* plus(u + 3.18*x1 + 2.72*x2 + 1.42*x3 + 3.81*x4 - b) ) ],
        [-(-d2 + 2.72* plus(u + 3.18*x1 + 2.72*x2 + 1.42*x3 + 3.81*x4 - b) ) ],
        [-(-d3 + 1.42 * plus(u + 3.18*x1 + 2.72*x2 + 1.42*x3 + 3.81*x4 - b)) ],
        [-(-d4 + 3.81 * plus(u + 3.18*x1 + 2.72*x2 + 1.42*x3 + 3.81*x4 - b)) ],
        [plus(u + 3.18*x1 + 2.72*x2 + 1.42*x3 + 3.81*x4 - b) - u]
      ])
   return rhs

def F_array_process(t, array):
    """
      batch size first

    """
    if array.ndim == 1:
       return F(array).reshape(-1)
    # creating target
    result = t[0] * F(array[0,:])
    for i in range(1, array.shape[0]):
        result = np.hstack( (result, t[i] * F(array[i,:]) ))
    return result.T

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
        self.output = nn.Linear(64, 5)
        self.activation = nn.ReLU6()

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
        ydata = []
        # this will create a batch of data points for training
        while True:
            for _ in range(self.batch_size):
                # generate random initial conditions
                # select a random number between 0 and 10
                idx = np.random.randint(0, len(self.t_vals))
                b = 7.81 #np.random.uniform(0, 10)
                x1 = np.random.uniform(0, 10)
                x2 = np.random.uniform(0, 10)
                x3 = np.random.uniform(0, 10)
                x4 = np.random.uniform(0, 10)
                u = np.random.uniform(0, 10)
                d1 = np.random.uniform(0, 100)
                d2 = np.random.uniform(0, 100)
                d3 = np.random.uniform(0, 100)
                d4 = np.random.uniform(0, 100)
                y_sample = np.array([x1, x2, x3, x4, u, b, d1, d2, d3, d4])
                #
                Xdata.append([self.t_vals[idx], d1, d2, d3, d4])
                ydata.append(y_sample)


            yield np.array(Xdata), np.array(ydata)

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

    def train_nn(self, t_values, batch_size = 128, num_epochs=2000, save_period=200):    
        # Train the model
        self.model.train()
        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_func = nn.MSELoss()

        datagen = data_loader( self.F, t_values, batch_size = batch_size)
        avg_loss = 0
        report_period = 200 
        for epoch in range(num_epochs):
            data_x, data_y = datagen.get_data().__next__()
            
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
                ydata = y_pred.cpu().numpy()  # Move to CPU if necessary
                ydata = np.hstack((ydata, data_y[:, 5:] ))
                gt_y = F_array_process(data_x[:, 0].cpu().numpy(), ydata)
                gt_y = torch.tensor(gt_y, dtype=torch.float32).to(self.device)

            # Loss computation
            loss = loss_func(y_pred, gt_y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            with torch.no_grad():
                avg_loss += loss.item()

            if epoch % report_period == 0:
                avg_loss /= report_period
                print(f'Epoch {epoch}, Loss: { avg_loss }')
                avg_loss = 0
        
            if epoch % save_period == 0 and epoch > 0:
                with torch.no_grad():
                    self.save_model()

        
    def predicts(self, t_eval, d1, d2, d3, d4 ):
        t_eval = torch.tensor(t_eval).reshape(-1,1).float()
        d1 = torch.tensor(d1).reshape(-1,1).float()
        d2 = torch.tensor(d2).reshape(-1,1).float()
        d3 = torch.tensor(d3).reshape(-1,1).float()
        d4 = torch.tensor(d4).reshape(-1,1).float()

        inx = torch.cat((t_eval, d1), -1)
        inx = torch.cat((inx, d2), -1)
        inx = torch.cat((inx, d3), -1)
        inx = torch.cat((inx, d4), -1)
        y_pred = self.model( inx ).detach().numpy()
        return y_pred



if __name__ =="__main__":

    # code for RHS interval b
    # Initial conditions for y
    y0 = np.array([0, 0, 0, 0, 0, 0, 0, 0]) # x1, x2, x3, x4, d1, d2, d3, d4

    # Time points at which to solve the system
    t0 = 0
    t1 = 10
    t_eval = np.arange(t0, t1, 0.5)

        # ------------------
    # solving the same problem using neural network
    # ------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnlpsolver = ModelTrainerInterval("weights", y0, F_array_process, device=device )

    # use the existing train data

    nnlpsolver.train_nn(t_eval, batch_size=128, num_epochs=2000)



    # predicting and plotting the results
    nnlpsolver.load_model()
    D_ = [29.99, 25.66, 13.39, 35.94]
    model_preds = nnlpsolver.predicts( [7.81]*len(t_eval), [D_[0]]*len(t_eval), [D_[1]]*len(t_eval), [D_[2]]*len(t_eval), [D_[3]]*len(t_eval) )


    # plot the results
    plt.figure()

    plt.plot(t_eval, model_preds[:,0], label='x1 nn')
    plt.plot(t_eval, model_preds[:,1], label='x2 nn')
    plt.plot(t_eval, model_preds[:,2], label='x3 nn')
    plt.plot(t_eval, model_preds[:,3], label='x4 nn')
    plt.plot(t_eval, model_preds[:,4], label='u nn')

    plt.legend()
    plt.show()
    plt.savefig("results/interval_ode.png")







    print("Done!")