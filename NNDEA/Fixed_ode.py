


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import torch.optim as optim
from nn_helper import NeuralNet, plus




# Define the system of ODEs
def F(X):
  x1, x2, x3, x4, u = X.squeeze()
  return np.array([
      [-(-9.54 +  3.18 * plus(u + 3.18 * x1 + 2.72 * x2 + 1.42 * x3 + 3.81 * x4 - 7.81))],
      [-(-8.16 + 2.72 * plus(u + 3.18 * x1 + 2.72 * x2 + 1.42 * x3 + 3.81 * x4 - 7.81))],
      [-(-4.26 + 1.42 * plus(u + 3.18 * x1 + 2.72 * x2 + 1.42 * x3 + 3.81 * x4 - 7.81))],
      [-(-11.43 + 3.81 * plus(u + 3.18 * x1 + 2.72 * x2 + 1.42 * x3 + 3.81 * x4 - 7.81))],
      [plus(u + 3.18 * x1 + 2.72 * x2 + 1.42 * x3 + 3.81 * x4 - 7.81) - u]
  ]).T

def F_array_process(t, array):
    """
      batch size first

    """
    if array.ndim == 1:
       return F(array).reshape(-1)

    result = F(array[0,:])
    for i in range(1, array.shape[0]):
        result = np.vstack( (result, F(array[i,:]) ))
    return result


class ModelTrainerFixed:
    def __init__(self, model_path: str, y_initial, RHSFunction: callable):
        self.model_path = model_path
        self.model = NeuralNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.F = RHSFunction
        self.y_initial = y_initial

    # Define the loss function
    def loss_function(self, t ):
        #
        y_pred = self.model(t)
        diff_eq = self.model(t) - t*self.F(t, y_pred.detach().numpy())
        loss = torch.mean(diff_eq**2)
        
        return loss

    def get_model(self):
        return self.model
    
    def load_model(self):
        self.model.load_state_dict(torch.load(f"{self.model_path}/model.pth"))
        self.model.eval()

    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.model_path}/model.pth")

    def train_nn(self, t_values, num_epochs=2000, save_period=200):    
        # Train the model
        batch_size = 128

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            idx = np.random.randint(0, len(t_values), batch_size)
            loss = self.loss_function( torch.tensor(t_values[idx]).reshape(-1,1) )
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        
            if epoch % save_period == 0 and epoch > 0:
                with torch.no_grad():
                    self.save_model()

        
    def predicts(self, t_values):
        y_pred = self.model( t_values ).detach().numpy()
        return y_pred


if __name__ == "__main__":

    # Initial conditions for y
    y0 = np.array([0, 0, 0, 0, 0])

    # Time points at which to solve the system
    t0 = 0
    t1 = 100
    t_eval = np.arange(t0, t1, 0.01)

    

  
    # ------------------
    # solving the same problem using neural network
    # ------------------
    nnlpsolver = ModelTrainerFixed("weights", y0, F_array_process)

    # use the existing train data

    nnlpsolver.train_nn(t_eval, num_epochs=2000)

    
    # Evaluate the model
    nnlpsolver.load_model()
    t_test = np.arange(t0, t1, 0.1)
    #

    y_pred = nnlpsolver.predicts( torch.tensor(t_test).reshape(-1,1) )
    
    # Plot the results

    sol = solve_ivp(F_array_process, [t0, t1], y0, t_eval=t_eval)

    plt.figure(figsize = (12, 4))
    plt.subplot(121)
    plt.plot(sol.t, sol.y[0], label='x1 gt')
    plt.plot(t_test, y_pred[:,0],"--",  label='x1 nn')
    plt.plot(sol.t, sol.y[1], label='x2 gt')
    plt.plot(t_test, y_pred[:,1],"--",  label='x2 nn')
    plt.plot(sol.t, sol.y[2], label='x3 gt')
    plt.plot(t_test, y_pred[:,2],"--",  label='x3 nn')
    plt.plot(sol.t, sol.y[3], label='x4 gt')
    plt.plot(t_test, y_pred[:,3],"--",  label='x4 nn')
    plt.plot(sol.t, sol.y[4], label='u gt')
    plt.plot(t_test, y_pred[:,4],"--",  label='u nn')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plt.plot(t_test, y_pred, label='Predicted')
    # #plt.plot(t_test, torch.exp(-t_test).numpy(), label='True Solution')  # True solution for dy/dt = -y
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.show()

    print(" Finished! ")

