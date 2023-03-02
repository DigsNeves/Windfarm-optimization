import numpy as np
import matplotlib.pyplot as plt

class FarmPlotter(object):
  def __init__(self, farm):
    self.farm = farm

  def plot(self):
    x = self.farm.turbine_position[:, 0]/self.farm.horizontal_separation
    y = self.farm.turbine_position[:, 1]/self.farm.vertical_separation
    
    plt.figure(figsize = (0.7 * self.farm.number_of_columns, 0.7 * self.farm.number_of_rows))
    plt.scatter(x, y, marker = "s", s = 300)
    plt.xticks(range(0, (self.farm.number_of_columns + 1)), [])
    plt.yticks(range(0, (self.farm.number_of_rows + 1)), [])
    plt.ylim([0, self.farm.number_of_rows])
    plt.xlim([0, self.farm.number_of_columns])
    plt.grid(True)
    plt.tick_params(axis = 'both', color = (0,0,0,0))
    plt.gca().invert_yaxis()

class EnergyComputer(object):
  def __init__(self, farm):
    self.farm = farm
  
  def compute(self, wind_vector):
    u_0 = np.linalg.norm(wind_vector)
    v = wind_vector/u_0
    c_1 = np.sqrt(self.farm.alpha**2 + 1)
    c_2 = np.sqrt((1 - self.farm.a) / (1 - 2 * self.farm.a))
    R = (self.farm.rotor_radius * c_2)

    L = np.zeros(self.farm.number_of_turbines, dtype = float)

    for index in range(self.farm.number_of_turbines):
        P_0 = self.farm.turbine_position[index] - self.farm.d * v
        P = self.farm.turbine_position - P_0
        abs_P = np.linalg.norm(P, axis = 1)
        prod_v_P = np.dot(P, v)
        
        x = prod_v_P - self.farm.d

        r = (1 - (2 * self.farm.a) / (1 + self.farm.alpha * x / R)**2)

        influenced = np.full(self.farm.number_of_turbines, True)
        influenced &= c_1 * prod_v_P >= abs_P 
        influenced &= x > 0
        influenced[index] = False
          

        L += np.where(influenced, (1 - r)**2, 0)

    U = np.where(L != 0.0, u_0 * (1 - np.sqrt(L)), u_0)
    
    return 0.3 * np.sum(U ** 3)

class Windfarm(object):
  def __init__(self, placement, horizontal_separation, vertical_separation, rotor_radius, alpha, C_t):
    self.horizontal_separation = horizontal_separation
    self.vertical_separation = vertical_separation
    self.rotor_radius = rotor_radius
    self.alpha = alpha
    self.C_t = C_t
    self.d = rotor_radius/alpha
    self.a = 0.5 - 0.5 * np.sqrt(1 - C_t)

    self.set_placement(placement)
    
  def set_placement(self, placement):
    self.placement = placement
    self.number_of_rows, self.number_of_columns = placement.shape
    self.number_of_turbines = np.sum(placement)
    self._initialize_turbine_positions()
    self.energy_computer = EnergyComputer(self)

  def _initialize_turbine_positions(self):
    self.turbine_position = []
    
    for row in range(self.number_of_rows):
        for column in range(self.number_of_columns):
            if self.placement[row, column]:
                
                x = self.horizontal_separation * (1/2 + column)
                y = self.vertical_separation * (1/2 + row)
                self.turbine_position.append([x, y]) 
    
    self.turbine_position = np.array(self.turbine_position)

  def show(self):
    FarmPlotter(self).plot()

  def produced_energy(self, u):
    return self.energy_computer.compute(u)

