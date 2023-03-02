import numpy as np

def crossover(parent_1, parent_2, vertical = True):
  M, N = parent_1.shape
  child_1 = np.copy(parent_1)
  child_2 = np.copy(parent_2)

  if vertical:
    cut = np.random.randint(1, N - 1)
    child_1[:, cut:] = parent_2[:, cut:]
    child_2[:, cut:] = parent_1[:, cut:]
  else:
    cut = np.random.randint(1, M - 1)
    child_1[cut:, :] = parent_2[cut:, :]
    child_2[cut:, :] = parent_1[cut:, :]

  return child_1, child_2

def inversion(parent, vertical = True):
  M, N = parent.shape
  K = N if vertical else M
  child = np.copy(parent)

  first_cut = np.random.randint(0, K - 1)
  second_cut = np.random.randint(first_cut + 1, K)
  
  if vertical:
    child[:, first_cut:second_cut + 1] = np.flip(parent[:, first_cut:second_cut + 1], axis = 1)
  else:
    child[first_cut:second_cut + 1, :] = np.flip(parent[first_cut:second_cut + 1, :], axis = 0)

  return child

def mutation(parent, probability):
  M, N = parent.shape
  unchanged = np.random.rand(M, N) > probability

  child = np.where(unchanged, parent, np.logical_not(parent))

  return child

