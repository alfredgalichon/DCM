# Illustration of convergence to max-stable
#
# import libraries and generate date
import numpy as np
import matplotlib.pyplot as plt
#
np.random.seed(7)
O,I = 1000,100000
#
lambda_exp = 0.8
alpha_pareto = 0.6
#
M1_o = np.zeros(O) # max of exponentials
M2_o = np.zeros(O) # max of Paretos
M3_o = np.zeros(O) # max of uniforms
for o in range(O):
  M1_o[o] = (np.random.exponential( scale = lambda_exp, size = I).max() / lambda_exp - np.log(I)) 
  M2_o[o] = np.random.pareto(a = alpha_pareto, size = I).max() / I**(1/alpha_pareto)
  M3_o[o] = I * np.random.uniform(size = I).max() - I
# assess convergence by rescaling to uniform using cdfs
U1tilde_o = np.exp(-np.exp( -  M1_o  )  )
U2tilde_o = np.exp(- M2_o**(-alpha_pareto))
U3tilde_o = np.exp( -(- M3_o) )
#
# plot empirical cdfs
plt.plot(np.sort(U1tilde_o), np.arange(1, O + 1) / O , marker=".", linestyle="none")
plt.gca().set(xlabel="t", ylabel="F(t)", title="Empirical CDF (Exponential => Type I)")
plt.grid(True)
plt.show()
#
plt.plot(np.sort(U2tilde_o), np.arange(1, O + 1) / O , marker=".", linestyle="none")
plt.gca().set(xlabel="t", ylabel="F(t)", title="Empirical CDF (Pareto => Type II)")
plt.grid(True)
plt.show()
#
plt.plot(np.sort(U3tilde_o), np.arange(1, O + 1) / O , marker=".", linestyle="none")
plt.gca().set(xlabel="t", ylabel="F(t)", title="Empirical CDF (Uniform => Type III)")
plt.grid(True)
plt.show()