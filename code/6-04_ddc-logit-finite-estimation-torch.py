# Estimation of DDC logit model w finite horizon in Torch
#
# Importing libraries and generating data:
import torch, numpy as np, scipy.sparse as sp
np.random.seed(7)
T,X,Y,K = 2,3,2,3
beta = 0.9
phi_x_y_k = np.random.normal(size=(X,Y,K))
muhat_txy = np.random.uniform(size=(T*X*Y))
P_xprime_xy = np.random.uniform(size= (X,X*Y))
P_xprime_xy = P_xprime_xy / P_xprime_xy.sum(axis=0)[None,:]
#####
P_xprime_xy = torch.from_numpy(P_xprime_xy).float()
phi_t_x_y_k = np.array([beta**t for t in range(T)])[:,None,None,None] *  phi_x_y_k[None,:,:,:]
phi_t_x_y_k = torch.from_numpy(phi_t_x_y_k).float()
Ehat_phi_k = torch.from_numpy(muhat_txy.T).float() @ phi_t_x_y_k.reshape((-1,K))
SigmaY = sp.kron( sp.eye(T*X),np.ones((1,Y)))
Psi = sp.kron(sp.diags(diagonals=[1]*(T-1), offsets=+1, shape=(T,T)),P_xprime_xy.T) - SigmaY.T
I = muhat_txy.sum()
V= np.diag(muhat_txy) / I - muhat_txy[:,None] @ muhat_txy[None,:] / (I*I)

def F(lambda_k):
    u_t_x = torch.zeros((T+1, X))
    for t in range(T-1, -1, -1):
        u_t_x[t,:] = torch.log(torch.exp( (phi_t_x_y_k[t,:,:,:].reshape((-1, K)) @ lambda_k + u_t_x[t+1,:] @ P_xprime_xy).reshape((X, Y))).sum(axis=1))
    return Ehat_phi_k @ lambda_k + torch.from_numpy(muhat_txy.T @ Psi).float() @ u_t_x[:-1,:].flatten()

lambda_k = torch.ones(K, requires_grad=True)
optimizer = torch.optim.Adam([lambda_k],lr=0.01, maximize=True)
num_steps = 1000
for step in range(num_steps):
    optimizer.zero_grad()  # Clear previous gradients
    res = F(lambda_k)      # Compute the function
    res.backward()         # Compute gradients
    optimizer.step()       # Update lambda_k
    if step % 100 == 0:    # Print info every 100 steps
        print(f"Step {step}, Function Value: {res.item()}")

print("lambdahat:", lambda_k) #print("Final Gradient:", lambda_k.grad)
lambda_k.grad = None
u_t_x = torch.zeros((T+1, X))
for t in range(T-1, -1, -1):
    u_t_x[t,:] = torch.log(torch.exp((phi_t_x_y_k[t,:,:,:].reshape((-1, K)) @ lambda_k + u_t_x[t+1,:] @ P_xprime_xy).reshape((X, Y))).sum(axis=1))

du = torch.zeros((T, X, K))
d2u = torch.zeros((T, X, K, K))
for t in range(T):
    for x in range(X):
          grad1 = torch.autograd.grad(u_t_x[t, x], lambda_k, create_graph=True, retain_graph=True, allow_unused=True)[0]
          du[t, x] = grad1.detach()
          for k in range(K):
              grad2 = torch.autograd.grad(grad1[k], lambda_k, retain_graph=True, allow_unused=True)[0]
              d2u[t, x, k] = grad2.detach()

neg_hess =  (- muhat_txy.T @ Psi @ d2u.numpy().reshape((T*X,-1))).reshape((K,K))
HinvB = np.linalg.solve(neg_hess, (phi_t_x_y_k.reshape((-1,K)) + Psi @ du.numpy().reshape((T*X,-1) )).T)
print('Cov(lambdahat)=')
print(HinvB @ V @ HinvB.T)