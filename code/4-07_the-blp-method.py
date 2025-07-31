    !pip install sparse
import numpy as np, scipy.sparse as sp, sparse as spx
from types import SimpleNamespace
#
def generate(T=5,Y=3,I=10,M=3,Kx=3,L=3,nbend=2,X=2,seed=77):
    np.random.seed(seed)
    D, S, Ke, K,db = Kx+2+nbend, L, 2, Kx+2,SimpleNamespace()
    x_y = np.arange(Y) if X is None else \
     np.random.choice(np.arange(X),(Y))
    db.p_ty = np.exp( np.random.normal(size = (T*Y)) / 5) / 2
    t2 = (db.p_ty * db.p_ty)[:,None] / 2 
    db.phie_ty_k= np.concatenate((db.p_ty[:,None], t2),axis=1)
    db.phieprime_ty_k = np.concatenate( \
     (np.ones((T*Y,1)),db.p_ty[:,None]),axis=1 )
    db.phix_ty_k = np.random.normal( size= (T*Y,Kx)) / 10
    db.phi_ty_k = np.block([[db.phie_ty_k,db.phix_ty_k]])
    kappa_ty=db.p_ty*0.5+np.random.normal(size = (T*Y))/10
    U_ty = db.phi_ty_k @  np.ones(K) + kappa_ty
    t1 = np.random.uniform (size=(T*Y,D-Ke))*db.p_ty[:,None] +\
     np.random.normal (size=(T*Y,D-Ke))
    db.zeta_ty_d=np.concatenate((t1,db.phie_ty_k),axis=1)
    eta_i_m = np.random.uniform(size = (I,M)) / 10
    t1 = -0.3 *db.p_ty.reshape((T,Y,1) ) + \
     + 0.95 * np.random.normal(size = (T,Y,1)) / 10 
    t2 = np.random.uniform(size = (T,Y,M-1)) / 10
    xi_t_y_m=np.concatenate(( t1,t2), axis=2)
    dxidp_t_y_m = np.concatenate (\
     (- 0.3 * np.ones((T,Y,1)) , np.zeros((T,Y,M-1))),axis=2)
    db.nu_tiy_m=(eta_i_m[None,:,None,:]*xi_t_y_m[:,None,:,:]).\
     reshape((T*I*Y,M))
    nutau_t_i_y=3* (db.nu_tiy_m).sum(axis= 1).reshape((T,I,Y))
    u_t_i = np.log(1 + np.exp(U_ty.reshape((T,Y))[:,None,:] +\
     nutau_t_i_y).sum(axis = 2))
    db.pihat_ty  = np.exp(U_ty.reshape((T,Y))[:,None,:] - \
     u_t_i[:,:,None]).mean(axis = 1).flatten()
    db.psi_ty_l = np.concatenate( (- db.p_ty[:,None], \
     np.random.normal(size = (T*Y,L-1))),axis=1)
    err_ty = -0.5*db.p_ty +  np.random.normal(size =(T*Y))
    db.chi_ty_s = db.psi_ty_l.copy()
    H_t_y_t_y = np.zeros((T,Y,T,Y))
    for t in range(T):
        H_t_y_t_y[t,:,t,:] = 1 * (x_y[:,None] == x_y[None,:])
    db.eta_im,db.xi_tym = eta_i_m.flatten(),xi_t_y_m.flatten() 
    db.dxidp_tym=dxidp_t_y_m.flatten()
    db.H_ty_ty = sp.coo_array(H_t_y_t_y.reshape((T*Y,T*Y)))
    for name, value in dict(T=T, Y=Y, I=I, M=M,Ke=Ke, Kx=Kx,\
     K=K, L=L, D=D, S=S, N= Ke+M).items():
        setattr(db, name, value)
    db.SigmaI = sp.kron(sp.kron(eye(T),np.ones((1,I))),eye(Y))
    db.SigmaY = sp.kron(eye(T*I),np.ones((1,Y)))
    db.nuprime_tiy_m = (eta_i_m[None,:,None,:] * \
     dxidp_t_y_m[:,None,:,:]).reshape((T*I*Y,M))
    db.dphieoverdp_ty_tyk = rkron( eye(db.T * db.Y),
                                  db.phieprime_ty_k)
    db.dnuoverdp_tiy_tym = rkron(sp.kron(eye(db.T) , sp.kron(\
     np.ones((db.I,1)),eye(db.Y))), db.nuprime_tiy_m, )
    return  db
#
# auxiliary functions
def eye(n):
    return sp.coo_array(sp.eye(n))
#
def spx_coo_to_sp(x):
    from scipy.sparse import coo_array
    return coo_array((x.data, x.coords), shape=x.shape)
#
def rkron(A, B):
    A = sp.csr_matrix(A)
    B = sp.csr_matrix(B)
    return sp.coo_array(sp.vstack([sp.kron(A.getrow(i), \
     B.getrow(i)) for i in range( A.shape[0])]))
#
def ipfp(db, nutau_tiy, tol=1e-5, maxiter = 100000):
    nutau_t_i_y=nutau_tiy.reshape((db.T,db.I,db.Y))
    pihat_t_y = db.pihat_ty.reshape((db.T,db.Y))
    U_t_y = np.zeros((db.T,db.Y))
    for iter in range(maxiter):
        max_t_i = np.maximum((U_t_y[:,None,:] 
                              + nutau_t_i_y).max(axis = 2),0)
        exp_t_i_y = np.exp(U_t_y[:,None,:]
                           + nutau_t_i_y - max_t_i[:,:,None])
        u_t_i = max_t_i + np.log (np.exp(-max_t_i) +
                                  exp_t_i_y.sum(axis = 2))
        max_t_y = (nutau_t_i_y - u_t_i[:,:,None]).max(axis = 1)
        exp_t_i_y = np.exp(nutau_t_i_y - u_t_i[:,:,None] 
                           - max_t_y[:,None,:])
        Uprime_t_y = np.log(db.I * pihat_t_y) - \
         (max_t_y+ np.log( exp_t_i_y.sum(axis=1) ) )
        if np.abs(U_t_y - Uprime_t_y).max()<tol:
            break
        else:
            U_t_y = Uprime_t_y 
    if (iter>=maxiter-1):
        print('ipfp failed')
    return U_t_y.flatten(), u_t_i.flatten()
#
# BLP procedure
#
def F(theta, W = None):
    tau_m = theta[:db.M].reshape((-1,1))
    alpha_ke = theta[db.M:].reshape((-1,1))
    # step 1
    nutau_tiy = (db.nu_tiy_m @ tau_m).flatten()
    Uhat_ty, uhat_ti =  ipfp(db,nutau_tiy)
    # step 2
    fhat_tiy = np.exp(db.SigmaI.T @ Uhat_ty+ nutau_tiy 
                      - db.SigmaY.T @ uhat_ti)
    Deltafhat = sp.diags_array(fhat_tiy )
    # step 3
    Rfhat = eye(db.T*db.I*db.Y) - sp.kron(eye(db.T*db.I), \
     np.ones((db.Y,db.Y))) @ Deltafhat
    # step 4
    part1 = db.SigmaI.T @ db.dphieoverdp_ty_tyk @ \
     sp.kron( np.eye(db.T*db.Y) ,alpha_ke )
    part2=db.dnuoverdp_tiy_tym @ sp.kron(eye(db.T*db.Y),tau_m)
    dVoverdp = part1+part2
    # step 5
    dpioverdp = db.SigmaI @ Deltafhat @ Rfhat @ dVoverdp/db.I
    Dhat = db.H_ty_ty * dpioverdp.T / db.I
    # step 6
    chat_ty = db.p_ty + sp.linalg.spsolve(sp.csc_matrix(Dhat),
                                          db.pihat_ty)
    # step 7
    b1 = (Uhat_ty.reshape((-1,1)) - db.phie_ty_k @ alpha_ke)
    YY = np.block([[ b1],[chat_ty.reshape((-1,1))]]) 
    XX = sp.bmat([[db.phix_ty_k, None],
                   [None,db.psi_ty_l]]).todense()
    ZZ = sp.bmat([[db.zeta_ty_d, None],
                   [None,db.chi_ty_s]]).todense()
    # step 8
    if W is None:
      W = np.linalg.inv(ZZ.T @ ZZ)
    Pi = ZZ @ W @ ZZ.T
    Upsilon = Pi - Pi.T @ XX @ np.linalg.inv(XX.T @Pi @XX) @ \
     XX.T @ Pi
    # step 9
    valF = (YY.T @ Upsilon @ YY / (2*db.T*db.Y)**2).item()
    # step 10
    mat1 = db.SigmaI @ Deltafhat @ db.SigmaY.T
    mat2 = db.SigmaI @ Deltafhat @ Rfhat @ db.nu_tiy_m
    dUoverdtau =  sp.linalg.spsolve(sp.csc_matrix(- db.I * \
     sp.diags_array(db.pihat_ty ) +  mat1 @ mat1.T) ,mat2) 
    # step 11
    dVhatoverdtau = db.SigmaI.T @ dUoverdtau + db.nu_tiy_m
    dVhatoverdtheta = np.block([[dVhatoverdtau,np.zeros( \
     (db.T*db.I*db.Y,db.Ke))]])
    # step 12
    transposed1 = spx.COO.from_scipy_sparse( \
     db.dnuoverdp_tiy_tym).reshape((db.T*db.I*db.Y,db.T* \
     db.Y,db.M)).transpose((0,2,1))
    d2Voverdtaudp_tiy_mty = transposed1.reshape((db.T*db.I*\
     db.Y,db.M*db.T*db.Y))
    d2Voverdpdalpha_tiy_tyk = spx.COO.from_scipy_sparse(\
     db.SigmaI.T @ db.dphieoverdp_ty_tyk)
    transposed2 = d2Voverdpdalpha_tiy_tyk.reshape((db.T*db.I*\
     db.Y,db.T*db.Y,db.Ke)).transpose((0,2,1))
    d2Voverdalphadp_tiy_kty = transposed2.reshape((db.T*db.I*\
     db.Y,db.Ke*db.T*db.Y))
    # step 13
    b1 = spx_coo_to_sp(d2Voverdtaudp_tiy_mty)
    b2 = spx_coo_to_sp(d2Voverdalphadp_tiy_kty)
    d2Voverdthetadp = sp.bmat([[b1,b2]])
    # step 14
    term1 = sp.kron(np.ones((1,db.N)) ,db.H_ty_ty)
    term2 = db.SigmaI @ Deltafhat @ Rfhat / db.I
    term3 = rkron(Rfhat @ dVhatoverdtheta, Rfhat @ dVoverdp)+\
     d2Voverdthetadp
    dDtopoverdtheta_ty_ntyprime = term1 * (term2 @ term3)
    # step 15
    dDtopoverdtheta_ty_n_typrime=dDtopoverdtheta_ty_ntyprime.\
     reshape((db.T*db.Y,-1,db.T*db.Y))
    dDoverdtheta = dDtopoverdtheta_ty_n_typrime.transpose(\
     (2,1,0)).reshape((db.T*db.Y,-1))
    # step 16
    term = dDoverdtheta @ sp.kron(eye(db.N), \
     (chat_ty-db.p_ty).reshape((-1,1)))
    dcoverdtheta = - sp.linalg.spsolve(sp.csc_matrix(Dhat),
                                       sp.csc_matrix(term))
    # step 17
    mat = np.block([[np.block([[dUoverdtau,-db.phie_ty_k]])],\
     [dcoverdtheta.todense()]])
    gradF = 2* YY.T @ Upsilon @ mat / (2*db.T*db.Y)**2
    return valF, gradF

db = generate()
F(np.ones(db.N))