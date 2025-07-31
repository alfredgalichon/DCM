# Loading Rust's Harold Zurcher data
import numpy as np
def load_Rust_data():
    pth = 'https://raw.githubusercontent.com/math-econ-code/'
    pth += 'mec_datasets/main/dynamicchoice_Rust/datafiles/'
    
    def getcleandata(name,nrow):
        filepath = pth + name + '.asc'
        thearray = np.genfromtxt(filepath, delimiter=None, dtype=float).reshape((nrow, -1), order='F')
        odometer1 = thearray[5, :]  # mileage at first replacement (0 if no replacement)
        odometer2 = thearray[8, :]  # mileage at second replacement (0 if no replacement)
        thearray = thearray[11:, :]
        replaced1 = (thearray >= odometer1) * (odometer1 > 0)  # replaced once
        replaced2 = (thearray >= odometer2) * (odometer2 > 0)  # replaced twice
        
        running_odometer = np.floor((thearray- odometer1 * replaced1 + (odometer1-odometer2)*replaced2 ) / 5000).astype(int)
        T,B = thearray.shape
        replact = np.array([[ (1 if (replaced1[t+1,b] and not replaced1[t,b]) or (replaced2[t+1,b] and not replaced2[t,b])  else 0) for b in range(B)]
                for t in range(T-1)]+
                        [[0 for b in range(B)]])
        increment = np.array([[ 0 for b in range(B)]]+
                            [[ running_odometer[t+1,b] - running_odometer[t,b] * (1-replact[t,b]) for b in range(B)] for t in range(T-1)])

        return np.block([[running_odometer.reshape((-1,1) ),replact.reshape((-1,1) ), increment.reshape((-1,1) )]])
    
    return np.vstack([getcleandata(name,nrow) for (name, nrow) in [ ('g870',36),('rt50',60),('t8h203',81),('a530875',128) ]])