#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import eigh
from scipy.sparse import csc_matrix

def BulkHam(JK, N1, N2, periodic = False, flux = False):
    '''Kitaev Hamiltonian in fixed flux pattern
    
       Inputs: JK : 1*3 matrix, Jx=J[0], Jy = J[1], Jz=J[2]
               N1,N2: int
               
       Return: H matrix'''
    
    SiteNum = N1 * N2 * 2
    Hr = np.zeros([SiteNum, SiteNum],dtype=complex)
    for r in range(N2):
        for c in range(N1):
            n = r * N1 + c #label of unit cell
            '''X-bonds'''
            next = 2 * n + 1
            Hr[2*n, next] +=  1j * JK[0]  
                
            '''Y-bonds'''
            if c > 0:
                next = 2*n -1
            elif c == 0 and periodic == True:
                next = 2 * (n + N1) -1
            else:
                next = 0.5
                    
            if next % 1 == 0:
                Hr[2*n, next] += 1j * JK[1]
   
            '''Z-bonds'''
            if r > 0:
                next = 2*(n - N1) + 1
            elif r == 0 and periodic == True:
                next = 2 * (n + N1 * N2 - N1) +1
            else:
                next = 0.5
                
            if next % 1 == 0:
                if flux == True and r == N2//2 and c >= N1//2:
                    Hr[2*n, next] +=  -1j * JK[2]
                else:
                    Hr[2*n, next] +=  1j * JK[2]
                    
    Hr = (Hr + np.conj(Hr).T)/2
    return Hr

def BulkHam_csc(JK, N1, N2, periodic = False, flux = False):
    '''Kitaev Hamiltonian in fixed flux pattern
    
       Inputs: JK : 1*3 matrix, Jx=J[0], Jy = J[1], Jz=J[2]
               N1,N2: int
               
       Return: positions and values of non-zero elements of HK^0'''
    
    SiteNum = N1 * N2 * 2
    
    row = []
    col = []
    data = []
    for r in range(N2):
        
        for c in range(N1):
            temp = {} # A temporary dict
            n = r * N1 + c #label of unit cell
            '''X-bonds'''
            next = 2*n + 1
            if next in temp:
                temp[next] += 1j * JK[0]
            else: 
                temp[next] = 1j * JK[0]
                
            '''Y-bonds'''
            if c > 0:
                next = 2*n -1
            elif c == 0 and periodic == True:
                next = 2 * (n + N1) -1
            else:
                next = 0.5
                    
            if next % 1 == 0:
                if next in temp:
                    temp[next] += 1j * JK[1]
                else: 
                    temp[next] = 1j * JK[1]
   
            '''Z-bonds'''
            if r > 0:
                next = 2*(n - N1) + 1
            elif r == 0 and periodic == True:
                next = 2 * (n + N1 * N2 - N1) +1
            else:
                next = 0.5
                
            if next % 1 == 0:
                if next in temp:
                    if flux == True and r == N2//2 and c >= N1//2:
                        temp[next] +=  -1j * JK[2]
                    else:
                        temp[next] +=  1j * JK[2]
                else: 
                    if flux == True and r == N2//2 and c >= N1//2:
                        temp[next] =  -1j * JK[2]
                    else:
                        temp[next] =  1j * JK[2]
            odTemp = sorted(temp)
            for e in odTemp:
                row.append(2*n)
                col.append(e)
                data.append(temp[e])
                    
    return row, col, data

def ImpHam_csc(JK, N1, N2, h, param):
    '''H_imp = H_int + H_h
    
       Inputs: h:1*3 matrix, hx=h[0], hy = h[1], hz=h[2]
               param[0:2]: i c0 b0^a , a = x,y,z
               param[3:5]: i ca ba^a , a = x,y,z
               param[6:8]: i c0 ca , a = x,y,z
               param[9:11]: i b0^a ba^a , a = x,y,z
               param[12:14]: Lagrange multipliers
               
       Return: positions and values of non-zero elements of H_imp'''
    # position of the impurity site
    imp = N1//2 * N1 + N2//2 
    
    # positions of atoms linked to the impurity
    next = np.array([2*imp +1, 2*imp -1, 2*(imp - N1) +1]) 
        
    SiteNum = N1 * N2 * 2
    row = []
    col = []
    data = []
    
    
    '''Effective magnetic field of i c0 b0^a'''
    heff = np.zeros(3, dtype = complex)
    for i in range(3):
        heff[i] = 1j * h[i] + 1j * param[12 + i] - 1j * JK[i]* param[3 + i]
        row.append( 2*imp )
        col.append( SiteNum + i )
        data.append(heff[i])

    '''H_int : JK[a] <i c0 b0^a> i ca ba^a'''
    for i in range(3):
        row.append( next[i] )
        col.append( SiteNum + 3 + i )
        data.append( - 1j * JK[i]* param[i]  )  
        
    '''H_int : JK[a] <i c0 ca> i b0^a ba^a'''
    for i in range(3):
        row.append( SiteNum + i )
        col.append( SiteNum + 3 + i )
        data.append( 1j * JK[i]* param[6 + i] )
        
    '''H_int : （ JK[a] <i b0^a ba^a>  -1） i c0 ca'''
    for i in range(3):
        row.append( 2*imp )
        col.append( next[i] )
        data.append( 1j * JK[i] * ( param[9 + i] - 1))
        
    ''' H_la : i b0^b ib^c'''
    for i in range(3):
        row.append( SiteNum + (i+1)%3 )
        col.append( SiteNum + (i+2)%3 )
        data.append( 1j * param[12 + i])

    return row, col, data

def makeH(row, col, data, dim):
    '''Make csc format Hamiltonian
       
       Inputs : row,col :position of the non-zero elementds of Hamiltonian matrix
                data: H(row,col) = data
                dim: Hamiltonian matrix dimension
       Return: Hcsc: csc formate Hamiltonian
       
       Notice: we only have to enter half of the elements and get the other half 
               by hermition conjugate.'''
    Hcsc = csc_matrix((data, (row, col)), shape=(dim, dim),dtype = complex)
    Hcsc = (Hcsc + Hcsc.getH()) /2 
    return Hcsc

def InitParameters(m0 = 0, ma = 0, cc = -0.52489267, ua = 1, la = 0):
    '''initiate parameters'''
    param = np.zeros(15)
    for i in range(3):
        param[i] += m0 
        param[3 + i] += ma 
        param[6 + i] += cc 
        param[9 + i] += ua 
        param[12 + i] += la
    return param

import math

def Fermi(eng, T):
    '''Fermi distribution function
       Inputs: eng : Energy
               T : Temperature '''
    fermi = 1 / (1 + np.exp(eng / T) )
    return fermi

def Gfs(N1, N2, Harr):
    '''Calculate the equal time Green's functions of order parameters.
       
       Inputs: N1, N2 : number of unit cells in two directions
              Harr: Hamiltonian array
                      
       Return: Nparam: new parameters'''
    
    Nparam = np.zeros(15, dtype = complex) 
    
    # eigen values and vectors of H
    Harr *= 2
    e, v = eigh(Harr, 'L')
    mask = e < 0
    u = v[:,mask].T
    ak = np.conj(u).T @ u
    
    # position of the impurity site
    imp = N1//2 * N1 + N2//2   
    # positions of atoms linked to the impurity
    next = np.array([2*imp +1, 2*imp -1, 2*(imp - N1) +1]) 
    SiteNum = N1 * N2 * 2 
    
    for i in range(3):
        '''i c0 b0^a'''
        Nparam[i] += ak[2*imp,SiteNum +i] * 1j * 2
            
        '''i ca ba^a'''
        Nparam[3 + i] += ak[next[i],SiteNum +3 +i] * 1j * 2
            
        '''i c0 ca'''
        Nparam[6 + i] += ak[2*imp,next[i]] * 1j * 2
            
        '''i b0^a ba^a'''
        Nparam[9 + i] += ak[SiteNum +i,SiteNum +3 +i] * 1j * 2

        '''lagrange multipliers'''
        Nparam[12 + i] += Nparam[i]
        Nparam[12 + i] += ak[SiteNum+(i+1)%3,SiteNum+(i+2)%3] * 1j * 2
    if all(Nparam.imag < 1.e-10):
        return Nparam.real
    else:
        return Nparam

def Gfs_slow(N1, N2, Harr):
    '''Calculate the equal time Green's functions of order parameters.
       This code is slow however explicity write the calculation formulas
       
       Inputs: N1, N2 : number of unit cells in two directions
              Harr: Hamiltonian array
                      
       Return: Nparam: new parameters'''
    
    Nparam = np.zeros(15, dtype = complex) 
    
    # eigen values and vectors of H
    Harr *= 2
    e, v = eigh(Harr, 'L')
    cv = np.conj(v)
    
    #v = v.T
    #cv = cv.T
    
    # position of the impurity site
    imp = N1//2 * N1 + N2//2   
    # positions of atoms linked to the impurity
    next = np.array([2*imp +1, 2*imp -1, 2*(imp - N1) +1]) 
    SiteNum = N1 * N2 * 2 
    
    Num = len(e)
    for l in range(Num//2):
        fermi = 2* Fermi(e[l], T)
        for i in range(3):
            '''i c0 b0^a'''
            Nparam[i] += cv[2*imp,l] * v[SiteNum +i,l] * 1j * fermi
            
            '''i ca ba^a'''
            Nparam[3 + i] += cv[next[i],l] * v[SiteNum +3 +i,l] * 1j * fermi
            
            '''i c0 ca'''
            Nparam[6 + i] += cv[2*imp,l] * v[next[i],l] * 1j * fermi
            
            '''i b0^a ba^a'''
            Nparam[9 + i] += cv[SiteNum +i,l] * v[SiteNum +3 +i, l] * 1j * fermi

            '''lagrange multipliers'''
            Nparam[12 + i] += Nparam[i]
            Nparam[12 + i] += cv[SiteNum+(i+1)%3,l] * v[SiteNum+(i+2)%3,l] * 1j * fermi
    if all(Nparam.imag < 1.e-10):
        return Nparam.real
    else:
        return Nparam
    
def UpdateParam(err, dt, param, Nparam):
    '''Update order parameters.
    
       Inputs: dt : time step
               error: initiate error
               param: order parameters
               Nparam: Newly calculate parameters using Gfs()
       
       Outputs: error
                param: updated parameters'''
    err = 0
    delta = 0
    
    for i in range(3):
        delta = Nparam[i] - param[i]
        param[i] = param[i] + dt * delta
        err = max(err, abs(delta))

        delta = Nparam[3 + i] - param[3 + i]
        param[3 + i] = param[3 + i] + dt * delta
        err = max(err, abs(delta))

        delta = Nparam[6 + i] - param[6 + i]
        param[6 + i] = param[6 + i] + dt * delta
        err = max(err, abs(delta))
        
        delta = Nparam[9 + i] - param[9 + i]
        param[9 + i] = param[9 + i] + dt * delta
        err = max(err, abs(delta))
        
        delta = Nparam[12 + i] - param[12 + i]
        param[12 + i] = param[12 + i] + dt * delta
        err = max(err, abs(delta))
    return param, err

def Iteration(Hbulk, param, tolerance):
    '''Iteration process of solving self-consistent equations
    
       Inputs: Hbulk: csc form bulk Hamiltonain H0
               param: initiate order parameters
               tolerance: Maximum error tolerance
       
       Outputs: param : Final parameters
                error : Final error
                count : Number of iterations'''
    error = 1
    count = 0
    while(error > tolerance):
        count += 1
        Nparam = np.zeros(15)
        
        # Generate impurity Hamiltonian
        rowImp = []
        colImp = []
        dataImp = []
        rowImp, colImp, dataImp = ImpHam_csc(JK, N1,N2, h, param)
    
        Hcsc = Hbulk + makeH(rowImp, colImp, dataImp, dim)
        Harr = Hcsc.toarray()
        
        Nparam = Gfs(N1,N2, Harr)
        
        dt = 0.05 + (1-2*np.arctan(error * 100)/math.pi)*0.04
        param, error = UpdateParam(error, dt, param, Nparam)
    return param, error, count

def GSenergy(Hbulk, param, h):
    '''Calculate the ground state energy
       
       Inputs: Hbulk: bulk Hamiltonian
               param: finalized order parameters
               h : magnetic field
               
       Output: ground state energy per site.'''
    eng = 0
    
    # Generate impurity Hamiltonian
    rowImp = []
    colImp = []
    dataImp = []
    rowImp, colImp, dataImp = ImpHam_csc(JK, N1,N2, h, param)
    
    # Generate total Hamiltonian
    Hcsc = Hbulk + makeH(rowImp, colImp, dataImp, dim)
    Harr = Hcsc.toarray()
    Harr *= 2
    
    # GS energy of H_{MF}
    e,v = eigh(Harr,'L')
    for i in e:
        if i < 0:
            eng += i
    
    # Collect constant terms
    for i in range(3):
        eng += - h[i] * param[i] # magnetic field energy
        eng += JK[i] * param[i] * param[3 + i]
        eng += - JK[i] * param[6 + i] * param[9 + i]
    return eng/SiteNum

# System setups
JK = np.array([1,1,1])
#T = 1.e-5
tolerance = 1.e-6

N1 = 17
N2 = 17
SiteNum = N1 * N2 * 2
dim = SiteNum + 6

param = InitParameters()

# Generate bulk Hamiltonian: flux fee
row = []
col = []
data = []
row, col, data = BulkHam_csc(JK, N1,N2)
Hbulk = makeH(row, col, data, dim)

# Apply a (111)-direction magnetic field
hmin = 0
hmax = 1
stepNum = 20

a = np.linspace(hmin,hmax,stepNum)
parameters = np.zeros([15, stepNum])

eng = np.zeros(stepNum)
for i in range(stepNum):
    h = a[i] * np.ones(3)
    param, error, count = Iteration(Hbulk, param, tolerance)
    parameters[:,i] += param
    eng[i] = GSenergy(Hbulk, param, h)

    
# Generate bulk Hamiltonian with flux
row_flux = []
col_flux = []
data_flux = []
row_flux, col_flux, data_flux = BulkHam_csc(JK, N1, N2, flux = True)
Hbulk_flux = makeH(row_flux, col_flux, data_flux, dim)

# Apply a (111)-direction magnetic field
parameters_flux = np.zeros([15, stepNum])

eng_flux = np.zeros(stepNum)
for i in range(stepNum):
    h = a[i] * np.ones(3)
    param, error, count = Iteration(Hbulk_flux, param, tolerance)
    parameters_flux[:,i] += param
    eng_flux[i] = GSenergy(Hbulk_flux, param, h)

