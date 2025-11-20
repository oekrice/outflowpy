"""
Pure python code for calculating outflow fields. 
Using coordinates and naming conventions from pfsspy, for compatibility with that module
"""

import numpy as np
import sys
from scipy.io import netcdf
from scipy.linalg import eigh_tridiagonal
from astropy.io import fits
#from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import outflowpy

def findvs(self, Paras):  #finds staggered v_out, unstaggered v_out and the appropriate derivative
    vc = self.rc*0
    vs = self.rs*0
    vd = self.rc*0
    for i in range(len(self.rc)):
        vc[i] = Paras.vout(self.rc[i])
        vd[i] = Paras.voutdiff(self.rc[i])
    for i in range(len(self.rs)):
        vs[i] = Paras.vout(self.rs[i])
    return vs, vc, vd

"""
Return the array corresponding to the lower boundary, based upon a specified function passed to this one
"""

def find_lower_boundary(self, lbound_fn):
    s, p = np.meshgrid(self.sc[1:-1], self.pc[1:-1],indexing='ij')
    #psi0 is the lower boundary condition, with dimensions equal to that of s and p above
    psi0 = lbound_fn(s, p)
    area = self.Sr[0,0,0]
    print('Divergence inside sphere: ', np.sum(psi0)*area)
    if abs(np.sum(psi0)*area) > 1e-10:
        raise Exception('Lower Boundary Condition not Divergence-Free')
    return psi0  #lower boundary condition

"""
Calculate the eigenvalues and eigenvectors in the azimuthal directionp. The eigenvales are integers in the infinite limit but are not necessarily so here.
"""

def findms(pc, dp): 
    # - We have to combine the eigenvectors from both the cosines and the sines, hence this is more complicated than the equivalent in the latitudinal directionp.
    num = len(pc)
    dvals = 2*np.ones((num))
    evals = -np.ones((num-1))
    dvals[0] = 1; dvals[-1] = 1
    w1,v1 = eigh_tridiagonal(dvals,evals)  #calculate sine eigenvalues
    ms1 = []
    for i in range(0,len(w1)):
        ms1.append(np.sqrt(abs(w1[i])/dp**2))
    dvals[0] = 3; dvals[-1] = 3
    w2,v2 = eigh_tridiagonal(dvals,evals)  #calculate cosine eigenvalues
    ms2 = []
    for i in range(0,len(w2)):
        ms2.append(np.sqrt(abs(w2[i])/dp**2))
    v = v1*0
    ms = ms1*0
    for i in range(len(ms1)):  #combine the two sets of eigenvalues as necessary, discarding the ones correspoinding to half-integers (which do not satisfy the boundary condition)
        if i%2 == 0:
            ms.append(ms1[i])
            v[:,i] = v1[:,i]
        else:
            ms.append(ms2[i])
            v[:,i] = v2[:,i]

    return np.array(ms), np.array(v)

"""
Calculate eigenvalues and eigenvectors in the latitudinal direction
"""

def findls(m, sc, sg, ds, ns):
    sigc = np.sqrt(1-sc**2) #sigma evaluated at the cell midpoints
    sigs = np.sqrt(1-sg**2)
    evals = -sigs[1:-1]**2  
    dvals = sigs[1:]**2 + sigs[:-1]**2 - (ds*m**2/sigc)*(np.arcsin(sg[:-1])-np.arcsin(sg[1:]))
    w,v = eigh_tridiagonal(dvals,evals)
    
    #Get all the eigenvalues to be a consistent sign, for debugging purposes. Hopefully would make no difference to things...
    for j in range(ns):
        v[:,j] = v[:,j]/np.sign(v[0,j] + 1e-10)

    return w/ds**2, v

""" 
Find the coefficients cmn in order to match the lower boundary condition
"""

def coeff(br0, q,p):
    lhs = np.sum(br0*q[:,np.newaxis]*p[np.newaxis,:])
    rhs = np.sum(q[:,np.newaxis]*q[:,np.newaxis]*p[np.newaxis,:]*p[np.newaxis,:])
    return lhs/rhs

"""
Calculate the required functions in the radial direction. Not an eigenvalue problem, unlike in the other directions.
"""

def findh(l, rcx, vcx, vdcx, dr):  #finds the H function, normalised to satisfy the lower boundary conditionp.
    #The exact calculation here shouldn't affect the solenoidal condition on B (that is ensured when calculating Q).
    hcx = np.zeros(len(rcx)) #Initialise grid. Hc is the only function with on-centre grid points

    hcx[-1] = 1; hcx[-2] = 0. #I've fiddled around with this a bit and this is the best numerical option -- matches almost exactly with PFSSpy (will include this in a test)

    for i in range(len(rcx) - 3, -1, -1): #work backwards from the top boundary
        #Calculate quantities A, B, C according to scheme described in the code
        A = 1
        B = 3 - vcx[i+1]*np.exp(rcx[i+1])
        C = 2 - l - 3*vcx[i+1]*np.exp(rcx[i+1]) - vdcx[i+1]*np.exp(rcx[i+1])
        top = hcx[i+1] * (2*A/(dr**2) - C) + hcx[i+2] * (-A/(dr**2) - B/(2*dr))
        bottom = (A/(dr**2) - B/(2*dr))
        hcx[i] = top/bottom

    grad = (hcx[1]*np.exp(rcx[1]) - hcx[0]*np.exp(rcx[0]))/dr - 0.5*(vcx[0]*np.exp(rcx[0])*hcx[0] + vcx[1]*np.exp(rcx[1])*hcx[1])

    return hcx/grad

def findg(hc,l, rg): #finds the G function from H using the specified scheme from the notes.
    gs = rg*0
    gs[0] = 1  #lower boundary condition on G
    for i in range(1,len(gs)):
        gs[i] = np.exp(-2*rg[i])*(0.5*l*hc[i]*(np.exp(2*rg[i]) - np.exp(2*rg[i-1])) + gs[i-1]*np.exp(2*rg[i-1]))
    return gs

"""
Function to calculate the magnetic field, using all of the above functions.
"""

class magnetic_field:   
    def __init__(self, Paras):
        self.v1 = Paras.v1
        self.nr = Paras.nr
        self.ns = Paras.ns
        self.nphi = Paras.nphi
        self.rcrit = Paras.r_c
        self.rss = Paras.rss
        self.r0, self.r1 = np.log(1.0), np.log(Paras.rss) #radius limit
        self.s0, self.s1 = -1,1       #theta limit
        self.p0, self.p1 = 0,2*np.pi   #phi limit
        #calculate step sizes
        self.dr = (self.r1-self.r0)/self.nr
        self.ds = (self.s1-self.s0)/self.ns
        self.dp = (self.p1-self.p0)/self.nphi
        #coordinate axes on gridpoints
        self.rs = np.linspace(self.r0,self.r1,Paras.nr+1)  
        self.ss = np.linspace(self.s0,self.s1,Paras.ns+1)
        self.ps = np.linspace(self.p0,self.p1,Paras.nphi+1)
        #coordinate axes on grid faces
        self.rc = np.linspace(self.r0-self.dr/2,self.r1 + self.dr/2,self.nr+2) 
        self.sc = np.linspace(self.s0-self.ds/2,self.s1 + self.ds/2,self.ns+2)
        self.sc[0] = self.sc[1]; self.sc[-1] = self.sc[-2]
        self.pc = np.linspace(self.p0-self.dp/2,self.p1 + self.dp/2,self.nphi+2)
        
        self.vs, self.vc, self.vd = self.findvs(Paras)

        def areas(self): #returns the areas of faces as a 3d array
            r,s,p = np.meshgrid(self.rs,self.ss,self.ps,indexing='ij')
            Sr = np.exp(2*r[:,1:,1:])*self.ds*self.dp
            Ss = 0.5 * (np.exp(2*r[1:,:,1:]) - np.exp(2*r[:-1,:,1:])) * np.sqrt(np.ones((self.nr,self.ns+1,self.nphi))-s[:-1,:,1:]**2) * self.dp
            Sp = 0.5 * (np.exp(2*r[1:,1:,:]) - np.exp(2*r[:-1,1:,:])) * (np.arcsin(s[1:,1:,:])-np.arcsin(s[1:,:-1,:]))
            return Sr, Ss, Sp

        self.Sr, self.Ss, self.Sp = areas(self)
        
        def volume(self): #used only in the energy calculation. The volume of each grid 'cube' is actually the same so this isn't necessary. But if a different coordinates system is used this can be modified.
            r,s,p = np.meshgrid(self.rs,self.sc[1:-1],self.pc[1:-1],indexing='ij')   #3d r box
            V = (4/3)*(np.exp(3*r[1:]) - np.exp(3*r[:-1]))*self.ds*self.dp
            return V
    
        self.V = volume(self)    
    

def outflow(input):
    r"""
    Compute outflow field.

    Extrapolates a 3D outflow field using an eigenfunction method in :math:`r,s,p`
    coordinates, on the dumfric grid
    (equally spaced in :math:`\rho = \ln(r/r_{sun})`,
    :math:`s= \cos(\theta)`, and :math:`p=\phi`).

    Parameters
    ----------
    input : :class:`Input`
        Input parameters.

    Returns
    -------
    out : :class:`Output`

    """

    print('Calculating outflow field')

    br0 = input.br
    #Create blank B fields with the correct dimensions (no ghost points necessary for now)
    br = np.zeros((input.grid.nr+1,input.grid.ns,input.grid.nphi))
    bs = np.zeros((input.grid.nr,input.grid.ns+1,input.grid.nphi))
    bp = np.zeros((input.grid.nr,input.grid.ns,input.grid.nphi+1))

    _ms, _trigs = input.grid.ms, input.grid.trigs

    _ls, _legs = input.grid.ls, input.grid.legs
    print('Eigenthings calculated')

    _cml = _ls*0 #Fourier coefficient matrices
    _sigs = np.sqrt(np.ones(input.grid.ns+1) - input.grid.sg**2)
    _sigc = np.sqrt(np.ones(input.grid.ns) - input.grid.sc**2)


    print('Doing Fourier/Legendre Transform and computing radial functions...')
    check = np.zeros((input.grid.ns,input.grid.nphi))
    count = 0; pcerror = 100.; oferror = 0.
    for i in range(len(_ls)):
        for j in range(len(_ls[i])):
            count += 1
            _cml[i][j] = coeff(br0, _legs[i,:,j], _trigs[:,i])  #Calculate boundary coefficients (based on orthogonality)


            if abs(_cml[i][j]) < 1e-10:
                _cml[i][j] = 0
            else:  

                _q = np.zeros((input.grid.ns+2))
                _q[1:-1] = _legs[i,:,j]  # Legendre functions
                _q[0] = _q[1]; _q[-1] = _q[-2]
                _p = np.zeros((input.grid.nphi + 2))
                _p[1:-1] = _trigs[:,i]     # Trig functions
                _p[0] = _p[-2]; _p[-1] = _p[1]
                _hcx = findh(_ls[i][j], input.grid.rcx, input.vcx, input.vdcx, input.grid.dr)     # Radial functions
                _gg = findg(_hcx,_ls[i][j],input.grid.rg)

                #Then add on each mode to the magnetic fields, differentiating as appropriate. THIS IS THE SLOW BIT! Not sure how it can be improved in python though              
                br += _cml[i,j] * _gg[:, np.newaxis, np.newaxis] * _q[np.newaxis, 1:-1, np.newaxis] * _p[np.newaxis, np.newaxis, 1:-1]
                bs += _cml[i,j] * _sigs[np.newaxis, :, np.newaxis] * _hcx[1:-1, np.newaxis, np.newaxis] * ((_q[1:] - _q[:-1])/input.grid.ds)[np.newaxis, :, np.newaxis] * _p[np.newaxis, np.newaxis, 1:-1]
                bp += _cml[i,j] * (1.0/_sigc)[np.newaxis, :, np.newaxis] * _hcx[1:-1, np.newaxis, np.newaxis] *  _q[np.newaxis, 1:-1, np.newaxis] * ((_p[1:] - _p[:-1])/input.grid.dp)[np.newaxis, np.newaxis, :]

                check += _q[1:-1, np.newaxis]*_p[np.newaxis, 1:-1]*_cml[i,j]   #calculating the lower boundary after each mode, to check against the target
                pcerror = 100.0*np.sum(np.abs(check-br0))/np.sum(np.abs(br0))
                oferror = np.abs(100.0*(np.sum(np.abs(check))-np.sum(np.abs(br0)))/np.sum(np.abs(br0)))
                if count%100 == 0:
                    print('Calculating... Lower Boundary Absolute Error: %06.2f%%, Approx Max. Open Flux Error: %06.2f%%, Modes calculated: %d/%d' % (pcerror,oferror,count,input.grid.ns*input.grid.nphi), end='\r')

    # print('Python br sum', np.sum(np.abs(br)))
    # print('Python bs sum', np.sum(np.abs(bs)))
    # print('Python bp sum', np.sum(np.abs(bp)))

    print('Calculating... Lower Boundary Absolute Error: %06.2f%%, Approx Max. Open Flux Error: %06.2f%%, Modes calculated: %d/%d' % (pcerror,oferror,count,input.grid.ns*input.grid.nphi))

    print('Magnetic field calculated. ')                
    br = np.swapaxes(br, 0, 2)
    bs = np.swapaxes(bs, 0, 2)
    bp = np.swapaxes(bp, 0, 2)
    
    return outflowpy.Output(br, bs, bp, input.grid, input.map)

def outflow_fortran(input, existing_fname = None):
    r"""
    Compute outflow field using the precompiled Fortran routine -- this is just a python wrapper.

    Extrapolates a 3D outflow field using an eigenfunction method in :math:`r,s,p`
    coordinates, on the dumfric grid
    (equally spaced in :math:`\rho = \ln(r/r_{sun})`,
    :math:`s= \cos(\theta)`, and :math:`p=\phi`).

    Parameters
    ----------
    input : :class:`Input`
        Input parameters.

    Returns
    -------
    out : :class:`Output`

    """
    from .outflow_calc import compute_outflow

    if not existing_fname:
        br, bs, bp = compute_outflow.compute_outeqm(input.br, input.grid.rg, input.grid.sg, input.grid.pg, input.grid.rcx, input.grid.sc, input.vcx, input.vdcx, input.grid.ls, input.grid.trigs, input.grid.legs)

    else:
        try:
            br = np.load(f'{existing_fname}_br.npy')
            bs = np.load(f'{existing_fname}_bs.npy')
            bp = np.load(f'{existing_fname}_bp.npy')
        except:
            print('Existing output file not found, so calculating a new one.')
            br, bs, bp = compute_outflow.compute_outeqm(input.br, input.grid.rg, input.grid.sg, input.grid.pg, input.grid.rcx, input.grid.sc, input.vcx, input.vdcx, input.grid.ls, input.grid.trigs, input.grid.legs)

    br = np.swapaxes(br, 0, 2)
    bs = np.swapaxes(bs, 0, 2)
    bp = np.swapaxes(bp, 0, 2)

    return outflowpy.Output(br, bs, bp, input.grid, input.map)

