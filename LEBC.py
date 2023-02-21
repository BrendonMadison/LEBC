#Test an input laser beam hitting the ILC250 electron beam
#Written by Brendon Madison as a part of research into new beam diagnostic measures and luminosity measures
#
#DATE_v0: 21st Feb. 2023
#
#Contents:
#
#FourVec class for handling Lorentz Fourvectors in python, numpy
#
#BeamVec class for defining particle beams and randomly generating particles for them
#
#GetCompton function which solves for differential cross-section in electron rest frame
#
#GetBoostedCompton function which takes FourVecs for photon and electron , boosts into electron rest frame ,
#####calls GetCompton and then boosts back into the lab fame
#
#PlotCompton which handles plotting the results from GetBoostedCompton
#
#THIS IS THE STAND ALONE VERSION! IT SHOULDN"T DO ANYTHING?
#
import matplotlib.pyplot as plt
from scipy.constants import e, h, hbar, alpha, c, m_e
import numpy as np
hev = 4.135667696e-15 #in units of eV/sec
#fundamentals folded together
f = (hbar * alpha / m_e / c)**2 / 2
#joules to eV
jte = 6.242e+18
#thomson scattering cross-section
thomx = 6.6524587321e-29
#electron mass in GeV
mme = 0.0005109989461

#Total number of particles in the beam packet
Npacket = 2.0e10

class FourVec:
    def __init__(self, et=1.0, x=0, y=0, z=0):
        self.et = et
        self.x = x
        self.y = y
        self.z = z
        self.m = np.sqrt(et*et - x*x - y*y - z*z)
        self.p = np.sqrt(x*x + y*y + z*z)
        self.pt = np.sqrt(x*x + y*y)
        self.theta = np.arccos((self.z)/self.p)
        self.costheta = (self.z)/self.p
        self.phi = np.arctan2(self.y,self.x)
        self.rapidityX = np.arctanh(self.x/self.et)
        self.rapidityY = np.arctanh(self.y/self.et)
        self.rapidityZ = np.arctanh(self.z/self.et)
    def Update(self):
        self.m = np.sqrt(self.et*self.et - self.x*self.x - self.y*self.y - self.z*self.z)
        self.p = np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        self.pt = np.sqrt(self.x*self.x + self.y*self.y)
        self.theta = np.arccos((self.z)/self.p)
        self.costheta = (self.z)/self.p
        self.phi = np.arctan2(self.y,self.x)
        self.rapidityX = np.arctanh(self.x/self.et)
        self.rapidityY = np.arctanh(self.y/self.et)
        self.rapidityZ = np.arctanh(self.z/self.et)
    def Boost(self,rpd,ax):
        #Given the axis (ax) and the rapidity (rpd) perform a boost
        if ax == "x" or ax == "X":
            newe = np.cosh(rpd)*self.et + np.sinh(rpd)*self.x
            newx = np.sinh(rpd)*self.et + np.cosh(rpd)*self.x
            self.et = newe
            self.x = newx
            self.Update()
        if ax == "y" or ax == "Y":
            newe = np.cosh(rpd)*self.et + np.sinh(rpd)*self.y
            newy = np.sinh(rpd)*self.et + np.cosh(rpd)*self.y
            self.et = newe
            self.y = newy
            self.Update()
        if ax == "z" or ax == "Z":
            newe = np.cosh(rpd)*self.et + np.sinh(rpd)*self.z
            newz = np.sinh(rpd)*self.et + np.cosh(rpd)*self.z
            self.et = newe
            self.z = newz   
            self.Update()
    def InvBoost(self,rpd,ax):
        #Given the axis (ax) and the rapidity (rpd) perform an inverse boost
        if ax == "x" or ax == "X":
            newe = np.cosh(rpd)*self.et - np.sinh(rpd)*self.x
            newx = -1.0*np.sinh(rpd)*self.et + np.cosh(rpd)*self.x
            self.et = newe
            self.x = newx
            self.Update()
        if ax == "y" or ax == "Y":
            newe = np.cosh(rpd)*self.et - np.sinh(rpd)*self.y
            newy = -1.0*np.sinh(rpd)*self.et + np.cosh(rpd)*self.y
            self.et = newe
            self.y = newy
            self.Update()
        if ax == "z" or ax == "Z":
            newe = np.cosh(rpd)*self.et - np.sinh(rpd)*self.z
            newz = -1.0*np.sinh(rpd)*self.et + np.cosh(rpd)*self.z
            self.et = newe
            self.z = newz
            self.Update()
    def Rotate(self,tht,ax):
        #Rotate xyz by the amount of theta and phi given
        if ax == "XY" or ax == "xy":
            newx = self.x * np.cos(tht) - self.y * np.sin(tht)
            newy = self.x * np.sin(tht) + self.y * np.cos(tht)
            self.x = newx
            self.y = newy
            self.Update()
        if ax == "XZ" or ax == "xz":
            newx = self.x * np.cos(tht) - self.z * np.sin(tht)
            newz = self.x * np.sin(tht) + self.z * np.cos(tht)
            self.x = newx
            self.z = newz
            self.Update()
        if ax == "YZ" or ax == "yz":
            newy = self.y * np.cos(tht) - self.z * np.sin(tht)
            newz = self.y * np.sin(tht) + self.z * np.cos(tht)
            self.y = newy
            self.z = newz
            self.Update()
            
class BeamVec:
    def __init__(self, N=0,E=0, x=0, y=0, z=0,m=0):
        #N is number of particles per packet
        #E is beam energy in GeV
        #x,y,z are rms spreads in meters
        #m is mass of the particle used in the beam in GeV
        self.N = N
        self.E = E
        self.x = x
        self.y = y
        self.z = z
        self.m = m
        #Get the number density of the center 1-sigma region
        self.density = N*0.68/(x*y*z)
    def LumiCalc(self,bc,ax):
        #Takes another BeamVec class and calculates the geometric luminosity
        if ax == "x" or ax == "X":
            return self.N*bc.N/(4.0*np.pi*np.sqrt(self.x*self.y*bc.x*bc.z))
        if ax == "y" or ax == "Y":
            return self.N*bc.N/(4.0*np.pi*np.sqrt(self.x*self.y*bc.y*bc.z))
        if ax == "z" or ax == "Z":
            return self.N*bc.N/(4.0*np.pi*np.sqrt(self.x*self.y*bc.x*bc.y))
    def GeneratePoints(self,num):
        #returns points corresponding to the beam with uncorrelated rms spreads
        mean = [0,0,0]
        cov = [[self.x,0,0],[0,self.y,0],[0,0,self.z]]
        return np.random.multivariate_normal(mean,cov,size=num)

def GetCompton(S,AngRes):
    #Get the differential cross section and energy (wavelength) spectra
    #for Compton scattering in the electron rest frame
    #input E should be in GeV
    #also takes the angular resolution (in radians)
    #returns the angles, differential cross-section and energies
    #Incoming photon frequency (s-1) and wavelength (m).
    theta = np.arange(0, np.pi, AngRes)
    nu = S * 1.e9 / hev
    lam = c / nu

    # Scattered photon wavelength (m).
    lamp = lam + h / m_e / c * (1 - np.cos(theta))
    newene = 1.0e-9 * jte*h*c/lamp
    P = lam / lamp
    # Differential cross section given by the Klein-Nishina formula.
    dsigma_dOmega = f * P**2 * (P + 1/P - np.sin(theta)**2)
    
    return theta , newene, dsigma_dOmega

def GetBoostedCompton(GamVec,EleVec,AngRes):
    #Computes Compton scattering given a boosted electron(positron)
    #GamVec and EleVec must be FourVec objects
    #Angular resolution is the step (in radians) for theta and phi
    #This particular solution assumes the input electron is along the z axis.
    #Get the initial rapidities as you'll need these to inverse boost and boost 
    rpz = EleVec.rapidityZ
    #Boost the photon into the electron rest frame
    GamVec.InvBoost(rpz,"Z")
    #We need to align the photon to the z axis so we can use spherical coordinates later
    phia = 1.0*GamVec.phi
    that = -1.0*GamVec.theta
    #Rotate the photon so that all momentum is along z
    GamVec.Rotate(phia,"XY")
    GamVec.Rotate(that,"XZ")
    #Computes the compton scattering as in the electron rest frame
    tht, ene , dsdo = GetCompton(GamVec.et,AngRes)
    phi = np.arange(0,2.0*np.pi,AngRes)
    #angles in the electron rest frame
    eang = []
    #energies in the lab frame
    enes = []
    #angles in the lab frame
    lang = []
    #cross-section
    xsec = []
    #list of photon FourVec in case you want to use them later
    ng = []
    for i in range(len(phi)):
        for j in range(len(tht)):
            eang.append([tht[j],phi[i]])
            #The new photon four vector
            newgam = FourVec(ene[j],ene[j]*np.sin(tht[j])*np.cos(phi[i]),ene[j]*np.sin(tht[j])*np.sin(phi[i]),np.cos(tht[j])*ene[j])
            newgam.Rotate(-1.0*that,"XZ")
            newgam.Rotate(-1.0*phia,"XY")
            newgam.Boost(rpz,"Z")
            enes.append(newgam.et)
            lang.append([newgam.theta,newgam.phi])
            #print(tmpvec.theta,tmpvec.phi)
            xsec.append(dsdo[j])
            ng.append(newgam)
    return np.array(eang),np.array(enes),np.array(lang),np.array(xsec),ng
    
def PlotCompton(pang,pene,epz,angres):
    #Takes photon crossing angle, photon energy, electron momentum and angular resolution in radians
    #Plots the compton scattering polar plots

    #Define the FourVec given the values
    PhoVec1 = FourVec(pene,0.0,np.sin(pang)*pene,np.cos(pang)*pene)
    EleVec1 = FourVec(np.sqrt(epz**2 + mme**2),0.0,0.0,epz)
    #Call the BoostedCompton function to get the required values
    ea,en,la,xs,newg = GetBoostedCompton(PhoVec1,EleVec1,angres)
    #Relay the photon energy extrema to the user
    print("Energy extrema:\n",np.amin(en),np.amax(en))

    DPI=100
    fig = plt.figure(figsize=(800/DPI, 1000/DPI))
    fig.suptitle(r"$p_{-}=$"+str("%.2e" % epz)+r" , $E_{\gamma}=$"+str("%.2e" % pene)+" Compton Scattering Electron Rest Frame")
    ax1 = fig.add_subplot(221, projection='polar')
    ax2 = fig.add_subplot(222, projection='polar')
    ax3 = fig.add_subplot(223, projection='polar')
    ax4 = fig.add_subplot(224, projection='polar')
    ax1.scatter([item[0] for item in ea],en,alpha=0.1)
    ax1.set_ylim(0.0,np.amax(en)*1.10)
    ax2.scatter([item[1] for item in ea],en,alpha=0.1)
    ax2.set_ylim(0.0,np.amax(en)*1.10)
    ax3.scatter([item[0] for item in ea],xs*1.0e31,alpha=0.1)
    ax3.set_ylim(0.0,np.amax(xs)*1.10*1.0e31)
    ax4.scatter([item[1] for item in ea],xs*1.0e31,alpha=0.1)
    ax4.set_ylim(0.0,np.amax(xs)*1.10*1.0e31)
    ax1.set_title("Photon Energy [GeV], Theta Plot")
    ax2.set_title("Photon Energy [GeV], Phi Plot")
    ax3.set_title("XSection [mb], Theta Plot")
    ax4.set_title("XSection [mb], Phi Plot")
    plt.savefig(str("%.2e" % epz)+str("%.2e" % pene)+"EleRestFrame.pdf")
    plt.show()

    fig = plt.figure(figsize=(800/DPI, 1000/DPI))
    fig.suptitle(r"$p_{-}=$"+str("%.2e" % epz)+r" , $E_{\gamma}=$"+str("%.2e" % pene)+" Compton Scattering Lab Frame")
    ax1 = fig.add_subplot(221, projection='polar')
    ax2 = fig.add_subplot(222, projection='polar')
    ax3 = fig.add_subplot(223, projection='polar')
    ax4 = fig.add_subplot(224, projection='polar')
    ax1.scatter([item[0] for item in la],en,alpha=0.1)
    ax1.set_ylim(0.0,np.amax(en)*1.10)
    ax2.scatter([item[1] for item in la],en,alpha=0.1)
    ax2.set_ylim(0.0,np.amax(en)*1.10)
    ax3.scatter([item[0] for item in la],xs*1.0e31,alpha=0.1)
    ax3.set_ylim(0.0,np.amax(xs)*1.10*1.0e31)
    ax4.scatter([item[1] for item in la],xs*1.0e31,alpha=0.1)
    ax4.set_ylim(0.0,np.amax(xs)*1.10*1.0e31)
    ax1.set_title("Photon Energy [GeV], Theta Plot")
    ax2.set_title("Photon Energy [GeV], Phi Plot")
    ax3.set_title("XSection [mb], Theta Plot")
    ax4.set_title("XSection [mb], Phi Plot")
    plt.savefig(str("%.2e" % epz)+str("%.2e" % pene)+"LabFrame.pdf")
    plt.show()
