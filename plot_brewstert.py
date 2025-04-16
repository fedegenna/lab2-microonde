import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2



def main():

    M_parallelo = [0.87, 0.90, 0.95, 0.92, 1.05, 1.09, 1.08, 1.07, 1.05, 1.04] 
    M_perpendicolare = [0.85, 0.87, 0.84, 0.79, 0.70, 0.69, 0.78, 0.79, 0.80, 0.80]
    err_M = 0.1 * np.ones(len(M_parallelo))  
    theta = [35, 40, 45, 50, 53, 54, 55, 56, 57, 58]
    err_theta = 1 * np.ones(len(theta))  # Errori associati ai raggi
    # Grafico
    fig, ax = plt.subplots()
  

    ax.errorbar(theta, M_parallelo, xerr=err_theta, yerr=err_M, fmt="o", label="Dati segnale onda polarizzata parallelamente con errori", color="red")
    ax.errorbar(theta, M_perpendicolare, xerr=err_theta, yerr=err_M, fmt="o", label="Dati segnale onda polarizzata perprendicolarmente con errori", color="blue")
    ax.set_xlabel("Angolo θ (gradi)")
    ax.set_ylabel("Misura M")
    ax.set_title("Grafico di M in funzione di theta")
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    main()