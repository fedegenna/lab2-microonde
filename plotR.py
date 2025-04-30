import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2



def main():

    M = [1.52, 1.45, 1.49, 1.66, 1.51, 1.74, 1.62, 1.79, 1.85, 1.74, 2.22, 2.14, 2.43, 2.78, 2.20, 2.87, 2.49, 2.62, 3.40, 2.69, 3.76, 3.54, 3.45] 
    err_M = np.zeros(len(M))  
    R = 82.1*np.ones(len(M)) - np.linspace(2, 46, len(M))  # Raggi
    err_R = 0.1 * np.ones(len(R))  # Errori associati ai raggi
    # Grafico
    fig, ax = plt.subplots()
  

    ax.errorbar(R, M, xerr=err_R, yerr=err_M, fmt="o", label="Dati con errori", color="blue")
    ax.set_xlabel("Raggio R")
    ax.set_ylabel("Misura M")
    ax.set_title("Grafico di M in funzione di R")
    ax.legend()
    plt.show()
    print(len(M))
if __name__ == "__main__":
    main()