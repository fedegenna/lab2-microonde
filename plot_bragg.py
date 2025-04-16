import numpy as np
import matplotlib.pyplot as plt



def main():

    M =  [3*0.14, 3*0.14, 3*0.22, 3*0.28, 3*0.33, 3*0.35, 3*0.20, 3*0.16, 3*0.06, 3*0.05, 0] 
    
    err_M = 0.1 * np.ones(len(M))  
    theta = 90* np.ones (11) - [30, 35, 36, 37, 38, 39, 40, 41, 45, 50, 55]
    err_theta = 1 * np.ones(len(theta))  # Errori associati ai raggi
    # Grafico
    fig, ax = plt.subplots()
  

    ax.errorbar(theta, M, xerr=err_theta, yerr=err_M, fmt="o", label="Dati segnale onda con errori", color="red")
    ax.set_xlabel("Angolo θ [°]")
    ax.set_ylabel("Misura del segnale [V]")
    ax.set_title("Grafico del segnale in funzione di theta")
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    main()