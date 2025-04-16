import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Modelli di dipendenza


def model(sin_theta, K):
    return np.sqrt(1.85) * np.cos((K * 5.9 * 10**(-2) * sin_theta)/2) * np.sin (K * 5.9 * 10**(-2) * sin_theta) * 2 / (K * sin_theta * 1.5 * 10**(-2))


def main():
    d= 5.9 * 10**(-2)
    a= 1.5 * 10**(-2)
    M_0= 1.85
    M = [1.30, 1.25, 1.21, 1.07, 0.98, 0.87, 0.08, 0.07, 0.66, 0.71, 0.74, 0.80, 0.92, 0.97, 0.96, 0.89, 0.76, 0.64, 0.42, 0.07, 0.20, 0.57, 0.56, 0.55, 0.62, 0.64, 0.61, 0.60, 0.66, 0.63, 0.60, 0.56, 0.37, 0.20, 0.05] 
    err_M = 0.03 * np.ones(len(M))  
    theta = np.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60, 65, 70])
    err_theta_gradi = 1 * np.ones(len(theta))  # Errori associati ai raggi
    err_theta = np.radians(err_theta_gradi)  # Conversione in radianti

    # Calcolo di sin(theta) in radianti
    sin_theta = np.sin(np.radians(theta))  # Calcolo di sin(theta) in radianti
    err_sin_theta = np.abs(np.cos(np.radians(theta)) * err_theta)  # Propagazione degli errori su sin(theta)

    # Fit con il modello cos(theta) polarizzazione perpendicolare
    least_squares = LeastSquares(theta, M, err_M, model)
    minuit = Minuit(least_squares, K = 220)
    minuit.migrad()
    chi2_theta= minuit.fval
    ndof = len(M) - len(minuit.values)
    p_value = 1 - chi2.cdf(chi2_theta, ndof)

    
    # Stampa dei risultati 
    print("Modello θ:")
    print(f"  χ² = {chi2_theta:.2f}, gradi di libertà = {ndof}, p-value = {p_value:.4f}")
   
    # Stampa dei parametri del fit 
    print("Parametri del fit:")
    print(f"  K = {minuit.values['K']:.4f}")

    # Grafico
    fig, ax = plt.subplots()
    theta_fit = np.linspace(0, 70, 100)
    sin_theta_fit = np.sin(np.radians(theta_fit))  # Calcolo di sin(theta) in radianti per il fit

    ax.errorbar(sin_theta, M, xerr=err_sin_theta, yerr=err_M, fmt="o", label="Dati con errori", color="blue")
    #ax.plot(sin_theta_fit, model (sin_theta_fit, minuit.values["K"]), label="Modello sinθ", color="red")
    ax.set_xlabel("sinθ")
    ax.set_ylabel("Misura del segnale [V]")
    ax.set_title("Interpolazione del segnale in funzione di θ")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

