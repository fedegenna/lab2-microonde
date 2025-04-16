import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Modelli di dipendenza
def model_1_over_R(R, a, b):
    return a / R + b

def model_1_over_R2(R, a, b):
    return a / R**2 + b

def model_intermediate(R, a, b, c):
    return a / R**c + b

def main():

    R = 82.1 * np.ones(20) - np.array([45.3, 43.8, 42.3, 41., 39.5, 38.1, 36.6, 35, 33.6, 32.4, 30.9, 29.5, 28.1, 26.6, 25.8, 23.9, 22.4, 21., 19.6, 18.1])  # Raggi dei massimi
    err_R =  0.1 * np.ones(len(R))  # Errori associati ai raggi
    M = 3 * np.array([0.92, 0.92, 0.88, 0.86, 0.84, 0.8, 0.76, 0.72, 0.66, 0.66, 0.66, 0.64, 0.64, 0.62, 0.6, 0.56, 0.54, 0.52, 0.5, 0.48]) #segnale
    err_M = 0.06 *np.ones(len(M))  # Errori associati a M

    # Propagazione degli errori su M considerando err_R
    err_M_total = err_M

    # Fit con il modello 1/R
    least_squares_1_over_R = LeastSquares(R, M, err_M_total, model_1_over_R)
    minuit_1_over_R = Minuit(least_squares_1_over_R, a=1, b=1)
    minuit_1_over_R.migrad()
    chi2_1_over_R = minuit_1_over_R.fval
    ndof_1_over_R = len(M) - len(minuit_1_over_R.values)
    p_value_1_over_R = 1 - chi2.cdf(chi2_1_over_R, ndof_1_over_R)

    # Fit con il modello 1/R^2
    least_squares_1_over_R2 = LeastSquares(R, M, err_M_total, model_1_over_R2)
    minuit_1_over_R2 = Minuit(least_squares_1_over_R2, a=1, b=1)
    minuit_1_over_R2.migrad()
    chi2_1_over_R2 = minuit_1_over_R2.fval
    ndof_1_over_R2 = len(M) - len(minuit_1_over_R2.values)
    p_value_1_over_R2 = 1 - chi2.cdf(chi2_1_over_R2, ndof_1_over_R2)

    # Fit con il modello intermedio
    least_squares_intermediate = LeastSquares(R, M, err_M_total, model_intermediate)
    minuit_intermediate = Minuit(least_squares_intermediate, a=1, b=1, c=1)
    minuit_intermediate.migrad()
    chi2_intermediate = minuit_intermediate.fval
    ndof_intermediate = len(M) - len(minuit_intermediate.values)
    p_value_intermediate = 1 - chi2.cdf(chi2_intermediate, ndof_intermediate)

    # Stampa dei risultati
    print("Modello 1/R:")
    print(f"  χ² = {chi2_1_over_R:.2f}, gradi di libertà = {ndof_1_over_R}, p-value = {p_value_1_over_R:.4f}")
    print("Modello 1/R²:")
    print(f"  χ² = {chi2_1_over_R2:.2f}, gradi di libertà = {ndof_1_over_R2}, p-value = {p_value_1_over_R2:.4f}")
    print("Modello 1/R^c:")
    print(f"  χ² = {chi2_intermediate:.2f}, gradi di libertà = {ndof_intermediate}, p-value = {p_value_intermediate:.4f}")

    # Grafico
    fig, ax = plt.subplots()
    R_fit = np.linspace(min(R), max(R), 100)

    ax.errorbar(R, M, xerr=err_R, yerr=err_M_total, fmt="o", label="Dati con errori")
    ax.plot(R_fit, model_1_over_R(R_fit, minuit_1_over_R.values["a"], minuit_1_over_R.values["b"]), label="Modello 1/R", color="red")
    ax.plot(R_fit, model_1_over_R2(R_fit, minuit_1_over_R2.values["a"], minuit_1_over_R2.values["b"]), label="Modello 1/R^2", color="green")
    ax.plot(R_fit, model_intermediate(R_fit, minuit_intermediate.values["a"], minuit_intermediate.values["b"], minuit_intermediate.values["c"]), label=f"Modello 1/R^c (c={minuit_intermediate.values['c']:.2f})", color="blue")

    ax.set_xlabel("Raggio R")
    ax.set_ylabel("Misura M")
    ax.set_title("Interpolazione di M in funzione di R")
    ax.legend()
    
    plt.show()
    print(M)
if __name__ == "__main__":
    main()