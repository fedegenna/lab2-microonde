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

    R = 82.1*np.ones(20) - np.array([45.1, 43.7, 42.4, 40.9, 39.6, 38.2, 36.8, 35.3, 33.8, 32.4, 31, 29.6, 28.2, 26.8, 25.5, 24, 22.6, 21.1, 19.7, 18.3])  # Raggi dei massimi
    err_R = 0.1 * np.ones(len(R))  # Errori associati ai raggi
    M = np.array([4.26, 4.15, 4.00, 3.83, 3.68, 3.53, 3.29, 3.18, 3.01, 2.87, 2.79, 2.75, 2.72, 2.65, 2.53, 2.38, 2.23, 2.09, 1.99, 1.90]) #segnale
    err_M = 0.01*np.ones(len(M))  # Errori associati a M

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