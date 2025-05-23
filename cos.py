import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Modelli di dipendenza
def model_cos_theta(cos_theta, a, b):
    return a * cos_theta + b

def model_cos_theta_squared(cos_theta, a, b):
    return a * cos_theta**2 + b

def model_intermediate(cos_theta, a, b, c):
    return a * cos_theta**c + b

def main():
    # Dati simulati (sostituisci con i tuoi dati reali)
    theta = np.linspace(0, np.pi / 2, 17)  # Angoli in radianti
    err_theta = 1 * np.pi / 180 * np.ones(len(theta))  # Errori associati agli angoli
    cos_theta = np.cos(theta)  # Calcolo di cos(theta)
    err_cos_theta = np.abs(np.sin(theta) * err_theta)  # Propagazione degli errori su cos(theta)

    M_repeated = np.array([
        [2.23, 2.18, 2.13],  # Misure ripetute per θ=0
        [2.22, 2.16, 2.12],  # Misure ripetute per θ=1
        [2.18, 2.11, 2.08],  # Misure ripetute per θ=2
        [2.12, 2.06, 2.01],  # Misure ripetute per θ=3
        [2.05, 1.98, 1.95],  # Misure ripetute per θ=4
        [1.94, 1.87, 1.85],  # Misure ripetute per θ=5
        [1.84, 1.73, 1.73],  # Misure ripetute per θ=6
        [1.72, 1.59, 1.61],  # Misure ripetute per θ=7
        [1.54, 1.44, 1.47],  # Misure ripetute per θ=8
        [1.37, 1.29, 1.31],  # Misure ripetute per θ=9
        [1.19, 1.12, 1.15],  # Misure ripetute per θ=10
        [1.02, 0.95, 0.95],  # Misure ripetute per θ=11
        [0.81, 0.75, 0.76],  # Misure ripetute per θ=12
        [0.60, 0.54, 0.56],  # Misure ripetute per θ=13
        [0.42, 0.33, 0.38],  # Misure ripetute per θ=14
        [0.22, 0.17, 0.21],  # Misure ripetute per θ=15
        [0.09, 0.06, 0.08],  # Misure ripetute per θ=16
    ])
    M = np.mean(M_repeated, axis=1)  # Media delle misure
    err_M = np.std(M_repeated, axis=1, ddof=1) / np.sqrt(M_repeated.shape[1])  # Deviazione standard della media

    # Funzione di costo modificata per includere gli errori su cos(theta)
    def cost_function(cos_theta, M, err_M, err_cos_theta, model, *params):
        residuals = (M - model(cos_theta, *params)) / err_M
        penalty = (err_cos_theta / err_M) ** 2
        return np.sum(residuals**2 + penalty)

    # Fit con il modello cos(theta)
    least_squares_cos_theta = LeastSquares(cos_theta[:len(M)], M, err_M, model_cos_theta)
    minuit_cos_theta = Minuit(least_squares_cos_theta, a=2, b=0.1)
    minuit_cos_theta.migrad()
    chi2_cos_theta = minuit_cos_theta.fval
    ndof_cos_theta = len(M) - len(minuit_cos_theta.values)
    p_value_cos_theta = 1 - chi2.cdf(chi2_cos_theta, ndof_cos_theta)

    # Fit con il modello cos(theta)^2
    
    least_squares_cos_theta_squared = LeastSquares(cos_theta[:len(M)], M, err_M, model_cos_theta_squared)
    minuit_cos_theta_squared = Minuit(least_squares_cos_theta_squared, a=1, b=0)
    minuit_cos_theta_squared.migrad()
    chi2_cos_theta_squared = minuit_cos_theta_squared.fval
    ndof_cos_theta_squared = len(M) - len(minuit_cos_theta_squared.values)
    p_value_cos_theta_squared = 1 - chi2.cdf(chi2_cos_theta_squared, ndof_cos_theta_squared)

    # Fit con il modello intermedio
    least_squares_intermediate = LeastSquares(cos_theta[:len(M)], M, err_M, model_intermediate)
    minuit_intermediate = Minuit(least_squares_intermediate, a=1, b=0, c=2)
    minuit_intermediate.migrad()
    chi2_intermediate = minuit_intermediate.fval
    ndof_intermediate = len(M) - len(minuit_intermediate.values)
    p_value_intermediate = 1 - chi2.cdf(chi2_intermediate, ndof_intermediate)
    
    # Stampa dei risultati
    print("Modello cos(θ):")
    print(f"  χ² = {chi2_cos_theta:.2f}, gradi di libertà = {ndof_cos_theta}, p-value = {p_value_cos_theta:.4f}")
    
    print("Modello cos²(θ):")
    print(f"  χ² = {chi2_cos_theta_squared:.2f}, gradi di libertà = {ndof_cos_theta_squared}, p-value = {p_value_cos_theta_squared:.4f}")
    print("Modello cos^c(θ):")
    print(f"  χ² = {chi2_intermediate:.2f}, gradi di libertà = {ndof_intermediate}, p-value = {p_value_intermediate:.4f}")
    
    # Grafico
    fig, ax = plt.subplots()
    cos_theta_fit = np.linspace(0, 1, 100)

    ax.errorbar(cos_theta[:len(M)], M, xerr=err_cos_theta[:len(M)], yerr=err_M, fmt="o", label="Dati con errori")
    ax.plot(cos_theta_fit, model_cos_theta(cos_theta_fit, minuit_cos_theta.values["a"], minuit_cos_theta.values["b"]), label="Modello cos(θ)", color="red")
    
    ax.plot(cos_theta_fit, model_cos_theta_squared(cos_theta_fit, minuit_cos_theta_squared.values["a"], minuit_cos_theta_squared.values["b"]), label="Modello cos²(θ)", color="green")
    ax.plot(cos_theta_fit, model_intermediate(cos_theta_fit, minuit_intermediate.values["a"], minuit_intermediate.values["b"], minuit_intermediate.values["c"]), label=f"Modello cos^c(θ) (c={minuit_intermediate.values['c']:.2f})", color="blue")
    
    ax.set_xlabel("cos(θ)")
    ax.set_ylabel("Misura M")
    ax.set_title("Interpolazione di M in funzione di cos(θ)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
