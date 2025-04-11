import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Modelli di dipendenza

def model_1_over_cos2(cos_theta, a, b):
    return a * (cos_theta **2) + b


def main():
    theta = np.linspace(0, np.pi / 2, 19)  # Angoli in radianti
    err_theta = 5 * np.pi / 180 * np.ones(len(theta))  # Errori associati agli angoli
    cos_theta = np.cos(theta)  # Calcolo di cos(theta)
    err_cos_theta = np.abs(np.sin(theta) * err_theta)  # Propagazione degli errori su cos(theta)

    M_repeated = [
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
        [0.01, 0.01, 0.02],  # Misure ripetute per θ=17
        [0.00, 0.00, 0.00],  # Misure ripetute per θ=18
    ]

    M = []  # Lista per le medie
    err_M = []  # Lista per gli errori delle medie

    for r in M_repeated:
        m = np.mean(r)
        err_m = np.std(r) / np.sqrt(3)
        M.append(m)
        err_M.append(err_m)

    M = np.array(M)
    err_M = np.array(err_M)

    # Fit con il modello 1/R^2
    least_squares_1_over_cos2 = LeastSquares(cos_theta, M, err_M, model_1_over_cos2)
    minuit_1_over_cos2 = Minuit(least_squares_1_over_cos2, a=10, b=-5)
    minuit_1_over_cos2.migrad()
    chi2_1_over_cos2 = minuit_1_over_cos2.fval
    ndof_1_over_cos2 = len(M) - len(minuit_1_over_cos2.values)
    p_value_1_over_cos2 = 1 - chi2.cdf(chi2_1_over_cos2, ndof_1_over_cos2)


    # Stampa dei risultati
    
    print("Modello cos²:")
    print(f"  χ² = {chi2_1_over_cos2:.2f}, gradi di libertà = {ndof_1_over_cos2}, p-value = {p_value_1_over_cos2:.4f}")
    
    print(f"  a = {minuit_1_over_cos2.values['a']:.4f} ± {minuit_1_over_cos2.errors['a']:.4f}")
    print(f"  b = {minuit_1_over_cos2.values['b']:.4f} ± {minuit_1_over_cos2.errors['b']:.4f}")


    print("Stato del fit:", minuit_1_over_cos2.fmin.is_valid)

    # Grafico
    fig, ax = plt.subplots()
    cos_fit = np.linspace(0, 1, 100)

    ax.errorbar(cos_theta, M, xerr=err_cos_theta, yerr=err_M, fmt="o", label="Dati con errori")
    
    ax.plot(cos_fit, model_1_over_cos2(cos_fit, minuit_1_over_cos2.values["a"], minuit_1_over_cos2.values["b"]), label="Modello cos^2", color="green")
    
    ax.set_xlabel("Cos(θ)")
    ax.set_ylabel("Misura del segnale")
    ax.set_title("Interpolazione del segnale in funzione di cos(θ)")
    ax.legend()
    print("Valori di M:", M)
    print("Valore massimo di M:", np.max(M))
    plt.show()
    
if __name__ == "__main__":
    main()
