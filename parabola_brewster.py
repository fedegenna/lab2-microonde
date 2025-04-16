import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Modelli di dipendenza


def model(theta, a, b, c):
    return a * theta**2 + b*theta + c


def main():
    # Dati simulati (sostituisci con i tuoi dati reali)
    M_perpendicolare = [0.79, 0.70, 0.69, 0.78, 0.79, 0.80, 0.80] 
    M_parallelo = [0.92, 1.05, 1.09, 1.08, 1.07, 1.05, 1.04]
    err_M = 0.1 * np.ones(len(M_perpendicolare))  
    theta = [50, 53, 54, 55, 56, 57, 58]
    err_theta = 1 * np.ones(len(theta))  # Errori associati ai raggi



    # Fit con il modello cos(theta) polarizzazione perpendicolare
    least_squares_theta_perpendicolare = LeastSquares(theta, M_perpendicolare, err_M, model)
    minuit_theta_perpendicolare = Minuit(least_squares_theta_perpendicolare, a=1, b=1, c=1)
    minuit_theta_perpendicolare.migrad()
    chi2_theta_perpendicolare = minuit_theta_perpendicolare.fval
    ndof_theta_perpendicolare = len(M_perpendicolare) - len(minuit_theta_perpendicolare.values)
    p_value_theta_perpendicolare = 1 - chi2.cdf(chi2_theta_perpendicolare, ndof_theta_perpendicolare)

    # Fit con il modello cos(theta) polarizzazione perpendicolare
    least_squares_theta_parallelo = LeastSquares(theta, M_parallelo, err_M, model)
    minuit_theta_parallelo = Minuit(least_squares_theta_parallelo, a=1, b=1, c=1)
    minuit_theta_parallelo.migrad()
    chi2_theta_parallelo = minuit_theta_parallelo.fval
    ndof_theta_parallelo = len(M_parallelo) - len(minuit_theta_parallelo.values)
    p_value_theta_parallelo = 1 - chi2.cdf(chi2_theta_parallelo, ndof_theta_parallelo)
    
    # Stampa dei risultati 
    print("Modello θ:")
    print(f"  χ² perpendicolare = {chi2_theta_perpendicolare:.2f}, gradi di libertà perpendicolare = {ndof_theta_perpendicolare}, p-value perpendicolare = {p_value_theta_perpendicolare:.4f}")
    print(f"  χ² parallelo = {chi2_theta_parallelo:.2f}, gradi di libertà perpendicolare= {ndof_theta_perpendicolare}, p-value parallelo= {p_value_theta_perpendicolare:.4f}")
    
    # Stampa dei parametri del fit PERPENDICOLARE
    print("Parametri del fit con polarizzazione perpendicolare:")
    print(f"  a = {minuit_theta_perpendicolare.values['a']:.4f}")
    print(f"  b = {minuit_theta_perpendicolare.values['b']:.4f}")
    print(f"  c = {minuit_theta_perpendicolare.values['c']:.4f}")

    # Stampa dei parametri del fit PARALLELO
    print("Parametri del fit con polarizzazione parallelo:")
    print(f"  a = {minuit_theta_parallelo.values['a']:.4f}")
    print(f"  b = {minuit_theta_parallelo.values['b']:.4f}")
    print(f"  c = {minuit_theta_parallelo.values['c']:.4f}")

    # Grafico
    fig, ax = plt.subplots()
    theta_fit = np.linspace(50, 60, 100)

    ax.errorbar(theta, M_perpendicolare, xerr=err_theta, yerr=err_M, fmt="o", label="Dati polarizzazione perpendicolare con errori", color="blue")
    ax.errorbar(theta, M_parallelo, xerr=err_theta, yerr=err_M, fmt="o", label="Dati polarizzazione parallela con errori", color ="red" )
    ax.plot(theta_fit, model (theta_fit, minuit_theta_perpendicolare.values["a"], minuit_theta_perpendicolare.values["b"], minuit_theta_perpendicolare.values["c"]), label="Modello θ, polarizzazione perpendicolare", color="blue")
    ax.plot(theta_fit, model (theta_fit, minuit_theta_parallelo.values["a"], minuit_theta_parallelo.values["b"], minuit_theta_parallelo.values["c"]), label="Modello θ, polarizzazione parallela", color="red")
    ax.set_xlabel("θ [°]")
    ax.set_ylabel("Misura M")
    ax.set_title("Interpolazione del segnale in funzione di θ")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()

