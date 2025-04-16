import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Modelli di dipendenza
def model_1_over_R(R, a, b):
    return a / R + b
def error_model_1_over_R(R, a, b, err_R, err_a, err_b, cov_ab):
    var_f = ((1/R)**2) * err_a**2 + err_b**2 + ((a/(R**2))**2) * err_R**2 + 2 * (1/R) * cov_ab
    return np.sqrt(var_f)

def model_1_over_R2(R, a, b):
    return a / R**2 + b

def model_intermediate(R, a, b, c):
    return a / R**c + b

def main():
    # Dati:
    r_misurati = np.array([
        
    [42.4, 42.3],
    [40.9, 41.0],
    [39.6, 39.5],
    [38.2, 38.1],
    [36.8, 36.6],
    [35.3, 35.0],
    [33.8, 33.6],
    [32.4, 32.4],
    [31.0, 30.9],
    [29.6, 29.5],
    
        ])
    r_mean = []
    err_r = []
    for r in r_misurati:
        r_mean.append(np.mean(r))
        err_r.append(np.std(r, ddof=1) / np.sqrt(2))  # Errore associato alla media
    r_veri = 100*np.ones(len(r_mean)) - 17.9- r_mean  # Raggio vero in cm, ossia tenendo conto della distanza tra gli horn e il fatto che la distanza era 100cm
    err_r_veri = np.sqrt(0.1*np.ones(len(r_mean))**2 + np.array(err_r)**2)  # Errore associato al raggio vero
    print(r_veri, err_r_veri)
    voltages = 3*np.array([
    4.00,
    3.83,
    3.68,
    3.53,
    3.29,
    3.18,
    3.01,
    2.87,
    2.79,
    2.75,
    
])
    err_voltages = 0.03 * np.ones(len(voltages))  # Errore associato alla tensione


    # Fit con il modello 1/R
    least_squares_1_over_R = LeastSquares(r_veri, voltages, err_voltages, model_1_over_R)
    minuit_1_over_R = Minuit(least_squares_1_over_R, a=1, b=1)
    minuit_1_over_R.migrad()
    chi2_1_over_R = minuit_1_over_R.fval
    ndof_1_over_R = len(voltages) - len(minuit_1_over_R.values)
    p_value_1_over_R = 1 - chi2.cdf(chi2_1_over_R, ndof_1_over_R)

    # Fit con il modello 1/R^2
    least_squares_1_over_R2 = LeastSquares(r_veri, voltages, err_voltages, model_1_over_R2)
    minuit_1_over_R2 = Minuit(least_squares_1_over_R2, a=1, b=1)
    minuit_1_over_R2.migrad()
    chi2_1_over_R2 = minuit_1_over_R2.fval
    ndof_1_over_R2 = len(voltages) - len(minuit_1_over_R2.values)
    p_value_1_over_R2 = 1 - chi2.cdf(chi2_1_over_R2, ndof_1_over_R2)

    # Fit con il modello intermedio
    least_squares_intermediate = LeastSquares(r_veri, voltages, err_voltages, model_intermediate)
    minuit_intermediate = Minuit(least_squares_intermediate, a=1, b=1, c=1)
    minuit_intermediate.migrad()
    chi2_intermediate = minuit_intermediate.fval
    ndof_intermediate = len(voltages) - len(minuit_intermediate.values)
    p_value_intermediate = 1 - chi2.cdf(chi2_intermediate, ndof_intermediate)

    # Stampa dei risultati
    print("Modello 1/R:")
    print(f"  χ² = {chi2_1_over_R:.2f}, gradi di libertà = {ndof_1_over_R}, p-value = {p_value_1_over_R:.4f}")
    print(minuit_1_over_R.values["a"], minuit_1_over_R.errors["a"])
    print(minuit_1_over_R.values["b"], minuit_1_over_R.errors["b"])
    print("Modello 1/R²:")
    print(f"  χ² = {chi2_1_over_R2:.2f}, gradi di libertà = {ndof_1_over_R2}, p-value = {p_value_1_over_R2:.4f}")
    print("Modello 1/R^c:")
    print(f"  χ² = {chi2_intermediate:.2f}, gradi di libertà = {ndof_intermediate}, p-value = {p_value_intermediate:.4f}")

    # Grafico
    fig, ax = plt.subplots()
    R_fit = np.linspace(min(r_veri), max(r_veri), 100)

    ax.errorbar(r_veri, voltages, xerr=err_r_veri, yerr=err_voltages, fmt="o", label="Dati con errori")
    ax.plot(R_fit, model_1_over_R(R_fit, minuit_1_over_R.values["a"], minuit_1_over_R.values["b"]), label="Modello 1/R", color="red")
    ax.plot(R_fit, model_1_over_R2(R_fit, minuit_1_over_R2.values["a"], minuit_1_over_R2.values["b"]), label="Modello 1/R^2", color="green")
    ax.plot(R_fit, model_intermediate(R_fit, minuit_intermediate.values["a"], minuit_intermediate.values["b"], minuit_intermediate.values["c"]), label=f"Modello 1/R^c (c={minuit_intermediate.values['c']:.2f})", color="blue")

    ax.set_xlabel("Raggio R")
    ax.set_ylabel("Misura M")
    ax.set_title("Interpolazione di M in funzione di R")
    ax.legend()
    plt.show()
    
    #seconda interpolazione tenendo conto dell'errore sulle distanze:
    
    err_voltages_post = np.zeros(len(voltages))
    for i in range(len(voltages)):
        err_voltages_post[i] = np.sqrt(err_voltages[i]**2 + error_model_1_over_R(r_veri[i], minuit_1_over_R.values['a'], minuit_1_over_R.values['b'], err_r_veri[i], minuit_1_over_R.errors['a'], minuit_1_over_R.errors['b'], minuit_1_over_R.covariance[0,1])**2)
    cost_func = LeastSquares(r_veri, voltages, err_voltages_post, model_1_over_R)
    minuit = Minuit(cost_func, a=minuit_1_over_R.values["a"], b=minuit_1_over_R.values["b"])
    minuit.migrad()
    chi2_post = minuit.fval
    ndof = len(voltages) - len(minuit.values)
    p_value = 1 - chi2.cdf(chi2_post, ndof)
    print("Modello 1/R con errore sulle distanze, prima iterazione:")
    print(f"  χ² = {chi2_post:.2f}, gradi di libertà = {ndof}, p-value = {p_value:.4f}")
    print(minuit.values["a"], minuit.errors["a"])
    print(minuit.values["b"], minuit.errors["b"])
    # Grafico
    fig, ax = plt.subplots()
    R_fit = np.linspace(min(r_veri), max(r_veri), 100)
    ax.errorbar(r_veri, voltages, xerr=err_r_veri, yerr=err_voltages_post, fmt="o", label="Dati con errori")
    ax.plot(R_fit, model_1_over_R(R_fit, minuit.values["a"], minuit.values["b"]), label="Modello 1/R", color="red")
    ax.set_xlabel("Raggio R")
    ax.set_ylabel("Misura M")
    ax.set_title("Interpolazione di M in funzione di R con errore sulle distanze")
    ax.legend()
    plt.show()
    
    #terza iterazione tenendo conto dell'errore sulle distanze:
    err_voltages_post_post = np.zeros(len(voltages))
    for i in range(len(voltages)):
        err_voltages_post_post[i] = np.sqrt(err_voltages_post[i]**2 + error_model_1_over_R(r_veri[i], minuit.values['a'], minuit.values['b'], err_r_veri[i], minuit.errors['a'], minuit.errors['b'], minuit.covariance[0,1])**2)
    cost_func = LeastSquares(r_veri, voltages, err_voltages_post_post, model_1_over_R)
    minuit_post = Minuit(cost_func, a=minuit.values["a"], b=minuit.values["b"])
    minuit_post.migrad()
    chi2_post_post = minuit_post.fval
    ndof_post_post = len(voltages) - len(minuit_post.values)
    p_value_post_post = 1 - chi2.cdf(chi2_post_post, ndof_post_post)
    print("Modello 1/R con errore sulle distanze, seconda iterazione:")
    print(f"  χ² = {chi2_post_post:.2f}, gradi di libertà = {ndof_post_post}, p-value = {p_value_post_post:.4f}")
    print(minuit_post.values["a"], minuit_post.errors["a"])
    print(minuit_post.values["b"], minuit_post.errors["b"])
    # Grafico
    fig, ax = plt.subplots()
    R_fit = np.linspace(min(r_veri), max(r_veri), 100)
    ax.errorbar(r_veri, voltages, xerr=err_r_veri, yerr=err_voltages_post_post, fmt="o", label="Dati con errori")
    ax.plot(R_fit, model_1_over_R(R_fit, minuit_post.values["a"], minuit_post.values["b"]), label="Modello 1/R", color="red")
    ax.set_xlabel("Raggio R")
    ax.set_ylabel("Misura M")
    ax.set_title("Interpolazione di M in funzione di R con errore sulle distanze")
    ax.legend()
    plt.show()
    
    
    #ricavazione lunghezza d'onda: la distanza tra due max (*due) è lambda 
    lambda_ = []
    err_lambda = []
    for i in range(len(voltages)-1):
        lambda_.append(np.abs(r_veri[i]-r_veri[i+1])*2)
        err_lambda.append(np.sqrt(err_r_veri[i]**2+err_r_veri[i+1]**2)*2)
    
    def media_pesata(valori, errori):
    
        # Calcolo dei pesi (inverso del quadrato degli errori)
        pesi = 1 / np.array(errori)**2

        # Calcolo della media pesata
        media_pesata = np.sum(valori * pesi) / np.sum(pesi)

        # Calcolo dell'errore sulla media pesata
        errore_media_pesata = np.sqrt(1 / np.sum(pesi))

        return media_pesata, errore_media_pesata

    lambda_vera = media_pesata(lambda_, err_lambda)
    print("Lunghezza d'onda media pesata:")
    print(lambda_vera[0], lambda_vera[1])
    
    
    
    
if __name__ == "__main__":
    main()
