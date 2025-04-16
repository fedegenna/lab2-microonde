import numpy as np

def lunghezza (d, theta_max, n):
    """
    Calcola la lunghezza d'onda in funzione della distanza d, dell'angolo theta e dell'ordine n.
    """
    return  d * np.sin(theta_max) / n

def lunghezza_errore (d, err_d, theta_max, err_theta_max, n):
    """
    Calcola l'errore sulla lunghezza d'onda in funzione dell'errore sulla distanza e sull'angolo.
    """
    return (np.sin(theta_max) / n) * np.sqrt(err_d**2 + (d * np.cos(theta_max) * err_theta_max)**2)

def compatibilità (lunghezza_onda, lunghezza_onda_errore, lunghezza_onda_teorica):    
    """
    Calcola la compatibilità tra lunghezza d'onda misurata e teorica.
    """
    return (lunghezza_onda - lunghezza_onda_teorica) / lunghezza_onda_errore

def main ():
    theta_max_1= np.radians (25) # angolo massimo (gradi)
    theta_max_2= np.radians (55) # angolo massimo (gradi)
    err_theta_max = np.radians (1)  # errore sull'angolo (gradi)
    d = (5.7+1.5) * 10**(-2)  # distanza tra le palle in metri
    err_d = np.sqrt(2) * 0.1 * 10**(-2) # errore sulla distanza
    n_1 = 1
    n_2 = 2  # ordine

    lunghezza_onda_1 = lunghezza(d, theta_max_1, n_1) 
    err_lunghezza_onda_1 = lunghezza_errore(d, err_d, theta_max_1, err_theta_max, n_1)
    lunghezza_onda_2 = lunghezza(d, theta_max_2, n_2)
    err_lunghezza_onda_2 = lunghezza_errore(d, err_d, theta_max_2, err_theta_max, n_2)

    print(f"Lunghezza d'onda corrispondente al primo ordine: {lunghezza_onda_1:.4f} m")
    print(f"Errore sulla lunghezza d'onda corrispondente al primo ordine: {err_lunghezza_onda_1:.4f} m")
    print(f"Lunghezza d'onda corrispondente al secondo ordine: {lunghezza_onda_2:.4f} m")
    print(f"Errore sulla lunghezza d'onda corrispondente al secondo ordine: {err_lunghezza_onda_2:.4f} m")

    lunghezza_onda_media_pesata = (lunghezza_onda_1 / err_lunghezza_onda_1**2 + lunghezza_onda_2 / err_lunghezza_onda_2**2) / (1/err_lunghezza_onda_1**2 + 1/err_lunghezza_onda_2**2)
    err_lunghezza_onda_media_pesata = np.sqrt(1/(1/err_lunghezza_onda_1**2 + 1/err_lunghezza_onda_2**2))
    print(f"Lunghezza d'onda media pesata: {lunghezza_onda_media_pesata:.4f} m")
    print(f"Errore sulla lunghezza d'onda media pesata: {err_lunghezza_onda_media_pesata:.4f} m")

    lunghezza_onda_teorica = 2.85 * 10**(-2)  # lunghezza d'onda teorica in metri
    compat = compatibilità(lunghezza_onda_media_pesata, err_lunghezza_onda_media_pesata, lunghezza_onda_teorica)
    print(f"Compatibilità: {compat:.4f} sigma")
    
if __name__ == "__main__":
    main()