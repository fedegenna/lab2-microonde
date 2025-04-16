import numpy as np

def lunghezza (d, theta_max, n):
    """
    Calcola la lunghezza d'onda in funzione della distanza d, dell'angolo theta e dell'ordine n.
    """
    return 2 * d * np.sin(theta_max) / n

def lunghezza_errore (d, err_d, theta_max, err_theta_max, n):
    """
    Calcola l'errore sulla lunghezza d'onda in funzione dell'errore sulla distanza e sull'angolo.
    """
    return (2 * np.sin(theta_max) / n) * np.sqrt(err_d**2 + (d * np.cos(theta_max) * err_theta_max)**2)

def compatibilità (lunghezza_onda, lunghezza_onda_errore, lunghezza_onda_teorica):    
    """
    Calcola la compatibilità tra lunghezza d'onda misurata e teorica.
    """
    return (lunghezza_onda - lunghezza_onda_teorica) / lunghezza_onda_errore

def main ():
    theta_max = np.radians (90-39) # angolo massimo (gradi)
    err_theta_max = np.radians (1)  # errore sull'angolo (gradi)
    d = 3.8 * 10**(-2)  # distanza tra le palle in metri
    err_d = 0.1 * 10**(-2) # errore sulla distanza
    n = 2  # ordine
    lunghezza_onda = lunghezza(d, theta_max, n) 
    err_lunghezza_onda = lunghezza_errore(d, err_d, theta_max, err_theta_max, n)
    print(f"Lunghezza d'onda: {lunghezza_onda:.4f} m")
    print(f"Errore sulla lunghezza d'onda: {err_lunghezza_onda:.4f} m")
    lunghezza_onda_teorica = 2.85 * 10**(-2)  # lunghezza d'onda teorica in metri
    compat = compatibilità(lunghezza_onda, err_lunghezza_onda, lunghezza_onda_teorica)
    print(f"Compatibilità: {compat:.4f} sigma")
    
if __name__ == "__main__":
    main()