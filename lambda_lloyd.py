import numpy as np

def lambda_spe (c, d, n) :

    result = 2*(np.sqrt(c**2 + d**2) - c) / n
    return result

def err_lambda_spe (c, d, n, err_c, err_d) :
    result = 2/n *np.sqrt((c/np.sqrt(c**2 + d**2)-1)**2 * err_c**2 + (d/np.sqrt(c**2 + d**2))**2 * err_d**2)
    return result

def main () :
    n_list = [2, 3, 4, 5, 6, 7]
    d_list = [125, 128.5, 131.5, 134, 136.5, 139]
    for i in range(len(n_list)) :
        d_list[i] = d_list[i] - 111
    err_d = 0.5
    c, err_c = 34, 0.1

    lambda_list = []
    err_lambda_list = []

    for n,d in zip(n_list, d_list) :
        lambda_list.append(lambda_spe(c, d, n))
        err_lambda_list.append(err_lambda_spe(c, d, n, err_c, err_d))

    print("lambda_list = ", lambda_list)
    print("err_lambda_list = ", err_lambda_list)

    lambda_pesato = 0
    err_lambda_pesato = 0

    quantità = 0
    for i in range(len(lambda_list)) :
        quantità += 1/err_lambda_list[i]**2

    for i in range(len(lambda_list)) :
        lambda_pesato += lambda_list[i] / err_lambda_list[i]**2
        
    lambda_pesato /= quantità
    err_lambda_pesato = np.sqrt(1/quantità)

    print("lambda_pesato = ", lambda_pesato)
    print("err_lambda_pesato = ", err_lambda_pesato)



if __name__ == "__main__" :
    main()
