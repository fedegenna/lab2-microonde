import matplotlib.pyplot as plt
import numpy as np

def main():
    theta = [0, 1, 2, 3, 4, 5, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60, 65, 70]
    Voltage = [1.30, 1.25, 1.21, 1.07, 0.98, 0.87, 0.08, 0.07, 0.66, 0.71, 0.74, 0.80, 0.92, 0.97, 0.96, 0.89, 0.76, 0.64, 0.42, 0.07, 0.20, 0.57, 0.56, 0.55, 0.62, 0.64, 0.61, 0.60, 0.66, 0.63, 0.60, 0.56, 0.37, 0.20, 0.05]
    fig, ax = plt.subplots()
    ax.scatter(theta, Voltage, color = 'blue', label = 'Dati sperimentali')
    ax.set_xlabel(r'$\theta$ (gradi)')
    ax.set_ylabel(r'$V_{out}$ (Volt)')
    ax.set_title('Doppia fenditura')
    ax.grid()
    ax.legend()
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xticks(np.arange(0, 71, 5))
    ax.set_yticks(np.arange(-0.1, 1.6, 0.2))
    ax.set_xticklabels(np.arange(0, 71, 5), rotation=45)
    ax.set_yticklabels(np.arange(-0.1, 1.6, 0.2), rotation=45)
    ax.set_aspect('auto')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)
    
    theta_true = np.linspace(0, 75, 1000)
    V_true = np.abs(1.30*np.cos(0.5*2.20*7.2*np.sin(np.radians(theta_true)))*np.sin(0.5*2.20*np.sin(np.radians(theta_true))*1.5)*(2/(2.2*np.sin(np.radians(theta_true))*1.5)))
    ax.plot(theta_true, V_true, color = 'red', label = 'Modello teorico')
    plt.show()
    
    
if __name__ == "__main__":
    main()
