import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(0.1,10,500)
error_train = 1/X
error_test = 0.1*X+1/X

extrem = 3.1622
plt.figure(figsize=(10, 6))
plt.plot(X, error_train, label="Error d entrenament", color="blue", linestyle='--',linewidth=2)
plt.plot(X, error_test, label="Error de prova", color="green", linewidth=2)

# Línea vertical roja
plt.axvline(extrem, color='red', linewidth=2)
x_flecha = 7
y1 = 1 / x_flecha
y2 = 0.1*x_flecha+1/x_flecha

x_tail = x_flecha
y_tail = y1
x_head = x_flecha
y_head = y2
dx = x_head - x_tail
dy = y_head - y_tail


plt.annotate("", xytext=(x_tail, y2), xy=(x_tail, y_tail),
            arrowprops=dict(arrowstyle="<->"))
# Texto explicativo
plt.text(0.8, 9, 'Zona de subajustament', fontsize=10, color='black')
plt.text(3.4, 9, 'Zona de sobreajustament', fontsize=10, color='black')
plt.text(extrem, -0.2, "Cap. òptima", fontsize=9, color='grey' )
plt.text(x_tail+0.1,y1+dy/2, "Dist. errors", fontsize=8, color='grey')
# Ajustes de la gráfica
plt.xlabel('Capacitat del model')
plt.ylabel('Error')
plt.title('Sobreajustament i Subajajustament en funció de la capacitat del model')
plt.legend()
plt.grid(False)
plt.tight_layout()

# Mostrar
plt.show()
