import matplotlib.pyplot as plt
import numpy as np
import mpld3

# Créer des données d'exemple
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Créer un graphique matplotlib
fig, ax = plt.subplots()
ax.plot(x, y, marker='o', linestyle='-', color='b', label='sin(x)')

# Ajouter des titres et des labels
ax.set_title('Graphique interactif avec mpld3', size=16)
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')

# Activer la légende
ax.legend()

# Convertir le graphique matplotlib en graphique interactif avec mpld3
mpld3.show()
