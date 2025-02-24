import numpy as np

# 1. Générer un vecteur de taille 3 de norme unitaire
v = np.random.rand(3)         # vecteur aléatoire
v = v / np.linalg.norm(v)       # normalisation pour obtenir une norme unitaire

# 2. Construire la matrice I - (v * v^T)
outer_product = np.outer(v, v)  # produit extérieur de v par lui-même
I = np.eye(3)                 # matrice identité 3x3
M = I - outer_product         # matrice opérateur de projection tangentielle

# 3. Calculer le déterminant de la matrice résultante
det = np.linalg.det(M)

# Affichage des résultats
print("Vecteur unitaire v :\n", v)
print("\nProduit extérieur (v x v^T) :\n", outer_product)
print("\nMatrice I - (v x v^T) :\n", M)
print("\nDéterminant de la matrice :", det)

# 4. Création d'un vecteur et affichage de sa projection tangentielle
x = np.random.rand(3)           # création d'un vecteur aléatoire
proj_x = M @ x                  # projection tangentielle de x sur l'espace orthogonal à v

print("\nVecteur x :\n", x)
print("\nProjection tangentielle de x (M @ x) :\n", proj_x)
