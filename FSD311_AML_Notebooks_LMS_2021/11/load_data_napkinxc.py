import numpy as np

from napkinxc.datasets import load_dataset



X_train_napxinxc, Y_train_napxinxc = load_dataset("eurlex-4k", "train")
X_train_arr_tot = X_train_napxinxc.toarray() # Les données de la librairie ont un format un peu particulier. On les remets sous forme matricielle python classique pour ce notebook
X_test_napxinxc, Y_test_napxinxc = load_dataset("eurlex-4k", "test")
X_test_arr_tot = X_test_napxinxc.toarray() # Les données de la librairie ont un format un peu particulier. On les remets sous forme matricielle python classique pour ce notebook

def transform_y_shape(y,nb_columns):
    new_y = np.zeros(shape=(len(y),nb_columns))
    for i in range(len(y)):
        for c in y[i]:
            new_y[i][int(c)] = 1
    return new_y

Y_train_arr_tot = transform_y_shape(Y_train_napxinxc,3993)
Y_test_arr_tot = transform_y_shape(Y_test_napxinxc,3993)


#####  Code original notebook ###
# X_train, Y_train = load_dataset("eurlex-4k", "train")
# X_train_arr_tot = X_train.toarray() # Les données de la librairie ont un format un peu particulier. On les remets sous forme matricielle python classique pour ce notebook
# X_test, Y_test = load_dataset("eurlex-4k", "test")
# X_test_arr_tot = X_test.toarray() # Les données de la librairie ont un format un peu particulier. On les remets sous forme matricielle python classique pour ce notebook

# print('Shape X_train : ', (X_train_arr_tot).shape)
# print(X_train_arr_tot)

# # X_train contient donc 15539 instance avec chacunes 5000 features qui sont des floats.

# ################### Y
# # Y_train contient pour chaque instance les différents labels lui correspondant,
# # caractérisés par des entiers. Cherchons à voir le nombre de labels différents dans le dataset.

# print('Len Y_train : ', len(Y_train))
# print('Y_train item example : ', Y_train[0])


# l = []
# for i in range(len(Y_train)): # Pour chaque instance dans Y
#     for el in (list(Y_train[i])): # Pour chacun de ses labels
#         if el not in l : # Si on ne l'a pas encore répertorié, on l'ajoute à notre liste l
#             l.append(el)
# print("Label mininum : ", min(l))
# print("Label maximum : ", max(l))
# print("Taille de la liste l : ", len(l))
# print("Nombre de labels correspondant à aucune instance : ", max(l)-len(l))


# On voit comme l'indique le nom du dataset l'indique (eurlex-4k) qu'il y a environ 4000 labels (3993) dans ce dataset. Cetains labels (206) ne correspondent à aucunes instance.

# # Pour pouvoir utiliser les données de Y pour les classifieurs que nous allons utiliser après,
# # nous allons créer les données Y_train_arr et Y_test_arr qui sont des matrices avec n lignes et 3993 colonnes 
# # (n étant le nombre d'instances présentes dans Y_train ou Y_test).
# # L'élément i,j de ces matrices sera de 1 si l'instance i est labélisée j, et 0 sinon.

# def transform_y_shape(y,nb_columns):
#     new_y = np.zeros(shape=(len(y),nb_columns))
#     for i in range(len(y)):
#         for c in y[i]:
#             new_y[i][int(c)] = 1
#     return new_y
# Y_train_arr_tot = transform_y_shape(Y_train,3993)
# Y_test_arr_tot = transform_y_shape(Y_test,3993)
# print("Shape Y_train_arr : ", Y_train_arr_tot.shape)
# print(Y_train_arr_tot)
# print("\nValeur du label 446 de l'instance 0 : ", Y_train_arr_tot[0][446])