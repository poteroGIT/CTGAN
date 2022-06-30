import matplotlib.pyplot as plt
import numpy as np
import sys

vector = [[0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0]]

datos = [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]

columnas = np.array(["100-100", "300-300", "500-500", "700-700", "900-900", "1100-1100"])


def leer_doc(doc, num):
    f = open(doc, "r")
    j = 0
    while True:
        linea = f.readline()
        if not linea:
            break
        trozos = linea.split()
        for i in range(num):
            vector[i][j] = float(trozos[i])
        j += 1


num = 3
leer_doc("copulaGan/evaluation/grafica.txt", num)
linearR_co = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
multiClass_co = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])
multiTree_co = np.array([vector[2][0], vector[2][1], vector[2][2], vector[2][3], vector[2][4], vector[2][5]])

num = 3
leer_doc("ctgan/evaluation/grafica.txt", num)
linearR_ct = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
multiClass_ct = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])
multiTree_ct = np.array([vector[2][0], vector[2][1], vector[2][2], vector[2][3], vector[2][4], vector[2][5]])

num = 3
leer_doc("tvae/evaluation/grafica.txt", num)
linearR_tv = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
multiClass_tv = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])
multiTree_tv = np.array([vector[2][0], vector[2][1], vector[2][2], vector[2][3], vector[2][4], vector[2][5]])

fig1 = plt.figure(figsize=(8, 8))
font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
plt.xlabel("\nepochs - batch-size", fontdict=font2)
plt.title("Price - linear regression\nDiamond", fontdict=font1)
plt.plot(columnas, linearR_co, marker='o', color='hotpink', linestyle='dashed', label="CopulaGAN")
plt.plot(columnas, linearR_ct, marker='o', color='blue', linestyle='dashed', label="CTGAN")
plt.plot(columnas, linearR_tv, marker='o', color='red', linestyle='dashed', label="TVAE")
plt.legend(loc="upper left", prop={"size": 10})
plt.savefig("graphics/linearRegression.png", format='png', dpi=800)

fig2 = plt.figure(figsize=(8, 8))
font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
plt.xlabel("\nepochs - batch-size", fontdict=font2)
plt.title("mba_spec - MultiClass\nStudents", fontdict=font1)
plt.plot(columnas, multiClass_co, marker='o', color='hotpink', linestyle='dashed', label="CopulaGAN")
plt.plot(columnas, multiClass_ct, marker='o', color='blue', linestyle='dashed', label="CTGAN")
plt.plot(columnas, multiClass_tv, marker='o', color='red', linestyle='dashed', label="TVAE")
plt.legend(loc="upper left", prop={"size": 10})
plt.savefig("graphics/multiClass.png", format='png', dpi=800)

fig3 = plt.figure(figsize=(8, 8))
font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
plt.xlabel("\nepochs - batch-size", fontdict=font2)
plt.title("mba_spec - MultiTree\nStudents", fontdict=font1)
plt.plot(columnas, multiTree_co, marker='o', color='hotpink', linestyle='dashed', label="CopulaGAN")
plt.plot(columnas, multiTree_ct, marker='o', color='blue', linestyle='dashed', label="CTGAN")
plt.plot(columnas, multiTree_tv, marker='o', color='red', linestyle='dashed', label="TVAE")
plt.legend(loc="upper left", prop={"size": 10})
plt.savefig("graphics/multiTree.png", format='png', dpi=800)
