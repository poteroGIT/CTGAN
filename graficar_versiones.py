import matplotlib.pyplot as plt
import numpy as np
import sys

medidas = 22
versiones = int(sys.argv[2])
vector = np.full((versiones, medidas), 0).tolist()
'''
Dimensión filas=nº de variables, columnas = nº versiones dimensión = 3 (copula,ctgan,tvae)
    variables: {e_ml,eva,m_est_cs,m_est_ks,m_det_ld,m_det_sv,medias,media} = 8
dimensión x filas x columnas
modelo x variable x valor_versión
'''
datos = np.full((3, 8, versiones), 0).tolist()


def leer_doc(doc, num):
    f = open(doc, "r")
    j = 0
    while True:
        linea = f.readline()
        if not linea or j > num:
            break
        trozos = linea.split()
        for i in range(22):
            try:
                vector[j-1][i] = float(trozos[i])
            except ValueError:
                break
        j += 1

    for i in range(len(datos[0][0])):
        w = 0
        for j in range(len(datos[0])):
            if w < 7:
                datos[0][j][i] = vector[i][3*w]
                datos[1][j][i] = vector[i][3*w+1]
                datos[2][j][i] = vector[i][3*w+2]
            else:
                datos[0][j][i] = vector[i][3*w]
            w += 1


if len(sys.argv) != 3:
    print("Asegurate de escribir todos los argumentos:\n"
          "\t> python3 graficar_versiones.py [nombre base de datos] [versiones]")

else:
    db = sys.argv[1]
    leer_doc("evaluation/"+db, versiones)

    columnas = np.array(["Modelo 1", "Modelo 2", "Modelo 3", "Modelo 4", "Modelo 5"])

    ################################
    #######Evaluación General#######
    ################################
    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Evaluación General\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][0], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][0], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][0], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_eva.png', format='png', dpi=800)

    ################################
    #######Métrica estadística######
    ################################
    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métrica estadística - CStest\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][1], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][1], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][1], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_m_est_cs.png', format='png', dpi=800)

    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métrica estadística - KStest\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][2], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][2], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][2], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_m_est_ks.png', format='png', dpi=800)

    ################################
    ###########ML Eficacy###########
    ################################
    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Machine Learning Eficacy\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][3], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][3], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][3], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_ml_ef.png', format='png', dpi=800)

    ################################
    #####Métricas de detección######
    ################################
    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métricas de detección - LogisticDetection\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][4], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][4], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][4], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_md_lg.png', format='png', dpi=800)

    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métricas de detección - SVCDetection\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][5], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][5], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][5], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_md_sv.png', format='png', dpi=800)

    ################################
    #########Valores Medios#########
    ################################
    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Valores Medios\n" + db.split(".")[0], fontdict=font1)
    plt.plot(columnas, datos[0][6], marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, datos[1][6], marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, datos[2][6], marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.plot(columnas, datos[0][7], marker='o', color='red', linestyle='dashed', label='Media total')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics_versions/" + db.split(".")[0] + '_medias.png', format='png', dpi=800)
