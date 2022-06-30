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
            vector[i][j] = float(trozos[i + 1])
        j += 1


if len(sys.argv) != 2:
    print("Asegurate de escribir todos los argumentos:\n"
          "\t> python3 graficar.py [nombre base de datos]")

else:
    db = sys.argv[1]

    ###################################################################################################################
    ########################################## - MACHINE LEARNING EFFICACY - ##########################################
    ###################################################################################################################
    num = 3
    leer_doc("copulaGan/evaluation/" + db + "_eficacia_ml.txt", num)
    ef_bin_cop = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    ef_mul_cop = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])
    ef_con_cop = np.array([vector[2][0], vector[2][1], vector[2][2], vector[2][3], vector[2][4], vector[2][5]])

    leer_doc("ctgan/evaluation/" + db + "_eficacia_ml.txt", num)
    ef_bin_ctg = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    ef_mul_ctg = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])
    ef_con_ctg = np.array([vector[2][0], vector[2][1], vector[2][2], vector[2][3], vector[2][4], vector[2][5]])

    leer_doc("tvae/evaluation/" + db + "_eficacia_ml.txt", num)
    ef_bin_tva = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    ef_mul_tva = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])
    ef_con_tva = np.array([vector[2][0], vector[2][1], vector[2][2], vector[2][3], vector[2][4], vector[2][5]])

    fig1 = plt.figure(figsize=(8, 8))
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 18}
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Machine Learning Efficacy - Binary\n"+db, fontdict=font1)
    plt.plot(columnas, ef_bin_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, ef_bin_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, ef_bin_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_eficacia_ml_binaria.png', format='png', dpi=800)

    fig2 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Machine Learning Efficacy - Multiclass\n" + db, fontdict=font1)
    plt.plot(columnas, ef_mul_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, ef_mul_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, ef_mul_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_eficacia_ml_multiclass.png', format='png', dpi=800)

    fig3 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Machine Learning Efficacy - Continuous\n" + db, fontdict=font1)
    plt.plot(columnas, ef_con_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, ef_con_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, ef_con_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_eficacia_ml_continuous.png', format='png', dpi=800)

    ###################################################################################################################
    ###################################################################################################################

    ###################################################################################################################
    ################################################# - EVALUACION - ##################################################
    ###################################################################################################################
    num = 1
    leer_doc("copulaGan/evaluation/" + db + "_evaluacion.txt", num)
    eva_cop = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])

    leer_doc("ctgan/evaluation/" + db + "_evaluacion.txt", num)
    eva_ctg = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])

    leer_doc("tvae/evaluation/" + db + "_evaluacion.txt", num)
    eva_tva = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])

    fig4 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Evaluación - " + db, fontdict=font1)
    plt.plot(columnas, eva_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, eva_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, eva_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_evaluacion.png', format='png', dpi=800)

    ###################################################################################################################
    ###################################################################################################################

    ###################################################################################################################
    ############################################ - METRICAS DE DETECCIÓN - ############################################
    ###################################################################################################################
    num = 2
    leer_doc("copulaGan/evaluation/" + db + "_m_deteccion.txt", num)
    det_ld_cop = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    det_sv_cop = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])

    leer_doc("ctgan/evaluation/" + db + "_m_deteccion.txt", num)
    det_ld_ctg = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    det_sv_ctg = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])


    leer_doc("tvae/evaluation/" + db + "_m_deteccion.txt", num)
    det_ld_tva = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    det_sv_tva = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])

    fig5 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métrica de Detección - LogisticDetection\n" + db, fontdict=font1)
    plt.plot(columnas, det_ld_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, det_ld_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, det_ld_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_m_deteccion_ld.png', format='png', dpi=800)

    fig5 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métrica de Detección - SVCDetection\n" + db, fontdict=font1)
    plt.plot(columnas, det_sv_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, det_sv_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, det_sv_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_m_deteccion_sv.png', format='png', dpi=800)
    ###################################################################################################################
    ###################################################################################################################

    ###################################################################################################################
    ############################################ - METRICAS ESTATÍSTICAS - ############################################
    ###################################################################################################################
    num = 2
    leer_doc("copulaGan/evaluation/" + db + "_m_estatistica.txt", num)
    est_cs_cop = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    est_ks_cop = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])

    leer_doc("ctgan/evaluation/" + db + "_m_estatistica.txt", num)
    est_cs_ctg = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    est_ks_ctg = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])

    leer_doc("tvae/evaluation/" + db + "_m_estatistica.txt", num)
    est_cs_tva = np.array([vector[0][0], vector[0][1], vector[0][2], vector[0][3], vector[0][4], vector[0][5]])
    est_ks_tva = np.array([vector[1][0], vector[1][1], vector[1][2], vector[1][3], vector[1][4], vector[1][5]])

    fig5 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métrica Estadísticas - CSTest\n" + db, fontdict=font1)
    plt.plot(columnas, est_cs_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, est_cs_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, est_cs_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_m_estatistica_cs.png', format='png', dpi=800)

    fig5 = plt.figure(figsize=(8, 8))
    plt.xlabel("\nepochs - batch-size", fontdict=font2)
    plt.title("Métrica Estadísticas - KSTest\n" + db, fontdict=font1)
    plt.plot(columnas, est_ks_cop, marker='o', color='hotpink', linestyle='dashed', label='copulaGAN')
    plt.plot(columnas, est_ks_ctg, marker='o', color='blue', linestyle='dashed', label='CTGAN')
    plt.plot(columnas, est_ks_tva, marker='o', color='red', linestyle='dashed', label='TVAE')
    plt.legend(loc="upper left", prop={"size": 10})
    plt.savefig("graphics/" + db + '_m_estatistica_ks.png', format='png', dpi=800)
    ###################################################################################################################
    ###################################################################################################################