from sdv.metrics.tabular import CSTest, KSTest, LinearRegression
from sdv.metrics.tabular import LogisticDetection, SVCDetection
from sdv.metrics.tabular import BinaryMLPClassifier, MulticlassMLPClassifier
from sdv.evaluation import evaluate
import pandas as pd
import sys

#Fichero de evaluación

def eficacia_ml():
    train_real = real_data.sample(int(len(real_data) * 0.75))
    train_copula = copula_data.sample(int(len(copula_data) * 0.75))
    train_ctgan = ctgan_data.sample(int(len(ctgan_data) * 0.75))
    train_tvae = tvae_data.sample(int(len(tvae_data) * 0.75))
    test = real_data[~real_data.index.isin(train_real.index)]

    if tipo == "b":
        r = BinaryMLPClassifier.compute(test, train_real, target=target)
        g_co = BinaryMLPClassifier.compute(test, train_copula, target=target)
        g_ct = BinaryMLPClassifier.compute(test, train_ctgan, target=target)
        g_tv = BinaryMLPClassifier.compute(test, train_tvae, target=target)
    elif tipo == "k":
        r = MulticlassMLPClassifier.compute(test, train_real, target=target)
        g_co = MulticlassMLPClassifier.compute(test, train_copula, target=target)
        g_ct = MulticlassMLPClassifier.compute(test, train_ctgan, target=target)
        g_tv = MulticlassMLPClassifier.compute(test, train_tvae, target=target)
    else:
        r = LinearRegression.compute(test, train_real, target=target)
        g_co = LinearRegression.compute(test, train_copula, target=target)
        g_ct = LinearRegression.compute(test, train_ctgan, target=target)
        g_tv = LinearRegression.compute(test, train_tvae, target=target)

    return [min(r, g_co) / max(r, g_co), min(r, g_ct) / max(r, g_ct), min(r, g_tv) / max(r, g_tv)]


def evaluar():
    return [evaluate(copula_data, real_data), evaluate(ctgan_data, real_data), evaluate(tvae_data, real_data)]


def m_estatistica():
    cstest = [CSTest.compute(real_data, copula_data), CSTest.compute(real_data, ctgan_data),
              CSTest.compute(real_data, tvae_data)]
    kstest = [KSTest.compute(real_data, copula_data), KSTest.compute(real_data, ctgan_data),
              KSTest.compute(real_data, tvae_data)]

    return cstest + kstest


def m_deteccion():
    ld = [1 - LogisticDetection.compute(real_data, copula_data), 1 - LogisticDetection.compute(real_data, ctgan_data),
          1 - LogisticDetection.compute(real_data, tvae_data)]
    svcd = [1 - SVCDetection.compute(real_data, copula_data), 1 - SVCDetection.compute(real_data, ctgan_data),
            1 - SVCDetection.compute(real_data, tvae_data)]

    return ld + svcd


if len(sys.argv) != 6:
    print("Indica las rutas de los archivos a evaluar (Si solo hay una versión => 0):\n"
          "\t> python3 evaluar_old.py [nombre base de datos] [nombre datos generados] [versiones] [target] "
          "[tipo de target]"
          "\nTipo de target:      Binario: b\n"
          "                     Categórico: k\n"
          "                     Continuo: c")
else:
    dbr = sys.argv[1]
    dbs = sys.argv[2]
    v = int(sys.argv[3])
    target = sys.argv[4]
    tipo = sys.argv[5]
    real_data = pd.read_csv("copulaGan/datos_reales/" + dbr, index_col=[0])
    file1 = open("evaluation/" + dbs.split(".")[0] + "_evaluation.txt", "a")
    file1.write("eva_copula,eva_ctgan,eva_tvae,m_est_cs_co,m_est_cs_ct,m_est_cs_tv,m_est_ks_co,m_est_ks_ct,"
                "m_est_ks_tv,e_ml_co,e_ml_ct,e_ml_tv,m_d_ld_co,m_d_ld_ct,m_d_ld_tv,m_d_sv_co,m_d_sv_ct,m_d_sv_tv,"
                "media_co,media_ct,media_tv,media\n")
    for i in range(v):
        if v == 1:
            name = dbs
        else:
            name = dbs.split("_")[0] + str(i + 1) + "_" + dbs.split("_", 1)[1]
        copula_data = pd.read_csv("copulaGan/datos_generados/" + name, index_col=[0])
        ctgan_data = pd.read_csv("ctgan/datos_generados/" + name, index_col=[0])
        tvae_data = pd.read_csv("tvae/datos_generados/" + name, index_col=[0])
        eva = evaluar()
        m_e = m_estatistica()
        e_ml = eficacia_ml()
        m_d = m_deteccion()
        media_co = eva[0]+m_e[0]+m_e[3]+m_d[0]+m_d[3]+e_ml[0]
        media_co = media_co/6
        media_ct = eva[1]+m_e[1]+m_e[4]+m_d[1]+m_d[4]+e_ml[1]
        media_ct = media_ct/6
        media_tv = eva[2]+m_e[2]+m_e[5]+m_d[2]+m_d[5]+e_ml[2]
        media_tv = media_tv/6
        media = (media_ct + media_co + media_tv)/3
        file1.write(str(eva[0])+" "+str(eva[1])+" "+str(eva[2])+" "+str(m_e[0])+" "+str(m_e[1])+" "+str(m_e[2])+" "+
                    str(m_e[3])+" "+str(m_e[4])+" "+str(m_e[5])+" "+str(e_ml[0])+" "+str(e_ml[1])+" "+str(e_ml[2])+" "+
                    str(m_d[0])+" "+str(m_d[1])+" "+str(m_d[2])+" "+str(m_d[3])+" "+str(m_d[4])+" "+str(m_d[5])+" "+
                    str(media_co)+" "+str(media_ct)+" "+str(media_tv)+" "+str(media)+"\n")
    file1.close()
