from sdv.metrics.tabular import CSTest, KSTest, LinearRegression
from sdv.metrics.tabular import LogisticDetection, SVCDetection
from sdv.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood
from sdv.metrics.tabular import BinaryMLPClassifier, MulticlassMLPClassifier
from sdv.evaluation import evaluate
import pandas as pd
import sys


def evaluar():
    print("\nCalculando...\n")
    res = evaluate(real_data, synthetic_data)
    print("Evaluación: " + str(res))
    file1 = open("./" + dbs.split("/")[0] + "/evaluation/" + dbr.split("/")[-1].split(".")[0] + "_evaluacion.txt",
                 "a")
    file1.write(dbr.split(".")[0].split("/")[-1] + " " + str(res) + "\n")
    file1.close()


def m_estatistica():
    print("\nCalculando...\n")
    cstest = CSTest.compute(real_data, synthetic_data)
    kstest = KSTest.compute(real_data, synthetic_data)
    print("CSTest = " + str(cstest))
    print("KSTest =" + str(kstest) + "\n")

    file1 = open("./" + dbs.split("/")[0] + "/evaluation/" + dbr.split("/")[-1].split(".")[0] + "_m_estatistica.txt",
                 "a")
    file1.write(dbr.split(".")[0].split("/")[-1] + " " + str(cstest) + " " + str(kstest) + "\n")
    file1.close()


########################################################################################
###########FALLO CON LA VERSIÓN DE NUMPY################################################
########################################################################################
def m_probabilidad():
    # fillna => NaN=0.0
    bnl = BNLikelihood.compute(real_data.fillna(0), synthetic_data.fillna(0))
    bnlog = BNLogLikelihood.compute(real_data.fillna(0), synthetic_data.fillna(0))
    gml = GMLogLikelihood.compute(real_data.fillna(0), synthetic_data.fillna(0))
    print("\nCalculando...\n")
    print("BNLikelihood = " + str(bnl))
    print("BNLogLikelihood = " + str(bnlog))
    print("BNLogLikelihood = " + str(gml) + "\n")

    file1 = open(dbs.split("/")[0] + "/evaluation/" + dbr.split("/")[-1].split(".")[0] + "_m_probabilidad.txt", "a")
    file1.write(dbr.split(".")[0].split("/")[-1] + " " + str(bnl) + " " + str(bnlog) + " " + str(gml) + "\n")
    file1.close()


def eficacia_ml():
    print(real_data.columns)
    if dbr.split("/")[-1].split(".")[0] == "bd_estudiantes":
        target = 'gender work_experience placed'
    elif dbr.split("/")[-1].split(".")[0] == "diamondL" or dbr.split("/")[-1].split(".")[0] == "diamond":
        target = ' '
    binary = 0
    if target == ' ':
        targets = target.split()
        print("\nCalculando BinaryMLPClassifier...\n")
        for i in targets:
            res = BinaryMLPClassifier.compute(synthetic_data, real_data, target=i)
            print("Para " + i + " : " + str(res))
            binary = binary + res
        try:
            media_bin = str(binary / (len(targets)))
        except ZeroDivisionError:
            media_bin = str(0)
        print("Media BinaryMLPClassifier = " + media_bin)

    print(real_data.columns)
    target = ' '
    if dbr.split("/")[-1].split(".")[0] == "bd_estudiantes":
        target = 'high_spec degree_type mba_spec experience_years duration'
    elif dbr.split("/")[-1].split(".")[0] == "diamondL" or dbr.split("/")[-1].split(".")[0] == "diamond":
        target = 'cut color clarity'

    targets = target.split()
    multiclass = 0
    print("\nCalculando MulticlassMLPClassifier...\n")
    for i in targets:
        res = MulticlassMLPClassifier.compute(synthetic_data.fillna(0), real_data.fillna(0), target=i)
        # res2 = LinearRegression.compute(synthetic_data.fillna(0), real_data.fillna(0), target=i)
        print("Para " + i + " : " + str(res))
        # print("Para linear" + i + " : " + str(res2))
        multiclass = multiclass + res
    media_mult = str(multiclass / (len(targets)))
    print("Media  MulticlassMLPClassifier = " + media_mult)

    print(real_data.columns)
    target = ' '
    if dbr.split("/")[-1].split(".")[0] == "bd_estudiantes":
        target = 'second_perc high_perc degree_perc employability_perc mba_perc salary'
    elif dbr.split("/")[-1].split(".")[0] == "diamondL" or dbr.split("/")[-1].split(".")[0] == "diamond":
        target = 'carat depth table price x y z'

    targets = target.split()
    regresor = 0
    print("\nCalculando MLPRegressor...\n")
    for i in targets:
        res = LinearRegression.compute(synthetic_data.fillna(0), real_data.fillna(0), target=i)
        print("Para " + i + " : " + str(res))
        regresor = regresor + res
    media_reg = str(regresor / (len(targets)))
    print("Media  MLPRegressor = " + media_reg)

    file1 = open(dbs.split("/")[0] + "/evaluation/" + dbr.split("/")[-1].split(".")[0] + "_eficacia_ml.txt", "a")
    file1.write(dbr.split(".")[0].split("/")[-1] + " " + str(media_bin) + " " + str(media_mult) + " " + str(media_reg) +
                "\n")
    file1.close()


def m_deteccion():
    print("\nCalculando...\n")
    ld = LogisticDetection.compute(real_data, synthetic_data)
    svcd = SVCDetection.compute(real_data, synthetic_data)
    print("(Cuanto más próximo a 0 mejor)")
    print("LogisticDetection = " + str(ld))
    print("SVCDetection = " + str(svcd))
    file1 = open(dbs.split("/")[0] + "/evaluation/" + dbr.split("/")[-1].split(".")[0] + "_m_deteccion.txt", "a")
    file1.write(dbr.split(".")[0].split("/")[-1] + " " + str(ld) + " " + str(svcd) + "\n")
    file1.close()


if len(sys.argv) != 3:
    print("Indica las rutas de los archivos a evaluar:\n"
          "\t> python3 evaluar_old.py [ruta datos reales] [ruta datos generados]")
else:
    dbr = sys.argv[1]
    dbs = sys.argv[2]
    real_data = pd.read_csv(dbr, index_col=[0])
    synthetic_data = pd.read_csv(dbs, index_col=[0])
    evaluar()
    m_estatistica()
    eficacia_ml()
    m_deteccion()
    metadata = {'fields': {'start_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
                           'end_date': {'type': 'datetime', 'format': '%Y-%m-%d'},
                           'salary': {'type': 'numerical', 'subtype': 'integer'},
                           'duration': {'type': 'categorical'},
                           'student_id': {'type': 'id', 'subtype': 'integer'},
                           'high_perc': {'type': 'numerical', 'subtype': 'float'},
                           'high_spec': {'type': 'categorical'},
                           'mba_spec': {'type': 'categorical'},
                           'second_perc': {'type': 'numerical', 'subtype': 'float'},
                           'gender': {'type': 'categorical'},
                           'degree_perc': {'type': 'numerical', 'subtype': 'float'},
                           'placed': {'type': 'boolean'},
                           'experience_years': {'type': 'numerical', 'subtype': 'float'},
                           'employability_perc': {'type': 'numerical', 'subtype': 'float'},
                           'mba_perc': {'type': 'numerical', 'subtype': 'float'},
                           'work_experience': {'type': 'boolean'},
                           'degree_type': {'type': 'categorical'}},
                'target': [],
                'constraints': [],
                'model_kwargs': {},
                'name': None,
                'primary_key': 'student_id',
                'sequence_index': None,
                'entity_columns': [],
                'context_columns': []}
    '''
    while 1:
        print("¿Qué evaluation quieres utilizar?"
              "\n\t1 - Métricas estatísticas."
              "\n\t2 - Métricas de probabilidad(falla)."
              "\n\t3 - Eficacia de machine learning."
              "\n\t4 - Métricas de detección."
              "\n\t5 - Salir.")
        ans = input("Respuesta (1-5): ")
        ans = int(ans)

        if ans == 1:
            m_estatistica()
        elif ans == 2:
            m_probabilidad()
        elif ans == 3:
            eficacia_ml()
        elif ans == 4:
            m_deteccion()
        else:
            break
    '''
