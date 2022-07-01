from sdv.metrics.timeseries import LSTMDetection, TSFCDetection
from sdv.metrics.timeseries import TSFClassifierEfficacy, LSTMClassifierEfficacy
import sys
import pandas as pd


def m_deteccion():
    ls = LSTMDetection.compute(real_data, synthetic_data, metadata)
    ts = TSFCDetection.compute(real_data, synthetic_data, metadata)

    return [ls, ts]


def eficacia_ml():
    metadata['target'] = target
    ts = TSFClassifierEfficacy.compute(real_data, synthetic_data, metadata, target=target)
    ls = LSTMClassifierEfficacy.compute(real_data, synthetic_data, metadata, target=target)
    return [ts, ls]


metadata = {'fields': {'Symbol': {'type': 'categorical'},
                       'Date': {'type': 'datetime'},
                       'Open': {'type': 'numerical', 'subtype': 'float'},
                       'Close': {'type': 'numerical', 'subtype': 'float'},
                       'Volume': {'type': 'numerical', 'subtype': 'float'},
                       'MarketCap': {'type': 'numerical', 'subtype': 'integer'},
                       'Sector': {'type': 'categorical'},
                       'Industry': {'type': 'categorical'}},
            'entity_columns': ['Symbol'],
            'sequence_index': 'Date',
            'context_columns': ['MarketCap', 'Sector', 'Industry'],
            'target': 'Symbol'
            }

if len(sys.argv) != 5:
    print("Indica las rutas de los archivos a evaluar (Si solo hay una versión => 0):\n"
          "\t> python3 evaluar_timeseries.py [nombre base de datos] [nombre datos generados] [versiones] [target]")
else:
    dbr = sys.argv[1]
    dbs = sys.argv[2]
    v = int(sys.argv[3])
    target = sys.argv[4]
    real_data = pd.read_csv("datos_reales/" + dbr, index_col=[0])
    file1 = open("evaluation/" + dbs.split(".")[0] + "_evaluation.txt", "a")
    file1.write(",media\n")

    for i in range(v):
        if v == 1:
            name = dbs
        else:
            name = dbs.split(".")[0] + str(i + 1) + ".csv"
        synthetic_data = pd.read_csv("datos_generados/" + name, index_col=[0])
        e_ml = eficacia_ml()
        m_d = m_deteccion()
        media = (e_ml[0] + e_ml[1] + m_d[0] + m_d[1]) / 4
        file1.write(str(e_ml[0]) + " " + str(e_ml[1]) + " " + str(m_d[0]) + " " + str(m_d[1]) + "\n")

    file1.close()
