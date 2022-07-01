from sdv.timeseries import PAR
import numpy as np
import pandas as pd
import sys

filas = np.full(200, 0).tolist()
context = np.full((15, 10), 0).tolist()
companies = 0
entity_columns = ['Symbol']
context_columns = ['MarketCap', 'Sector', 'Industry']
sequence_index = 'Date'


def calcular_filas():
    global companies
    f = open("datos_reales/" + nm_model.split(".")[0]+".csv")
    num_ant = 0
    while 1:
        linea = f.readline()
        if not linea:
            filas[companies] = int(num) - int(num_ant) -1
            break
        num, cmp = [linea.split(",")[0], linea.split(",")[1]]
        try:
            if int(num) > 0:
                if cmp_ant != cmp:
                    filas[companies] = int(num) - int(num_ant) - 1
                    num_ant = num
                    companies += 1
        except ValueError:
            continue
        cmp_ant = cmp

    return cmp_ant


def calcular_contexto():
    for w in range(len(context)):
        f = open("borrar.csv", "r")
        data = pd.read_csv("datos_reales/companies.csv", index_col=[0])
        data = data.groupby('Sector').apply(list)
        data.to_csv("borrar.csv")
        j = 0
        while 1:
            linea = f.readline()
            if not linea:
                break
            context[w][j] = linea.split(",")[0]
            j += 1
        print(context[w])
        f.close()


if len(sys.argv) != 3:
    print("Asegurate de introducir todos los argumentos:\n"
          "\t> python3 generar_datos_timeseries.py [nombre del modelo sin versión] [número de versiones]")
else:
    nm_model = sys.argv[1]
    v = int(sys.argv[2])
    num_filas = calcular_filas()

    for i in range(1):
        if v == 1:
            nm_v = nm_model
        else:
            nm_v = nm_model.split(".")[0] + str(i + 1) + ".pkl"
        print(nm_v)
        model = PAR.load("modelos/" + nm_v)
        calcular_contexto()
        '''
        new_data = model.sample(num_sequences=companies, sequence_length=filas[j])

        for j in range(2):
            if j == 0:
                new_data = model.sample(num_sequences=1, sequence_length=filas[j])
            else:
                new_data2 = model.sample(num_sequences=1, sequence_length=filas[j])
                new_data = new_data.concat(new_data2, axis=0,ignore_index=True)
        new_data.to_csv("borrar.csv")
        '''
        #new_data.to_csv("datos_generados/" + nm_v.split(".")[0] + ".csv")

