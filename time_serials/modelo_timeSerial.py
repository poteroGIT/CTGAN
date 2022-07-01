from sdv.timeseries import PAR
from sdv.demo import load_timeseries_demo
import pandas as pd
import time
import sys

if len(sys.argv) != 2:
    print("Indica el nombre de la base de datos que va a entrenar al modelo:\n"
          "\t> python3 modelo_timeSerial.py [nombre archivo]")
else:
    db = sys.argv[1]
    v = int(input("¿Cuántos modelos quieres generar? "))
    data = load_timeseries_demo()
    entity_columns = ['Symbol']
    context_columns = ['MarketCap', 'Sector', 'Industry']
    sequence_index = 'Date'
    model = PAR(
        entity_columns=entity_columns,
        context_columns=context_columns,
        sequence_index=sequence_index,
    )

    for i in range(v):
        t1 = time.time()
        model.fit(data)
        t2 = time.time()
        if v == 1:
            str1 = db.split(".")[0] + ".pkl"
        else:
            str1 = db.split(".")[0] + str(i + 1) + ".pkl"
        file1 = open("tiempos_modelos-" + db.split(".")[0] + ".txt", "a")
        file1.write(str1 + " " + str(t2 - t1) + "\n")
        file1.close()
        model.save("/modelos/" + str1)
