from sdv.tabular import CopulaGAN
import pandas as pd
import sys
import time

if len(sys.argv) != 4:
    print("Indica el nombre de la base de datos que va a entrenar al modelo:\n"
          "\t> python3 modelo_tvae.py [ruta modelo] [epochs] [batch size]")
else:
    db = sys.argv[1]
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    v = int(input("¿Cuántos modelos quieres generar? "))
    data = pd.read_csv(db, index_col=[0])
    model = CopulaGAN(epochs=epochs, batch_size=batch_size)
    for i in range(v):
        t1 = time.time()
        model.fit(data)
        t2 = time.time()
        if v == 1:
            name = db.split(".")[0].split("/")[1]
        else:
            name = db.split(".")[0].split("/")[1]+str(i+1)
        str1 = name + "_" + str(epochs) + "_" + str(batch_size) + ".pkl"
        file1 = open("tiempos_modelos-"+db.split(".")[0].split("/")[1]+".txt", "a")
        file1.write(str1 + " " + str(t2-t1) + "\n")
        file1.close()
        model.save("./modelos/" + str1)
