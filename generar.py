from sdv.tabular import CopulaGAN, CTGAN, TVAE
import sys

if len(sys.argv) != 4:
    print("Asegurate de introducir todos los argumentos:\n"
          "\t> python3 generar.py [nombre del modelo sin versión] [número de filas a generar] [número de versiones]")
else:
    nm_model = sys.argv[1]
    num_rows = int(sys.argv[2])
    v = int(sys.argv[3])
    print(str(len(nm_model.split("_",1))))
    for i in range(v):
        if v == 1:
            nm_v = nm_model
        else:
            nm_v = nm_model.split("_")[0]+str(i+1)+"_"+nm_model.split("_", 1)[1]
        print(nm_v)
        model = CopulaGAN.load("copulaGan/modelos/"+nm_v)
        new_data = model.sample(num_rows=num_rows)
        new_data.to_csv("copulaGan/datos_generados/" + nm_v.split(".")[0] + "_gen" + str(num_rows) + ".csv")

        model = CTGAN.load("ctgan/modelos/"+nm_v)
        new_data = model.sample(num_rows=num_rows)
        new_data.to_csv("ctgan/datos_generados/" + nm_v.split(".")[0]+"_gen" + str(num_rows) + ".csv")

        model = TVAE.load("tvae/modelos/"+nm_v)
        new_data = model.sample(num_rows=num_rows)
        new_data.to_csv("tvae/datos_generados/" + nm_v.split(".")[0]+ "_gen" + str(num_rows) + ".csv")