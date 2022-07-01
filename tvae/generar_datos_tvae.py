from sdv.tabular import CopulaGAN
import sys

if len(sys.argv) != 3:
    print("Indica el nombre del modelo que se va a utilizar:\n"
          "\t> python3 generar_datos_tvae.py [ruta modelo] [número de filas a generar]")
else:
    nm_model = sys.argv[1]
    num_rows = int(sys.argv[2])
    model = CopulaGAN.load(nm_model)
    new_data = model.sample(num_rows=num_rows)
    new_data.to_csv("./datos_generados/" + nm_model.split(".")[0].split("/")[1] + "_gen" + str(num_rows) + ".csv")
