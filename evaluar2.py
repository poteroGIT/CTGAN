from sdv.metrics.tabular import CSTest, KSTest, LinearRegression
from sdv.metrics.tabular import LogisticDetection, SVCDetection
from sdv.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood
from sdv.metrics.tabular import BinaryMLPClassifier, MulticlassMLPClassifier, MLPRegressor, \
    MulticlassDecisionTreeClassifier
from sdv.evaluation import evaluate
import pandas as pd
import sys


dbrDiamond = sys.argv[1]
dbsDiamond = sys.argv[2]
dbrStudent = sys.argv[3]
dbsStudent = sys.argv[4]

real_data_diamond = pd.read_csv(dbrDiamond, index_col=[0])
synthetic_data_diamond = pd.read_csv(dbsDiamond, index_col=[0])
real_data_student = pd.read_csv(dbrStudent, index_col=[0])
synthetic_data_student = pd.read_csv(dbsStudent, index_col=[0])


trainSR = real_data_student.sample(int(len(real_data_student)*0.75))
trainSS = synthetic_data_student.sample(int(len(synthetic_data_student)*0.75))
testS = real_data_student[~real_data_student.index.isin(trainSR.index)]
trainDR = real_data_diamond.sample(int(len(real_data_diamond)*0.75))
trainDS = synthetic_data_diamond.sample(int(len(synthetic_data_diamond)*0.75))
testD = real_data_diamond[~real_data_diamond.index.isin(trainDR.index)]

resRL = LinearRegression.compute(testD, trainDR, target="price")
resRM = MulticlassMLPClassifier.compute(testS, trainSR, target="mba_spec")
resRT = MulticlassDecisionTreeClassifier.compute(testS, trainSR, target="mba_spec")

resSL = LinearRegression.compute(testD, trainDS, target="price")
resSM = MulticlassMLPClassifier.compute(testS, trainSR, target="mba_spec")
resST = MulticlassDecisionTreeClassifier.compute(testS, trainSR, target="mba_spec")

if resRL > resSL:
    res1 = resSL/resRL
else:
    res1 = resRL/resSL

if resRM > resSM:
    res2 = resSM/resRM
else:
    res2 = resRM/resSM

if resRT > resST:
    res3 = resST/resRT
else:
    res3 = resRT/resST


file1 = open(dbsStudent.split("/")[0] + "/evaluation/" + "grafica.txt", "a")
file1.write(str(res1) + " " + str(res2) + " " + str(res3) + "\n")
file1.close()

print(str(resRL)+" "+str(resRM)+" "+str(resRT))
print(str(resSL)+" "+str(resSM)+" "+str(resST))
print(str(res1)+" "+str(res2)+" "+str(res3))
