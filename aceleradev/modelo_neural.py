import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from keras import regularizers
from tensorflow.python.keras import layers

base_total = pd.read_csv("/content/drive/My Drive/train.csv", header=0)


base_test = pd.read_csv("/content/drive/My Drive/test.csv", header=0)

base_selected_features = base_total.loc[:, ["NU_INSCRICAO","CO_UF_RESIDENCIA","SG_UF_RESIDENCIA","NU_IDADE","TP_SEXO","TP_COR_RACA","TP_NACIONALIDADE","TP_ST_CONCLUSAO","TP_ANO_CONCLUIU","TP_ESCOLA","TP_ENSINO","IN_TREINEIRO","TP_DEPENDENCIA_ADM_ESC","IN_BAIXA_VISAO","IN_CEGUEIRA","IN_SURDEZ","IN_DISLEXIA","IN_DISCALCULIA","IN_SABATISTA","IN_GESTANTE","IN_IDOSO","TP_PRESENCA_CN","TP_PRESENCA_CH","TP_PRESENCA_LC","CO_PROVA_CN","CO_PROVA_CH","CO_PROVA_LC","CO_PROVA_MT","NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_LC","TP_LINGUA","TP_STATUS_REDACAO","NU_NOTA_COMP1","NU_NOTA_COMP2","NU_NOTA_COMP3","NU_NOTA_COMP4","NU_NOTA_COMP5","NU_NOTA_REDACAO","Q001","Q002","Q006","Q024","Q025","Q026","Q027","Q047","NU_NOTA_MT"]]

base_selected_features = base_selected_features.fillna(999)


test_selected_features = base_test.loc[:, ["NU_INSCRICAO","CO_UF_RESIDENCIA","SG_UF_RESIDENCIA","NU_IDADE","TP_SEXO","TP_COR_RACA","TP_NACIONALIDADE","TP_ST_CONCLUSAO","TP_ANO_CONCLUIU","TP_ESCOLA","TP_ENSINO","IN_TREINEIRO","TP_DEPENDENCIA_ADM_ESC","IN_BAIXA_VISAO","IN_CEGUEIRA","IN_SURDEZ","IN_DISLEXIA","IN_DISCALCULIA","IN_SABATISTA","IN_GESTANTE","IN_IDOSO","TP_PRESENCA_CN","TP_PRESENCA_CH","TP_PRESENCA_LC","CO_PROVA_CN","CO_PROVA_CH","CO_PROVA_LC","CO_PROVA_MT","NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_LC","TP_LINGUA","TP_STATUS_REDACAO","NU_NOTA_COMP1","NU_NOTA_COMP2","NU_NOTA_COMP3","NU_NOTA_COMP4","NU_NOTA_COMP5","NU_NOTA_REDACAO","Q001","Q002","Q006","Q024","Q025","Q026","Q027","Q047"]]


test_selected_features = test_selected_features.fillna(999)



index_test = np.array(test_selected_features["NU_INSCRICAO"])



a = np.array(base_selected_features.CO_UF_RESIDENCIA.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.CO_UF_RESIDENCIA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a = np.array(base_selected_features.SG_UF_RESIDENCIA.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.SG_UF_RESIDENCIA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a = np.array(base_selected_features.TP_SEXO.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.TP_SEXO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a == np.array(base_selected_features.TP_COR_RACA.drop_duplicates().sort_values(ascending=True))
b == np.array(test_selected_features.TP_COR_RACA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a = np.array(base_selected_features.TP_SEXO.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.TP_SEXO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a == np.array(base_selected_features.TP_COR_RACA.drop_duplicates().sort_values(ascending=True))
b == np.array(test_selected_features.TP_COR_RACA.drop_duplicates().sort_values(ascending=True))

print(a == b)
print("\n")

a = np.array(base_selected_features.TP_NACIONALIDADE.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.TP_NACIONALIDADE.drop_duplicates().sort_values(ascending=True))

print(a == b)
print("\n")

a = np.array(base_selected_features.TP_ST_CONCLUSAO.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.TP_ST_CONCLUSAO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a = np.array(base_selected_features.TP_ANO_CONCLUIU.drop_duplicates().sort_values(ascending=True))
b = np.array(test_selected_features.TP_ANO_CONCLUIU.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a = np.array(base_selected_features.TP_ESCOLA.drop_duplicates().sort_values(ascending=True))
b = np.array(base_selected_features.TP_ESCOLA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_ENSINO.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_ENSINO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_TREINEIRO.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_TREINEIRO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_DEPENDENCIA_ADM_ESC.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_DEPENDENCIA_ADM_ESC.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_BAIXA_VISAO.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_BAIXA_VISAO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_CEGUEIRA.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_CEGUEIRA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_SURDEZ.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_SURDEZ.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_DISLEXIA.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_DISLEXIA.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_DISCALCULIA.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_DISCALCULIA.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_SABATISTA.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_SABATISTA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_GESTANTE.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_GESTANTE.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.IN_IDOSO.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.IN_IDOSO.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_PRESENCA_CN.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_PRESENCA_CN.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_PRESENCA_CH.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_PRESENCA_CH.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_PRESENCA_LC.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_PRESENCA_LC.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.CO_PROVA_CN.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.CO_PROVA_CN.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.CO_PROVA_CH.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.CO_PROVA_CH.drop_duplicates().sort_values(ascending=True))
print(a)
print(b)
print(a == b)
print("\n")

a  = np.array(base_selected_features.CO_PROVA_LC.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.CO_PROVA_LC.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.CO_PROVA_MT.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.CO_PROVA_MT.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_LINGUA.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_LINGUA.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.TP_STATUS_REDACAO.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.TP_STATUS_REDACAO.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.Q001.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q001.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.Q002.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q002.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.Q006.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q006.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.Q024.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q024.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

a  = np.array(base_selected_features.Q025.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q025.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")


a  = np.array(base_selected_features.Q026.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q026.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("\n")

#a  = np.array(base_selected_features.Q027.drop_duplicates().sort_values(ascending=True))
#b  = np.array(test_selected_features.Q027.drop_duplicates().sort_values(ascending=True))

a  = base_selected_features.Q027.drop_duplicates()
b  = test_selected_features.Q027.drop_duplicates()



print(a)
print(b)
#print(a == b)
print("\n")

a  = np.array(base_selected_features.Q047.drop_duplicates().sort_values(ascending=True))
b  = np.array(test_selected_features.Q047.drop_duplicates().sort_values(ascending=True))
print(a == b)
print("Q47")
print("\n")



base_selected_features = base_selected_features.loc[:, ["NU_INSCRICAO",
    "CO_UF_RESIDENCIA",
    "SG_UF_RESIDENCIA",
    "NU_IDADE",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_NACIONALIDADE",
    "TP_ST_CONCLUSAO",
    "TP_ANO_CONCLUIU",
    "TP_ENSINO",
    "IN_TREINEIRO",
    "TP_DEPENDENCIA_ADM_ESC",
    "IN_BAIXA_VISAO",
    "IN_SURDEZ",
    "IN_SABATISTA",
    "IN_GESTANTE",
    "TP_PRESENCA_LC",
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "TP_LINGUA",
    "TP_STATUS_REDACAO",
    "NU_NOTA_COMP1",
    "NU_NOTA_COMP2",
    "NU_NOTA_COMP3",
    "NU_NOTA_COMP4",
    "NU_NOTA_COMP5",
    "NU_NOTA_REDACAO",
    "Q001",
    "Q002",
    "Q006",
    "Q024",
    "Q025",
    "Q026",
    "Q027",
    "Q047",
    "NU_NOTA_MT"]]







test_selected_features = test_selected_features.loc[:, ["NU_INSCRICAO",
    "CO_UF_RESIDENCIA",
    "SG_UF_RESIDENCIA",
    "NU_IDADE",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_NACIONALIDADE",
    "TP_ST_CONCLUSAO",
    "TP_ANO_CONCLUIU",
    "TP_ENSINO",
    "IN_TREINEIRO",
    "TP_DEPENDENCIA_ADM_ESC",
    "IN_BAIXA_VISAO",
    "IN_SURDEZ",
    "IN_SABATISTA",
    "IN_GESTANTE",
    "TP_PRESENCA_LC",
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "TP_LINGUA",
    "TP_STATUS_REDACAO",
    "NU_NOTA_COMP1",
    "NU_NOTA_COMP2",
    "NU_NOTA_COMP3",
    "NU_NOTA_COMP4",
    "NU_NOTA_COMP5",
    "NU_NOTA_REDACAO",
    "Q001",
    "Q002",
    "Q006",
    "Q024",
    "Q025",
    "Q026",
    "Q027",
    "Q047"]]






#print(test_selected_features.shape)

test_selected_features = test_selected_features.set_index('NU_INSCRICAO')
base_selected_features = base_selected_features.set_index('NU_INSCRICAO')

#print(test_selected_features.info())
#print(base_selected_features.info())

#print(test_selected_features.head())





base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['CO_UF_RESIDENCIA'] ,prefix='CO_UF_RESIDENCIA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['SG_UF_RESIDENCIA'] ,prefix='SG_UF_RESIDENCIA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_SEXO'] ,prefix='TP_SEXO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_COR_RACA'] ,prefix='TP_COR_RACA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_NACIONALIDADE'] ,prefix='TP_NACIONALIDADE', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ST_CONCLUSAO'] ,prefix='TP_ST_CONCLUSAO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ANO_CONCLUIU'] ,prefix='TP_ANO_CONCLUIU', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ESCOLA'] ,prefix='TP_ESCOLA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ENSINO'] ,prefix='TP_ENSINO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_TREINEIRO'] ,prefix='IN_TREINEIRO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_DEPENDENCIA_ADM_ESC'] ,prefix='TP_DEPENDENCIA_ADM_ESC', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_BAIXA_VISAO'] ,prefix='IN_BAIXA_VISAO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_SURDEZ'] ,prefix='IN_SURDEZ', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_DISLEXIA'] ,prefix='IN_DISLEXIA', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_DISCALCULIA'] ,prefix='IN_DISCALCULIA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_SABATISTA'] ,prefix='IN_SABATISTA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_GESTANTE'] ,prefix='IN_GESTANTE', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['IN_IDOSO'] ,prefix='IN_IDOSO', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_CN'] ,prefix='TP_PRESENCA_CN', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_CH'] ,prefix='TP_PRESENCA_CH', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_LC'] ,prefix='TP_PRESENCA_LC', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_LINGUA'] ,prefix='TP_LINGUA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_STATUS_REDACAO'] ,prefix='TP_STATUS_REDACAO', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q001'] ,prefix='Q001', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q002'] ,prefix='Q002', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q006'] ,prefix='Q006', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q024'] ,prefix='Q024', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q025'] ,prefix='Q025', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q026'] ,prefix='Q026', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q027'] ,prefix='Q027', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q047'] ,prefix='Q047', drop_first=True    )], axis=1 )




base_selected_features.drop(['CO_UF_RESIDENCIA'],axis=1, inplace=True)
base_selected_features.drop(['SG_UF_RESIDENCIA'],axis=1, inplace=True)
base_selected_features.drop(['TP_SEXO'],axis=1, inplace=True)
base_selected_features.drop(['TP_COR_RACA'],axis=1, inplace=True)

base_selected_features.drop(['TP_NACIONALIDADE'],axis=1, inplace=True)
base_selected_features.drop(['TP_ST_CONCLUSAO'],axis=1, inplace=True)
base_selected_features.drop(['TP_ANO_CONCLUIU'],axis=1, inplace=True)

#base_selected_features.drop(['TP_ESCOLA'],axis=1, inplace=True)
base_selected_features.drop(['TP_ENSINO'],axis=1, inplace=True)
base_selected_features.drop(['IN_TREINEIRO'],axis=1, inplace=True)
base_selected_features.drop(['TP_DEPENDENCIA_ADM_ESC'],axis=1, inplace=True)
base_selected_features.drop(['IN_BAIXA_VISAO'],axis=1, inplace=True)
base_selected_features.drop(['IN_SURDEZ'],axis=1, inplace=True)

#base_selected_features.drop(['IN_DISLEXIA'],axis=1, inplace=True)
#base_selected_features.drop(['IN_DISCALCULIA'],axis=1, inplace=True)
base_selected_features.drop(['IN_SABATISTA'],axis=1, inplace=True)
base_selected_features.drop(['IN_GESTANTE'],axis=1, inplace=True)
#base_selected_features.drop(['IN_IDOSO'],axis=1, inplace=True)
#base_selected_features.drop(['TP_PRESENCA_CN'],axis=1, inplace=True)
#base_selected_features.drop(['TP_PRESENCA_CH'],axis=1, inplace=True)
base_selected_features.drop(['TP_PRESENCA_LC'],axis=1, inplace=True)
base_selected_features.drop(['TP_LINGUA'],axis=1, inplace=True)
base_selected_features.drop(['TP_STATUS_REDACAO'],axis=1, inplace=True)

base_selected_features.drop(['Q001'],axis=1, inplace=True)
base_selected_features.drop(['Q002'],axis=1, inplace=True)
base_selected_features.drop(['Q006'],axis=1, inplace=True)
base_selected_features.drop(['Q024'],axis=1, inplace=True)
base_selected_features.drop(['Q025'],axis=1, inplace=True)
base_selected_features.drop(['Q026'],axis=1, inplace=True)
base_selected_features.drop(['Q027'],axis=1, inplace=True)
base_selected_features.drop(['Q047'],axis=1, inplace=True)
















test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['CO_UF_RESIDENCIA'] ,prefix='CO_UF_RESIDENCIA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['SG_UF_RESIDENCIA'] ,prefix='SG_UF_RESIDENCIA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_SEXO'] ,prefix='TP_SEXO', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_COR_RACA'] ,prefix='TP_COR_RACA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_NACIONALIDADE'] ,prefix='TP_NACIONALIDADE', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_ST_CONCLUSAO'] ,prefix='TP_ST_CONCLUSAO', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_ANO_CONCLUIU'] ,prefix='TP_ANO_CONCLUIU', drop_first=True)], axis=1 )

#test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_ESCOLA'] ,prefix='TP_ESCOLA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_ENSINO'] ,prefix='TP_ENSINO', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_TREINEIRO'] ,prefix='IN_TREINEIRO', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_DEPENDENCIA_ADM_ESC'] ,prefix='TP_DEPENDENCIA_ADM_ESC', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_BAIXA_VISAO'] ,prefix='IN_BAIXA_VISAO', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_SURDEZ'] ,prefix='IN_SURDEZ', drop_first=True)], axis=1 )

#test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_DISLEXIA'] ,prefix='IN_DISLEXIA', drop_first=True)], axis=1 )

#test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_DISCALCULIA'] ,prefix='IN_DISCALCULIA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_SABATISTA'] ,prefix='IN_SABATISTA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_GESTANTE'] ,prefix='IN_GESTANTE', drop_first=True)], axis=1 )

#test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['IN_IDOSO'] ,prefix='IN_IDOSO', drop_first=True)], axis=1 )

#test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_PRESENCA_CN'] ,prefix='TP_PRESENCA_CN', drop_first=True)], axis=1 )

#test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_PRESENCA_CH'] ,prefix='TP_PRESENCA_CH', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_PRESENCA_LC'] ,prefix='TP_PRESENCA_LC', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_LINGUA'] ,prefix='TP_LINGUA', drop_first=True)], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['TP_STATUS_REDACAO'] ,prefix='TP_STATUS_REDACAO', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q001'] ,prefix='Q001', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q002'] ,prefix='Q002', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q006'] ,prefix='Q006', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q024'] ,prefix='Q024', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q025'] ,prefix='Q025', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q026'] ,prefix='Q026', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q027'] ,prefix='Q027', drop_first=True    )], axis=1 )

test_selected_features = pd.concat( [test_selected_features, pd.get_dummies(test_selected_features['Q047'] ,prefix='Q047', drop_first=True    )], axis=1 )





test_selected_features.drop(['CO_UF_RESIDENCIA'],axis=1, inplace=True)
test_selected_features.drop(['SG_UF_RESIDENCIA'],axis=1, inplace=True)
test_selected_features.drop(['TP_SEXO'],axis=1, inplace=True)
test_selected_features.drop(['TP_COR_RACA'],axis=1, inplace=True)

test_selected_features.drop(['TP_NACIONALIDADE'],axis=1, inplace=True)
test_selected_features.drop(['TP_ST_CONCLUSAO'],axis=1, inplace=True)
test_selected_features.drop(['TP_ANO_CONCLUIU'],axis=1, inplace=True)

#test_selected_features.drop(['TP_ESCOLA'],axis=1, inplace=True)
test_selected_features.drop(['TP_ENSINO'],axis=1, inplace=True)
test_selected_features.drop(['IN_TREINEIRO'],axis=1, inplace=True)
test_selected_features.drop(['TP_DEPENDENCIA_ADM_ESC'],axis=1, inplace=True)
test_selected_features.drop(['IN_BAIXA_VISAO'],axis=1, inplace=True)
test_selected_features.drop(['IN_SURDEZ'],axis=1, inplace=True)

#test_selected_features.drop(['IN_DISLEXIA'],axis=1, inplace=True)
#test_selected_features.drop(['IN_DISCALCULIA'],axis=1, inplace=True)
test_selected_features.drop(['IN_SABATISTA'],axis=1, inplace=True)
test_selected_features.drop(['IN_GESTANTE'],axis=1, inplace=True)
#test_selected_features.drop(['IN_IDOSO'],axis=1, inplace=True)
#test_selected_features.drop(['TP_PRESENCA_CN'],axis=1, inplace=True)
#test_selected_features.drop(['TP_PRESENCA_CH'],axis=1, inplace=True)
test_selected_features.drop(['TP_PRESENCA_LC'],axis=1, inplace=True)
test_selected_features.drop(['TP_LINGUA'],axis=1, inplace=True)
test_selected_features.drop(['TP_STATUS_REDACAO'],axis=1, inplace=True)

test_selected_features.drop(['Q001'],axis=1, inplace=True)
test_selected_features.drop(['Q002'],axis=1, inplace=True)
test_selected_features.drop(['Q006'],axis=1, inplace=True)
test_selected_features.drop(['Q024'],axis=1, inplace=True)
test_selected_features.drop(['Q025'],axis=1, inplace=True)
test_selected_features.drop(['Q026'],axis=1, inplace=True)
test_selected_features.drop(['Q027'],axis=1, inplace=True)
test_selected_features.drop(['Q047'],axis=1, inplace=True)


print(test_selected_features.shape)
print(base_selected_features.shape)

sc = StandardScaler()




X_train, X_test, y_train, y_test = train_test_split(
        base_selected_features.loc[:, base_selected_features.columns != "NU_NOTA_MT"],
        base_selected_features.loc[:, base_selected_features.columns == "NU_NOTA_MT"],
        test_size=0.33, random_state=42)


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
'''
regr = RandomForestRegressor(n_estimators = 100,
                             min_samples_split = 5,
                             min_samples_leaf = 1, 
                             max_features = 'auto', 
                             max_depth = 10,
                             bootstrap =True)

regr.fit(X_train, y_train)
y_predict_train = regr.predict(X_train)
y_predict_test = regr.predict(X_test)

r2_train = r2_score(y_train,y_predict_train)
r2_test = r2_score(y_test,y_predict_test)

print("R2 treino: " + str(r2_train))
print("R2 test: " + str(r2_test))
mse_a = mean_squared_error(y_test, regr.predict(X_test))
print("MSE test: " + str(mse_a))

'''


model = Sequential()

# Add a dropout layer for input layer
model.add(layers.Dropout(0.2, input_shape=(163,)))

# Add fully connected layer with a ReLU activation function
model.add(layers.Dense(units=16, activation='relu'))

# Add a dropout layer for previous hidden layer
model.add(layers.Dropout(0.5))

# Add fully connected layer with a ReLU activation function
model.add(layers.Dense(units=16, activation='relu'))

# Add a dropout layer for previous hidden layer
model.add(layers.Dropout(0.5))

# Add fully connected layer with a sigmoid activation function
model.add(layers.Dense(units=1, activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))


y_predict_train = model.predict(X_train)
y_predict_test = model.predict(X_test)

r2_train = r2_score(y_train,y_predict_train)
r2_test = r2_score(y_test,y_predict_test)

print("R2 treino: " + str(r2_train))
print("R2 test: " + str(r2_test))
mse_a = mean_squared_error(y_test, regr.predict(X_test))
print("MSE test: " + str(mse_a))






'''



print("\n")
regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
y_predict_train = regr.predict(X_train)
y_predict_test = regr.predict(X_test)

r2_train = r2_score(y_train,y_predict_train)
r2_test = r2_score(y_test,y_predict_test)

print("R2 treino: " + str(r2_train))
print("R2 test: " + str(r2_test))
print("MSE test: " + str(mean_squared_error(y_test, regr.predict(X_test))))

'''
test_selected_features["NU_NOTA_MT"] = regr.predict(test_selected_features)
df = test_selected_features.loc[:, test_selected_features.columns == "NU_NOTA_MT"]
df.to_csv("answer.csv", index=True)




