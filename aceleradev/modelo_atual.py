import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

base_total = pd.read_csv("train.csv", header=0)


base_selected_features = base_total.loc[:, ["NU_IDADE",
                                            "TP_SEXO",
                                            "TP_ESTADO_CIVIL",
                                            "TP_COR_RACA",
                                            "TP_NACIONALIDADE",
                                            "CO_UF_NASCIMENTO",
                                            "TP_ST_CONCLUSAO",
                                            "TP_ANO_CONCLUIU",
                                            "TP_ESCOLA",
                                            "TP_ENSINO",
                                            "IN_TREINEIRO",
                                            "NU_NOTA_MT"]]


base_selected_features = base_selected_features.dropna()



base_selected_features = base_selected_features.loc[:, ["NU_IDADE",
                                            "TP_SEXO",
                                            "TP_ESTADO_CIVIL",
                                            "TP_COR_RACA",
                                            "CO_UF_NASCIMENTO",
                                            "TP_ESCOLA",
                                            "NU_NOTA_MT"]]


base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_SEXO'] ,prefix='TP_SEXO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ESTADO_CIVIL'] ,prefix='TP_ESTADO_CIVIL', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_COR_RACA'] ,prefix='TP_COR_RACA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['CO_UF_NASCIMENTO'] ,prefix='CO_UF_NASCIMENTO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ESCOLA'] ,prefix='TP_ESCOLA', drop_first=True)], axis=1 )


base_selected_features.drop(['TP_SEXO'],axis=1, inplace=True)
base_selected_features.drop(['TP_ESTADO_CIVIL'],axis=1, inplace=True)
base_selected_features.drop(['TP_COR_RACA'],axis=1, inplace=True)
base_selected_features.drop(['CO_UF_NASCIMENTO'],axis=1, inplace=True)
base_selected_features.drop(['TP_ESCOLA'],axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(
        base_selected_features.loc[:, base_selected_features.columns != "NU_NOTA_MT"],
        base_selected_features.loc[:, base_selected_features.columns == "NU_NOTA_MT"],
        test_size=0.33, random_state=42)




regr = RandomForestRegressor(n_estimators=100, random_state=0)

y_train = np.reshape(y_train, (np.shape(y_train)[0],))
print(np.shape(y_train))
regr.fit(X_train, y_train)
y_predict_train = regr.predict(X_train)
y_predict_test = regr.predict(X_test)

r2_train = r2_score(y_train,y_predict_train)
r2_test = r2_score(y_test,y_predict_test)

print("R2 treino: " + str(r2_train))
print("R2 test: " + str(r2_test))

