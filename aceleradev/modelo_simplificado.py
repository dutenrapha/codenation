import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor

base_total = pd.read_csv("train.csv", header=0)


base_selected_features = base_total.loc[:, ["NU_IDADE",
                                            "TP_SEXO",
                                            "TP_COR_RACA",
                                            "NU_NOTA_CN",
                                            "NU_NOTA_CH",
                                            "NU_NOTA_LC",     
                                            "NU_NOTA_REDACAO",
                                            "Q001",
                                            "Q002",
                                            "Q003",
                                            "Q004",
                                            "Q005",
                                            "Q006",
                                            "Q007",
                                            "Q008",
                                            "Q009",
                                            "Q010",
                                            "Q011",
                                            "Q012",
                                            "Q013",
                                            "Q014",
                                            "Q015",
                                            "Q016",
                                            "Q017",
                                            "Q018",
                                            "Q019",
                                            "Q020",
                                            "Q021",
                                            "Q022",
                                            "Q023",
                                            "Q024",
                                            "Q025",
                                            "NU_NOTA_MT"]]




base_selected_features = base_selected_features.dropna()



base_selected_features = base_selected_features.loc[:, ["NU_IDADE",
                                            "TP_SEXO",                                        
                                            "NU_NOTA_CN",
                                            "NU_NOTA_CH",
                                            "NU_NOTA_LC",
                                            "NU_NOTA_REDACAO",
                                            "Q001",
                                            "Q002",
                                            "Q003",
                                            "Q004",
                                            "Q005",
                                            "Q006",
                                            "Q007",
                                            "Q008",
                                            "Q009",
                                            "Q010",
                                            "Q011",
                                            "Q012",
                                            "Q013",
                                            "Q014",
                                            "Q015",
                                            "Q016",
                                            "Q017",
                                            "Q018",
                                            "Q019",
                                            "Q020",
                                            "Q021",
                                            "Q022",
                                            "Q023",
                                            "Q024",
                                            "Q025",
                                            "NU_NOTA_MT"]]




base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_SEXO'] ,prefix='TP_SEXO', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ESTADO_CIVIL'] ,prefix='TP_ESTADO_CIVIL', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_COR_RACA'] ,prefix='TP_COR_RACA', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['CO_UF_NASCIMENTO'] ,prefix='CO_UF_NASCIMENTO', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_ESCOLA'] ,prefix='TP_ESCOLA', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_DEPENDENCIA_ADM_ESC'] ,prefix='TP_DEPENDENCIA_ADM_ESC', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_LOCALIZACAO_ESC'] ,prefix='TP_LOCALIZACAO_ESC', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_SIT_FUNC_ESC'] ,prefix='TP_SIT_FUNC_ESC', drop_first=True)], axis=1 )

#base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['CO_UF_ESC'] ,prefix='CO_UF_ESC', drop_first=True)], axis=1 )


'''

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_STATUS_REDACAO'] ,prefix='TP_STATUS_REDACAO', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_LINGUA'] ,prefix='TP_LINGUA', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_CN'] ,prefix='TP_PRESENCA_CN', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_CH'] ,prefix='TP_PRESENCA_CH', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_LC'] ,prefix='TP_PRESENCA_LC', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['TP_PRESENCA_MT'] ,prefix='TP_PRESENCA_MT', drop_first=True)], axis=1 )

'''

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q001'] ,prefix='Q001', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q002'] ,prefix='Q002', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q003'] ,prefix='Q003', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q004'] ,prefix='Q004', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q005'] ,prefix='Q005', drop_first=True)], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q006'] ,prefix='Q006', drop_first=True)], axis=1 )


base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q007'] ,prefix='Q007', drop_first=True)], axis=1 )


base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q008'] ,prefix='Q008', drop_first=True    )], axis=1 )





base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q009'] ,prefix='Q009', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q010'] ,prefix='Q010', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q011'] ,prefix='Q011', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q012'] ,prefix='Q012', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q013'] ,prefix='Q013', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q014'] ,prefix='Q014', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q015'] ,prefix='Q015', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q016'] ,prefix='Q016', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q017'] ,prefix='Q017', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q018'] ,prefix='Q018', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q019'] ,prefix='Q019', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q020'] ,prefix='Q020', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q021'] ,prefix='Q021', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q022'] ,prefix='Q022', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q023'] ,prefix='Q023', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q024'] ,prefix='Q024', drop_first=True    )], axis=1 )

base_selected_features = pd.concat( [base_selected_features, pd.get_dummies(base_selected_features['Q025'] ,prefix='Q025', drop_first=True    )], axis=1 )


'''






















'''
base_selected_features.drop(['TP_SEXO'],axis=1, inplace=True)
#base_selected_features.drop(['TP_ESTADO_CIVIL'],axis=1, inplace=True)
#base_selected_features.drop(['TP_COR_RACA'],axis=1, inplace=True)
#base_selected_features.drop(['CO_UF_NASCIMENTO'],axis=1, inplace=True)
#base_selected_features.drop(['TP_ESCOLA'],axis=1, inplace=True)
#base_selected_features.drop(['TP_DEPENDENCIA_ADM_ESC'],axis=1, inplace=True)
#base_selected_features.drop(['TP_LOCALIZACAO_ESC'],axis=1, inplace=True)
#base_selected_features.drop(['TP_SIT_FUNC_ESC'],axis=1, inplace=True)
#base_selected_features.drop(['CO_UF_ESC'],axis=1, inplace=True)

'''
base_selected_features.drop(['TP_STATUS_REDACAO'],axis=1, inplace=True)
base_selected_features.drop(['TP_LINGUA'],axis=1, inplace=True)
base_selected_features.drop(['TP_PRESENCA_CN'],axis=1, inplace=True)
base_selected_features.drop(['TP_PRESENCA_CH'],axis=1, inplace=True)
base_selected_features.drop(['TP_PRESENCA_LC'],axis=1, inplace=True)
base_selected_features.drop(['TP_PRESENCA_MT'],axis=1, inplace=True)



'''

base_selected_features.drop(['Q001'],axis=1, inplace=True)
base_selected_features.drop(['Q002'],axis=1, inplace=True)
base_selected_features.drop(['Q003'],axis=1, inplace=True)
base_selected_features.drop(['Q004'],axis=1, inplace=True)
base_selected_features.drop(['Q005'],axis=1, inplace=True)
base_selected_features.drop(['Q006'],axis=1, inplace=True)
base_selected_features.drop(['Q007'],axis=1, inplace=True)

base_selected_features.drop(['Q008'],axis=1, inplace=True)
base_selected_features.drop(['Q009'],axis=1, inplace=True)
base_selected_features.drop(['Q010'],axis=1, inplace=True)
base_selected_features.drop(['Q011'],axis=1, inplace=True)
base_selected_features.drop(['Q012'],axis=1, inplace=True)
base_selected_features.drop(['Q013'],axis=1, inplace=True)
base_selected_features.drop(['Q014'],axis=1, inplace=True)
base_selected_features.drop(['Q015'],axis=1, inplace=True)
base_selected_features.drop(['Q016'],axis=1, inplace=True)
base_selected_features.drop(['Q017'],axis=1, inplace=True)
base_selected_features.drop(['Q018'],axis=1, inplace=True)
base_selected_features.drop(['Q019'],axis=1, inplace=True)
base_selected_features.drop(['Q020'],axis=1, inplace=True)
base_selected_features.drop(['Q021'],axis=1, inplace=True)
base_selected_features.drop(['Q022'],axis=1, inplace=True)
base_selected_features.drop(['Q023'],axis=1, inplace=True)
base_selected_features.drop(['Q024'],axis=1, inplace=True)
base_selected_features.drop(['Q025'],axis=1, inplace=True)




X_train, X_test, y_train, y_test = train_test_split(
        base_selected_features.loc[:, base_selected_features.columns != "NU_NOTA_MT"],
        base_selected_features.loc[:, base_selected_features.columns == "NU_NOTA_MT"],
        test_size=0.33, random_state=42)




regr = RandomForestRegressor(n_estimators=1000, random_state=0)

print(np.shape(y_train))
print(np.shape(X_train))
regr.fit(X_train, y_train)
y_predict_train = regr.predict(X_train)
y_predict_test = regr.predict(X_test)

r2_train_rd = r2_score(y_train,y_predict_train)
r2_test_rd = r2_score(y_test,y_predict_test)

print("R2 treino rd: " + str(r2_train_rd))
print("R2 test rd: " + str(r2_test_rd))


#clf = tree.DecisionTreeRegressor()

#clf = svm.SVR()


clf = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')


clf.fit(X_train, y_train)
y_predict_train = clf.predict(X_train)
y_predict_test = clf.predict(X_test)

r2_train_rd_clf = r2_score(y_train,y_predict_train)
r2_test_rd_clf = r2_score(y_test,y_predict_test)

print("R2 treino clf: " + str(r2_train_rd_clf))
print("R2 test clf: " + str(r2_test_rd_clf))




