import numpy as np, itertools, pandas as pd, matplotlib.pyplot as plt, pydotplus, matplotlib.image as mpimg
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from six import StringIO
from IPython.display import Image
import os

#os.environ["PATH"] += os.pathsep + '/Graphviz2.38/bin'
#Importando o dataset para o dataframe
df = pd.read_csv('../datasets/dataset_einstein.csv', delimiter=';')

#Mostrando as primeiras 5 linhas
print(df.head(5))

count_row = df.shape[0] # Pegando os numeros de registros
count_col = df.shape[1] # Pegando os numeros de colunas

print((count_row))
print((count_col))
# Reparem que ha muitos registros em que ha dados faltando nos campos

# Removendo os registros nos quais pelomenos um campo esta em branco(NAN
df = df.dropna()
print(df.head(5))

print('Quantidades de campos(colunas): ', df.shape[1])
print('Total de registros: ', df.shape[0])

# Vamos verificar se o banco de dados esta balanceado ou desbalanceado.
print('Total de registros negativos: ', df[df['SARS-Cov-2 exam result'] == 'negative'].shape[0])
print('Total de registros positivos: ', df[df['SARS-Cov-2 exam result'] == 'positive'].shape[0])

print()
# Vamos jogas as etiquetas para Y
Y = df['SARS-Cov-2 exam result'].values
print(Y)

# X sera a nossa matrix com as featrures
# Vamos pegar os campos de treinamento (Hemoglobin, Leukocytes, Basophils, Proteina C reativa mg/dl)
print()
X = df[['Hemoglobin', 'Leukocytes', 'Basophils', 'Proteina C reativa mg/dL']].values
print(X)

# Separar dados para treino e para teste.
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=.2, random_state=3)

# Criar um algoritmo de arvore de decisao
algoritm_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=5)
# Agora em minha arvore eu tenho associada a ela o algoritmo de treinamento
# Basicamente a receita que vimos na parte teorica

# Agora precisamos treina-la
modelo = algoritm_arvore.fit(X_treino, Y_treino)

# Podemos mostrar as features mais importantes
print(modelo.feature_importances_)

nome_features = ['Hemoglobin', 'Leukocytes', 'Basophils', 'Proteina C reativa mg/dL']
nome_classes = modelo.classes_

# # Montar a imgem da arvore
# dot_data = StringIO()
# #dot_data = tree.export_graphviz(my_tree_one, out_file=None, feature_names=featureNames)
# export_graphviz(modelo, out_file=dot_data, filled=True, feature_names=nome_features, class_names=nome_classes, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
# graph.write_png("arvore.png")
# Image('arvore.png')

importances = modelo.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices],
    color="b",
    align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()

#Indice das features
# 0 - 'Hemoglobin',
# 1 - 'Leukocytes'
# 2 - 'Basophils',
# 3 - 'Proteina C reativa mg/dL']

# APLICANDO O MODELO NA BASE DE TESTES E ARMAZENDO O RESULTADO EM Y_PREDICOES
Y_predicoes = modelo.predict(X_teste)

#AVALIAÇÃO DO MODELO
#VAMOS AVALIAR O VALOR REAL DO DATASET Y_TESTE COM AS PREDIÇÕES
print("ACURÁCIA DA ÁRVORE: ", accuracy_score(Y_teste, Y_predicoes))
print (classification_report(Y_teste, Y_predicoes))

# PRECISÃO: DAS CLASSIFICAÇÕES QUE O MODELO FEZ PARA UMA DETERMINADA CLASSE
# RECALL: DOS POSSÍVEIS DATAPOINTS PERTECENTES A UMA DETERMINADA CLASSE

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão sem normalizacão ')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo real')
    plt.xlabel('Rótulo prevista')


matrix_confusao = confusion_matrix(Y_teste, Y_predicoes)
plt.figure()
plot_confusion_matrix(matrix_confusao, classes=nome_classes,
                      title='Matrix de Confusao')