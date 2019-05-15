import pandas as pd
import numpy as np

import seaborn as sns
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def kMeans(noClusters, trainDataset, featureX, featureY):
    from sklearn.cluster import KMeans

    # Realiza el cluster del dataset
    clusters = KMeans(n_clusters=noClusters).fit_predict(trainDataset)

    # Graficar cluster de dos variables
    sns.scatterplot(trainDataset[featureX], trainDataset[featureY], hue=clusters)


def hierarchicalClustering(clusters, featureX, featureY, dataset):
    from sklearn.cluster import AgglomerativeClustering

    # for para probar cada tipo de linkage
    for linkage in ('ward', 'average', 'complete', 'single'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=clusters)

        clustering.fit(dataset[[featureX, featureY]])

        #grafica del cluster
        sns.scatterplot(x=featureX, y=featureY, data=dataset, hue=clustering.labels_)
        plt.figure()


def meanShiftClustering(dataset, featureX, featureY):
    from sklearn.cluster import MeanShift, estimate_bandwidth

    #el siguiente bandwith se puede detectar automaticamente usando:
    bandwidth = estimate_bandwidth(dataset, quantile=0.4)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(dataset)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # Agregar los centros a un dataframe
    centers_df = pd.DataFrame(cluster_centers).head()
    centers_df['x'] = centers_df[0]
    centers_df['y'] = centers_df[4]
    centers_df.head()

    #Graficas los cluster en dos dimensiones y superponer los centros
    sns.scatterplot(x=featureX, y=featureY, data=dataset, hue=labels)
    sns.scatterplot(x='x', y='y', data=centers_df, color='green', s=100)


def LDA(targetData, featureData):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    #Definir y hacer fit la data estandarizada (estandarizar data previo a la ejecucion de la funcion)
    lda = LDA()
    ld = lda.fit_transform(featureData, targetData)
    lda_df = pd.DataFrame(data=ld,
                          columns=['LDA1', 'LDA2'])
    lda_df['Cluster'] = targetData

    #Imprimir los resultados de la clasificacion del Training Data
    print('Accuracy of LDA classifier on training set: {:.2f}'
          .format(lda.score(featureData, targetData)))

    lda_df.head()
    lda.predict(featureData)

    # Scatter plot del primer y segundo LDA
    sns.lmplot(x="LDA1", y="LDA2",
               data=lda_df,
               fit_reg=False,
               hue='Cluster',  # color por cluster
               legend=True,
               scatter_kws={"s": 80})  # especificar el tamaño del punto

def PCA(featureData, targetData):
    # Definir el numero de componentes principales
    # Source: https://cmdlinetips.com/2018/03/pca-example-in-python-with-scikit-learn/

    pca = PCA(n_components= 3)

    # Fit a los features
    pc = pca.fit_transform(featureData)

    # Dataframe of principal components and wine labels
    pc_df = pd.DataFrame(data=pc,
                         columns=['PC1', 'PC2', 'PC3'])
    pc_df['Cluster'] = targetData
    pc_df.head()

    # Scatter plot del primer y segundo PCA
    sns.lmplot(x="PC1", y="PC2",
               data=pc_df,
               fit_reg=False,
               hue='Cluster',  # color por cluster
               legend=True,
               scatter_kws={"s": 80})  # especificar el tamaño de los puntos


def logisticRegression(featureData, targetData):
    from sklearn.linear_model import LogisticRegression
    # instanciar el modelo con los parametros de default
    logreg = LogisticRegression()

    #hacer fit al modelo
    lr_simple = logreg.fit(featureData, targetData)

    # Coeficientes del fitted model
    lr_simple.coef_

    # Intercepto del fitted model
    lr_simple.intercept_

    # Grafica de la funcion Sigmoide
    x_simple = featureData
    x_space = np.linspace(-5, 10, 100)

    loss = 1 / (1 + np.exp(-(x_space * lr_simple.coef_ + lr_simple.intercept_).ravel()))
    plt.plot(x_space, loss, color='red', linewidth=3)
    plt.scatter(featureData['tsh_diff'].ravel(), targetData, color='black', zorder=20)


def plotSVC(svc, param, X, y):
    clf = svc
    clf.fit(X, y)

    plt.clf()
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circular la test data
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X.iloc[:, 0].min()
    x_max = X.iloc[:, 0].max()
    y_min = X.iloc[:, 1].min()
    y_max = X.iloc[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    pre_z = svc.predict(np.c_[XX.ravel(), YY.ravel()])

    Z = pre_z.reshape(XX.shape)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'])

    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)
    plt.title(param)
    plt.show()


def SVM(kernel, featureData, targetData, gamma, c, degree):
    from sklearn.svm import SVC
    # kernels: 'linear', 'rbf', 'poly'

    clf = SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)
    clf.fit(featureData, targetData)

    predictions = clf.predict(featureData)

    results = pd.DataFrame({'Labels': targetData, 'Predictions': predictions})

    pd.concat([featureData, results], axis=1, sort=False).tail(15)

    plotSVC(clf, kernel, featureData, targetData)


def linearRegressino(trainX, trainY, testX):
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std

    # trainX solo acepta una variable (columna)
    simple_model = sm.OLS(trainY, trainX)
    simple_result = simple_model.fit()

    # Imprimir los resultados
    print(simple_result.summary())

    # Predecir los valores
    y_pred_simple = simple_result.predict(trainX)

    # Graficar la regression
    sns.scatterplot(x=testX, y=y_test.values.ravel())
    sns.lineplot(x=testX, y=y_pred_simple + 150)
    plt.title('Regression plot')

    # Grafica de residuos
    sns.scatterplot(x=y_pred_simple, y=(y_pred_simple - y_test.values.ravel()))
    plt.hlines(y=-150, xmin=-100, xmax=200)
    plt.title('Residual plot')


def multipleLinearRegression(trainX, trainY):
    import statsmodels.api as sm

    multiple_model = sm.OLS(trainY, trainX)

    # hace fit al modelo
    multiple_result = multiple_model.fit()
    print(multiple_result.summary())


def polynomialRegression(trainX, trainY, testX, degree):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from statsmodels.stats.anova import anova_lm
    import sklearn.metrics

    # Entrenar al modelo lineal para futuras comparaciones
    #Agrega una constante al dataset
    #trainX = sm.add_constant(trainX)
    simple_model = sm.OLS(trainY, trainX)

    simple_result = simple_model.fit()

    # Predecir en el test set y graficar resultados
    X_test = sm.add_constant(X_test)
    y_pred_simple = simple_result.predict(testX)
    sns.scatterplot(x=testX, y=y_test.values.ravel())
    sns.lineplot(x=testX, y=y_pred_simple)

    # Imprimir la evaluation metrics table
    print(simple_result.summary())

    # Create polynomial regression features of nth degree
    # Crea una regression polinomial de n grado
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly_train = poly_reg.fit_transform(pd.DataFrame(trainX))
    X_poly_test = poly_reg.fit_transform(pd.DataFrame(testX))
    poly_result = poly_reg.fit(X_poly_train, y_train)

    # Fit linear model now polynomial features
    poly_model = LinearRegression()
    poly_result = poly_model.fit(X_poly_train, y_train)
    y_poly_pred = poly_model.predict(X_poly_test)

    # Grafico para comparar modelos
    sns.scatterplot(x=testX, y=y_test.values.ravel())
    sns.lineplot(x=testX, y=y_pred_simple)
    sns.lineplot(x=testX, y=y_poly_pred.ravel())

    # Re entrenar para imprimir la summary table
    poly_model = sm.OLS(y_train, X_poly_train)
    poly_result = poly_model.fit()

    # Imprimir la model evaluation metrics
    print(poly_result.summary())

    # comparar los dos modelos usando ANOVA
    anovaResults = anova_lm(simple_result, poly_result)
    print(anovaResults)


def decisionTree(trainX, trainY):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    clf = DecisionTreeClassifier()
    clf = clf.fit(trainX, trainY)


    y_pred = clf.predict(X_test)
    accScr = []
    accScr.append(accuracy_score(y_pred, y_test))
    # Imprime el accuracy
    accScr[0]


def randomForest(trainX, trainY, testX, testY, noArboles):
    # n_estimators = Numero de arboles
    # n_jobs = cuantos procesadores queremos usar (-1 = todos)
    clf = RandomForestRegressor(n_estimators=noArboles, n_jobs=-1)

    clf.fit(trainX, trainY)
    from math import sqrt
    tree_predictions = clf.predict(testX)
    sqrt(mean_squared_error(testY, tree_predictions))
    # Se puede jugar con los parametros y encontrar los que mejor se
    #ajustan a nuestro data


def adaBoost(trainX, trainY, testX, testY, noArboles):
    adaboost_reg = AdaBoostRegressor(n_estimators=noArboles, learning_rate=1, loss='linear')
    adaboost_reg.fit(trainX, trainY)

    # Get training and test predictions
    prediction_train = adaboost_reg.score(trainX, trainY)
    prediction_test = adaboost_reg.score(testX, testY)
    adb = []
    print('Prediction Train: ', prediction_train)
    print('Prediction Test: ', prediction_test)
    adb.append(prediction_test)

