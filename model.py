import mlflow.sklearn
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

with mlflow.start_run(run_name='iris classification') as run:
    # Iris dataset classification
    print("Iris dataset classification with SVC")
    iris = load_iris()
    x, y = iris.data, iris.target
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)

    lsvc = LinearSVC()
    print(lsvc)

    lsvc.fit(xtrain, ytrain)
    mlflow.sklearn.log_model(lsvc, "iris-classifier")
    score = lsvc.score(xtrain, ytrain)
    print("Score: ", score)

    cv_scores = cross_val_score(lsvc, xtrain, ytrain, cv=10)
    print("CV average score: %.2f" % cv_scores.mean())

    ypred = lsvc.predict(xtest)

    cm = confusion_matrix(ytest, ypred)
    print(cm)
    plot_confusion_matrix(lsvc, xtest, ytest)
    plt.savefig("confusion-matrix.png")
    mlflow.log_artifact("confusion-matrix.png")

    cr = classification_report(ytest, ypred, output_dict=True)
    recall_0 = cr['0']['recall']
    f1_score_0 = cr['0']['f1-score']
    recall_1 = cr['1']['recall']
    f1_score_1 = cr['1']['f1-score']


    acc = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred, average='micro')
    # confusion matrix values
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]

    mlflow.log_metric("accuracy_score", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("true_positive", tp)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)
    mlflow.log_metric("recall_0", recall_0)
    mlflow.log_metric("f1_score_0", f1_score_0)
    mlflow.log_metric("recall_1", recall_1)
    mlflow.log_metric("f1_score_1", f1_score_1)

