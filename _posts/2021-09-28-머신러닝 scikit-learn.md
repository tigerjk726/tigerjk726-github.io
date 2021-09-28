#  머신러닝 scikit-learn

첫 번째 모델을 학습도 시켜보고, 성능도 평가해봤으니 이제는 다른 모델들도 활용해 보겠습니다!

다른 모델들을 사용하는 것 또한, 편리하게 설계된 scikit-learn 덕분에 아주 간단합니다.
scikit-learn을 사용하는 것에 익숙해진다면 심지어 한 줄의 코드만 수정해도 된다구요!

다른 모델들을 다루기 전에 위에서 사용했던 Decision Tree 모델을 학습시키고 예측하는 과정을 한 번에 담아보겠습니다.

```
# (1) 필요한 모듈 import
```

```
from sklearn.datasets import load_iris
```

```
from sklearn.model_selection import train_test_split
```

```
from sklearn.tree import DecisionTreeClassifier
```

```
from sklearn.metrics import classification_report
```

```

```

```
# (2) 데이터 준비
```

```
iris = load_iris()
```

```
iris_data = iris.data
```

```
iris_label = iris.target
```

```

```

```
# (3) train, test 데이터 분리
```

```
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
```

```
                                                    iris_label, 
```

```
                                                    test_size=0.2, 
```

```
                                                    random_state=7)
```

```

```

```
# (4) 모델 학습 및 예측
```

```
decision_tree = DecisionTreeClassifier(random_state=32)
```

```
decision_tree.fit(X_train, y_train)
```

```
y_pred = decision_tree.predict(X_test)
```

```

```

```
print(classification_report(y_test, y_pred))
```

코드 실행 

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         7
           1       0.91      0.83      0.87        12
           2       0.83      0.91      0.87        11

    accuracy                           0.90        30
   macro avg       0.91      0.91      0.91        30
weighted avg       0.90      0.90      0.90        30
```

Decision Tree를 여러개 모아놓은 **RandomForest**입니다.
위에서 RandomForest는 Decision Tree 모델을 여러개 합쳐놓음으로써 Decision Tree의 단점을 극복한 모델이라고 소개했죠.
이러한 기법을 **앙상블(Ensemble)** 기법이라고 합니다. 단일 모델을 여러 개 사용하는 방법을 취함으로써 모델 한 개만 사용할 때의 단점을 집단지성으로 극복하는 개념이죠.

이러한 개념을 잘 설명한 다음 글을 읽어봅시다.

- [군중은 똑똑하다 — Random Forest](https://medium.com/@deepvalidation/title-3b0e263605de)



Random Forest는 여러개의 의사 결정 트리를 모아 놓은것으로, 각각의 의사 결정 트리를 만들기 위해 쓰이는 특성들을 랜덤으로 선택한다.

이는 상위 모델들이 예측하는 편향된 결과보다, 다양한 모델들의 결과를 반영함으로써 더 다양한 데이터에 대한 의사결정을 내릴 수 있게 합니다.

이러한 이유로 RandomForest는 `sklearn.ensemble` 패키지 내에 들어있습니다. 다음과 같이 사용할 수 있습니다.

```
from sklearn.ensemble import RandomForestClassifier
```

```

```

```
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
```

```
                                                    iris_label, 
```

```
                                                    test_size=0.2, 
```

```
                                                    random_state=21)
```

```

```

```
random_forest = RandomForestClassifier(random_state=32)
```

```
random_forest.fit(X_train, y_train)
```

```
y_pred = random_forest.predict(X_test)
```

```

```

```
print(classification_report(y_test, y_pred))
```

코드 실행 

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      0.83      0.91        12
           2       0.78      1.00      0.88         7

    accuracy                           0.93        30
   macro avg       0.93      0.94      0.93        30
weighted avg       0.95      0.93      0.93        30
```



### 다른 scikit-learn 내장 분류모델

------

이 외에 scikit-learn에 내장된 기본 분류 모델들을 몇 가지 더 사용해 보겠습니다.
오늘은 각 모델에 대한 깊은 이론적인 내용보다는, 사용해 볼 수 있는 것에 초점을 맞추어 연습을 해 볼 것입니다.
각 모델에 대한 내용들을 이해하기 위한 좋은 글들을 하나씩 첨부하니, 꼭 한 번씩 읽고 넘어가기를 권합니다.

코드 사용은 아시다시피 어렵지 않습니다. 전체 틀은 모두 같으니, 직접 코드를 짜보세요!

#### Support Vector Machine (SVM)

다음 글을 읽고 Support Vector Machine에 대해 알아봅시다.

- [Support Vector Machine (SVM, 서포트 벡터 머신)](https://excelsior-cjh.tistory.com/66?category=918734)

SVM 모델은 다음과 같이 사용합니다.

```
from sklearn import svm
```

```
svm_model = svm.SVC()
```



```
from sklearn.ensemble import RandomForestClassifier
```

```

```

```
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
```

```
                                                    iris_label, 
```

```
                                                    test_size=0.2, 
```

```
                                                    random_state=21)
```

```

```

```
svm_model = svm.SVC()
```

```
svm_model.fit(X_train, y_train)
```

```
y_pred = svm_model.predict(X_test)
```

```

```

```
print(classification_report(y_test, y_pred))
```

코드 실행 

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.91      0.83      0.87        12
           2       0.75      0.86      0.80         7

    accuracy                           0.90        30
   macro avg       0.89      0.90      0.89        30
weighted avg       0.91      0.90      0.90        30
```



#### Stochastic Gradient Descent Classifier (SGDClassifier)

다음 글을 읽고 Stochastic Gradient Descent Classifier에 대해 알아봅시다.

- [사이킷런 공식문서-Stochastic Gradient Descent Classifier](https://scikit-learn.org/stable/modules/sgd.html)

SGD Classifier 모델은 다음과 같이 사용합니다.

```
from sklearn.linear_model import SGDClassifier
```

```
sgd_model = SGDClassifier()
```

```

```

```
print(sgd_model._estimator_type)
```

```
from sklearn.ensemble import RandomForestClassifier
```

```

```

```
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
```

```
                                                    iris_label, 
```

```
                                                    test_size=0.2, 
```

```
                                                    random_state=21)
```

```

```

```
sgd_model = SGDClassifier()
```

```
sgd_model.fit(X_train, y_train)
```

```
y_pred = sgd_model.predict(X_test)
```

```

```

```
print(classification_report(y_test, y_pred))
```

코드 실행 

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      0.83      0.91        12
           2       0.78      1.00      0.88         7

    accuracy                           0.93        30
   macro avg       0.93      0.94      0.93        30
weighted avg       0.95      0.93      0.93        30
```



#### Logistic Regression

다음 글을 읽고 Logistic Regression 모델에 대해 알아봅시다.

- [로지스틱회귀(Logistic Regression) 쉽게 이해하기](http://hleecaster.com/ml-logistic-regression-concept/)

Logistic Regression 모델은 다음과 같이 사용합니다.

```
from sklearn.linear_model import LogisticRegression
```

```
logistic_model = LogisticRegression()
```

```

```

```
print(logistic_model._estimator_type)
```

```
from sklearn.ensemble import RandomForestClassifier
```

```

```

```
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
```

```
                                                    iris_label, 
```

```
                                                    test_size=0.2, 
```

```
                                                    random_state=21)
```



```
logistic_model = LogisticRegression()
```

```
logistic_model.fit(X_train, y_train)
```

```
y_pred = logistic_model.predict(X_test)
```



코드 실행 

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      0.83      0.91        12
           2       0.78      1.00      0.88         7

    accuracy                           0.93        30
   macro avg       0.93      0.94      0.93        30
weighted avg       0.95      0.93      0.93        30
```

