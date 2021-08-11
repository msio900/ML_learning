# 기본 라이브러리 불러오기
import warnings

from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

'''
[Step 1] 데이터 준비/ 기본 설정
'''

# Breast Cancer 데이터셋 가져오기 (출처: UCI ML Repository)
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header=None)

# 열 이름 지정
df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
              'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses', 'class']

#  IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 15)

'''
[Step 2] 데이터 탐색
'''

# 데이터 살펴보기
print(df.head())
print('\n')

# 데이터 자료형 확인
print(df.info())
print('\n')

# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')

# bare_nuclei 열의 자료형 변경 (문자열 ->숫자)
# bare_nuclei 열의 고유값 확인
print(df['bare_nuclei'].unique())
print('\n')

df['bare_nuclei'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['bare_nuclei'] = df['bare_nuclei'].astype('int')       # 문자열을 정수형으로 변환

print(df.describe())                                      # 데이터 통계 요약정보 확인
print('\n')

'''
[Step 3] 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)
'''

# 속성(변수) 선택
X = df[['clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial',
        'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']]  # 설명 변수 X
y = df['class']  # 예측 변수 Y

print(X);
print(y);

# 설명 변수 데이터를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

print(X);

# train data 와 test data로 구분(7:3 비율)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10)

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')

'''
[Step 4] Decision Tree 분류 모형 - sklearn 사용
'''

# DT 모형 성능 평가 - 평가지표 계산
tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree_model.fit(X_train, y_train)
y_hat = tree_model.predict(X_test)      # 2: benign(양성), 4: malignant(악성)
tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report)
knn_acc = accuracy_score(y_test,y_hat);
print('DT',knn_acc);

# SVM 모형 성능 평가 - 평가지표 계산
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_hat = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test,y_hat);
print('SVM',svm_acc);

# KNN모형 성능 평가 - 평가지표 계산
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_hat = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test,y_hat);
print('KNN',knn_acc);

# 앙상블모델 1 - voting 모형 성능 평가 - 평가지표 계산

hvc = VotingClassifier(estimators=[('KNN',knn_model),
                                   ('SVM',svm_model),
                                   ('DT',tree_model)],voting='hard')
hvc.fit(X_train,y_train);
y_hat = hvc.predict(X_test);
hvc_acc = accuracy_score(y_test,y_hat);
print('HVC',hvc_acc);

# 앙상블모델 2 -배깅 (random forest) 모형 성능 평가 - 평가지표 계산
rfc = RandomForestClassifier(n_estimators=50,max_depth=5,random_state=10);
rfc.fit(X_train, y_train);
y_hat = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test,y_hat);
print('RFC',rfc_acc);

# 앙상블모델 3 -  Boosting 모형 성능 평가 - 평가지표 계산
warnings.filterwarnings("ignore")
xgbc = XGBClassifier(n_estimators=50,max_depth=5,random_state=10);
xgbc.fit(X_train, y_train);
y_hat = xgbc.predict(X_test)
xgbc_acc = accuracy_score(y_test,y_hat);
print('XGC',xgbc_acc);
