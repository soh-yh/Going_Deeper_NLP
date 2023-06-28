## **Code Peer Review Templete**
------------------
- 코더 : 소용현
- 리뷰어 : Donggyu Kim

## **PRT(PeerReviewTemplate)**
------------------  
- [?] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?** (?/x/o)
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**
- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**
- [x] **5. 코드가 간결한가요?**

## Evidences
### section 1

평가문항	상세기준
1. 분류 모델의 accuracy가 기준 이상 높게 나왔는가?	3가지 단어 개수에 대해 8가지 머신러닝 기법을 적용하여 그중 최적의 솔루션을 도출하였다.
```python
print("num_words=None, DTM을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(None,True)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=None, TFIDF을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(None,False)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=10000, DTM을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(10000,True)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=10000, TFIDF을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(10000,False)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=5000, DTM을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(5000,True)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=5000, TFIDF을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(5000,False)
fit_ml(x_train, y_train, x_test, y_test)

NB 정확도: 0.7226179875333927
CNB 정확도: 0.7782724844167409
```
In the situation where the NB model was expected to be the lowest, the model's accuracy of 72% occurred.  
It is expected that a sufficiently good result will come out, but it is regrettable that there is no visualization data.
  
2. 분류 모델의 F1 score가 기준 이상 높게 나왔는가?	Vocabulary size에 따른 각 머신러닝 모델의 성능변화 추이를 살피고, 해당 머신러닝 알고리즘의 특성에 근거해 원인을 분석하였다.

there is no f1 score
  
4. 딥러닝 모델을 활용해 성능이 비교 및 확인되었는가?	동일한 데이터셋과 전처리 조건으로 딥러닝 모델의 성능과 비교하여 결과에 따른 원인을 분석하였다.

```
# 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(units=128))
model.add(Dense(units=46, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))

# 테스트 데이터에 대한 예측
y_pred = model.predict(x_test)

# 분류 보고서 출력
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
report = classification_report(y_test_labels, y_pred_labels)
print(report)
```

An appropriate deep learning model was created, and it was output as a report.

### section 2

yes.
There are kindfull comments.

```
word_index = reuters.get_word_index(path="reuters_word_index.json")
index_to_word = { index+3 : word for word, index in word_index.items() }
# index_to_word에 숫자 0은 <pad>, 숫자 1은 <sos>, 숫자 2는 <unk>를 넣어줍니다.
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
  index_to_word[index]=token
# print(' '.join([index_to_word[index] for index in x_train[0]]))
```
### section 4

```python
#보팅
voting_classifier = VotingClassifier(estimators=[('lr', lr), ('cb', cb), ('gnb', grbt)],voting='soft')
voting_classifier.fit(x_train, y_train)
predicted = voting_classifier.predict(x_test) #테스트 데이터에 대한 예측
print("보팅 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교

```
In the case of voting classifier, its usage is tricky. But, it seems to have been understood and utilized enoughfully.

### section 5

The code is fit ml function for batch fitting.
```python
def fit_ml(x_train, y_train, x_test, y_test) :
    #NB
    model = MultinomialNB()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test) #테스트 데이터에 대한 예측
    print("NB 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
    
    #CNB
    cb = ComplementNB()
    cb.fit(x_train, y_train)
    predicted = cb.predict(x_test) #테스트 데이터에 대한 예측
    print("CNB 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
    
    #로지스틱회귀
    lr = LogisticRegression(C=10000, penalty='l2', max_iter=3000)
    lr.fit(x_train, y_train)
    predicted = lr.predict(x_test) #테스트 데이터에 대한 예측
    print("로지스틱회귀 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
    
    #svc
    lsvc = LinearSVC(C=1000, penalty='l1', max_iter=3000, dual=False)
    lsvc.fit(x_train, y_train)
    predicted = lsvc.predict(x_test) #테스트 데이터에 대한 예측
    print("SVC 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
    
    #tree
    tree = DecisionTreeClassifier(max_depth=10, random_state=0)
    tree.fit(x_train, y_train)
    predicted = tree.predict(x_test) #테스트 데이터에 대한 예측
    print("tree 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
    
    #RandomForest
    forest = RandomForestClassifier(n_estimators =5, random_state=0)
    forest.fit(x_train, y_train)
    predicted = forest.predict(x_test) #테스트 데이터에 대한 예측
    print("RandomForest 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
    
    #GradientBoosting
    grbt = GradientBoostingClassifier(random_state=0) # verbose=3
    grbt.fit(x_train, y_train)
    predicted = grbt.predict(x_test) #테스트 데이터에 대한 예측
    print("GradientBoosting 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교

    #보팅
    voting_classifier = VotingClassifier(estimators=[('lr', lr), ('cb', cb), ('gnb', grbt)],voting='soft')
    voting_classifier.fit(x_train, y_train)
    predicted = voting_classifier.predict(x_test) #테스트 데이터에 대한 예측
    print("보팅 정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
```

The below code is experiment code.
```python
print("num_words=None, DTM을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(None,True)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=None, TFIDF을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(None,False)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=10000, DTM을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(10000,True)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=10000, TFIDF을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(10000,False)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=5000, DTM을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(5000,True)
fit_ml(x_train, y_train, x_test, y_test)
print("num_words=5000, TFIDF을 활용한 정확도") 
x_train, y_train, x_test, y_test = reuters_load_ml(5000,False)
fit_ml(x_train, y_train, x_test, y_test)
```

He tried to avoid unuseful duplicated code by using functions.
So, in the experiment part, the code is simple and clear.

## **참고링크 및 코드 개선 여부**



