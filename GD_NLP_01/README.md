# AIFFEL Campus Online 4th Code Peer Review Templete

- 코더 : 소용현
- 리뷰어 : 임지혜


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [O] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```python
"""
val_accuracy: 0.852
1537/1537 - 13s - loss: 0.3412 - accuracy: 0.8525
[0.34121909737586975, 0.8525480628013611] -> 정확도 80%이상
"""
```

- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?
```python
#함수에 대한 설명을 자세하게 적어줌
    X_train = []
    for sentence in train_data['document']:
        temp_X = tokenizer.morphs(sentence)  # 토큰화
#         tagging = tokenizer.pos(sentence) # 형태소 분석
#         temp_X = [t[0] for t in tagging if ('V' in t[1] or 'N' in t[1]) and 'UNKNOWN' not in t[1] and len(t[0])>1] #형태소 분석 결과 중 명사/동사/형용사와 같은 의미있는 단어만 선택 
#         temp_X = [t[0] for t in tagging if 'UNKNOWN' not in t[1] and len(t[0])>1]
```

- ['x] 3.코드가 에러를 유발할 가능성이 있나요?
  > 없는것 같습니다, 에러없이 mecab, spm모두 정확도 80%이상을 달성했습니다

  
- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
#모델 선언시 언어모델에 적합하게 가장 널리 쓰이는 RNN인 LSTM, bidirectional을 사용함
modelLSTM_SPM.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512))) 

#LSTM특성 고려하여 패딩위치 post -> pre로 조절
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                        value=word_to_index["<PAD>"],
                                                        padding='pre', # 혹은 'pre'
                                                        maxlen=maxlen)
```

- [O] 5.코드가 간결한가요?
```python
#모델 compile및 fit을 한꺼번에 처리하는 코드 선언 -> 중복코드 제거
def modelFitTemp(model,X_train,y_train,epochs=20) :
    # validation set 분리
    x_val = X_train[:len(X_train)//7]   
    y_val = y_train[:len(X_train)//7]

    # validation set을 제외한 나머지 
    partial_x_train = X_train[len(X_train)//7:]  
    partial_y_train = y_train[len(X_train)//7:]
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    epochs=epochs  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]
    history = model.fit(~~)
    return history


#tokenizer함수를 따로 만들지 않고 load_data_spm안에 함께 정리
def load_data_spm(train_data, test_data, s):
    X_train = []
    for sentence in train_data['document']:
        temp_X = s.EncodeAsIds(sentence)  # 토큰화
        X_train.append(temp_X)

    X_test = []
    for sentence in test_data['document']:
        temp_X = s.EncodeAsIds(sentence)  # 토큰화
        X_test.append(temp_X)
    
    with open("./korean_spm.vocab", 'r') as f:
        vocab = f.readlines()
```
