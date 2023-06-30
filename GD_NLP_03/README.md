# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 소용현
- 리뷰어 : 김다인


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- <img width="655" alt="image" src="https://github.com/soh-yh/Going_Deeper_NLP/assets/94978101/980d28c9-d1ca-4a36-8652-854755a4697c">
```python
import numpy as np; 
import seaborn as sns; 
import matplotlib.pyplot as plt
np.random.seed(0)

# 한글 지원 폰트
sns.set(font='NanumGothic')

# 마이너스 부호 

plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 15))
ax = sns.heatmap(matrix, xticklabels=genre_name, yticklabels=genre_name, annot=True,  cmap='RdYlGn_r')
ax
```
히트맵 구현까지 잘 구성되었습니다.   

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
  ```python
  #tfidf를 문서 두개에 적용하게 되면 빈도수 상위 기준 대부분이 df가 같다. 즉 idf가 같다.
#그러니 wordcount로 바꿔 비율을 구해보기로 했다.
word_counts_art = Counter(art)
word_counts_gen = Counter(gen)

#art에 등장한 단어 카운트를 gen에 등장한 카운트로 나눈다
word_ratios1 = {word: count / word_counts_gen[word] if word in word_counts_gen else 1 for word, count in word_counts_art.items() if count > 30}
sorted_word_ratios_art = sorted(word_ratios1.items(), key=lambda x: x[1], reverse=True)
#gen에 등장한 단어 카운트를 art에 등장한 카운트로 나눈다
word_ratios2 = {word: count / word_counts_art[word] if word in word_counts_art else 1 for word, count in word_counts_gen.items() if count > 30}
sorted_word_ratios_gen = sorted(word_ratios2.items(), key=lambda x: x[1], reverse=True)
print('예술영화를 대표하는 단어들:')
for i in range(100):
    print(sorted_word_ratios_art[i][0], end=', ')
print('\n')
    
print('일반영화를 대표하는 단어들:')
for i in range(100):
    print(sorted_word_ratios_gen[i][0], end=', ')
  ```
해당 코드를 왜 사용했는지에 대한 설명들이 충분해 이해하기 쉬웠다.   

- [❌] 3.코드가 에러를 유발할 가능성이 있나요?
  문제사항 없이 깔끔하게 작성되었습니다

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
 ```ptython
  #장르별 키워드도 잘 뽑힌 것 처럼 보이니 그냥 사용하겠다.
attributes = []
for tt in range(len(temp2)) :
    attr = []
    for i in range(min(30,len(temp2[tt]))) :
        attr.append(temp2[tt][i][0])
    attributes.append(attr)
matrix = [[0 for _ in range(len(genre_name))] for _ in range(len(genre_name))]
X = np.array([model.wv[word] for word in target_art])
Y = np.array([model.wv[word] for word in target_gen])

#마지막 히트맵 보기 힘들어서 그냥 다 구하는걸로 바꿨다
for i in range(len(genre_name)):
    for j in range(len(genre_name)):
        A = np.array([model.wv[word] for word in attributes[i]])
        B = np.array([model.wv[word] for word in attributes[j]])
        matrix[i][j] = weat_score(X, Y, A, B)
```
질문에 대답이 가능할 만큼 해당 주제에 대한 이해도가 있었고, 주석도 꼼꼼하게 달아주셨다.   

- [⭕] 5.코드가 간결한가요?
```python
#단어중복문제를 거의 해결했으니 그대로 사용하도록 하겠다.
target_art, target_gen = [], []
for i in range(100):
    target_art.append(sorted_word_ratios_art[i][0])
for i in range(100):
    target_gen.append(sorted_word_ratios_gen[i][0])
```
중복 코드는 생략해서 깔끔하게 작성해주셨다.

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
