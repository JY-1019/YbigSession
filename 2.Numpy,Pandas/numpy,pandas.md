# Numpy

## 1. Numpy란
- Numpy는 C언어로 구현된 파이썬 라이브러리로써, 고성능 수치계산을 위해 제작.
- C언어 기반으로 빠른 속도를 자랑
- Python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리


## 2. Numpy 기초


### 1) Array 정의 및 사용
- Numpy 배열 생성
```py
array = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12]])
array

# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
```
- shape/dim/size

    shape : 배열의 shape를 반환

    dim : 배열의 차원 수를 반환

    size : 원소의 총 갯수 반환
```py
print(array.shape)
print(array.ndim)
print(array.size)

# (4, 3)
# 2
# 12
```
- dtype

    배열의 자료형을 알 수 있음
    (numpy의 ndarray는 모든 원소가 같은 자료형)

```py
print(array.dtype)

# int64
```

* numpy에서 사용되는 자료형

    부호가 있는 정수 : int(8, 16, 32, 64)
    부호가 없는 정수 : uint(8 ,16, 32, 54)
    실수 : float(16, 32, 64, 128)
    complex : (64, 128, 256)
    불리언 : bool
    문자열 : string
    파이썬 : 오프젝트 object
    유니코드 : unicode

- zeros

    zeros는 입력받는 형태만큼 인자가 0으로 채워진 행렬을 만든다
```py
np.zeros(10)

# array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

np.zeros((3,5))

# array([[0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.]])
```
- ones

    ones는 입력받는 형태만큼 인자가 1으로 채워진 행렬을 만든다

```py
np.ones((3,5))


# array([[1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1.],
#        [1., 1., 1., 1., 1.]])
```

- arange

    arange는 파이썬 range와 비슷한 개념

    한자리수(n)만 입력하면 0부터 n-1까지 1씩 늘어남

    3번째 자리에 늘어나는 간격을 설정하여 사용 가능

```py
np.arange(1,10)

# array([0, 2, 4, 6, 8])

np.arange(0,9,2)

# array([0, 2, 4, 6, 8])
```

- reshape
    
    재배열

    reshape(-1) : 원래 배열의 길이와 남은 차원으로부터 추정

```py
array = np.arange(8)
array

# array([0, 1, 2, 3, 4, 5, 6, 7])

array_reshaped = array.reshape(2, 4)
array_reshaped

# array([[0, 1, 2, 3],
#        [4, 5, 6, 7]])

array_reshaped = array.reshape(-1,4)
array_reshaped

# array([[0, 1, 2, 3],
#        [4, 5, 6, 7]])
```

### 2) Array 연산

- 예제 array 생성
```py
array1_ = np.array([[1,2,3],[4,5,6]])
array1_

# array([[1, 2, 3],
#        [4, 5, 6]])
array2_ = np.array([[10,11,12],[13,14,15]])
array2_

# array([[10, 11, 12],
#        [13, 14, 15]])
```

- 평균
```py
np.mean(array1_)

# 3.5
```
- 덧셈
```py
array1_ + array2_

np.add(array1_ ,array2_)

# array([[11, 13, 15],
#        [17, 19, 21]])
```

- 뺄셈
```py
array1_ - array2_
np.subtract(array1_, array2_)

# array([[-9, -9, -9],
#        [-9, -9, -9]])
```

- 곱셈
```py
array1_ * array2_

np.multiply(array1_, array2_)

# array([[10, 22, 36],
#        [52, 70, 90]])
```

- 행렬 곱
```py
a = np.arange(1, 11)
b = np.arange(11, 21)

print(a.dot(b))

# 935
```
- 나눗셈
```py
np.divide(array1_ , array2_)

# array([[10, 11, 12],
#        [13, 14, 15]])
```

- broadcasting

    shape가 다를 때도 연산이 가능함.

```py
array1_

# array([[1, 2, 3],
#        [4, 5, 6]])

array3_ = np.array([10, 11, 12])
array3_

# array([10, 11, 12])

array1_ + array3_

# array([[11, 13, 15],
#        [14, 16, 18]])

array1_ * array3_

# array([[10, 22, 36],
#        [40, 55, 72]])
```

### 3) Array Indexing
- 기본 인덱싱
```py
arr2 = np.array([[1,2,3,4],
                 [5,6,7,8],
                 [9,10,11,12]])
```
```py
# 2차원의 array를 인덱싱을 하기 위해선 2개의 인자 입력
# [행#, 열#]
print(arr2[0,0])
print(arr2[0][0])

# 1
# 1
--------------------------
# 2번째 array의 전체 불러오기
arr2[2,:]

# array([ 9, 10, 11, 12])
--------------------------
# 2번째 array의 3번째 값 불러오기
arr2[2,3]

# array([ 4,  8, 12])
--------------------------
# 첫번째, 세번째 요소를 인덱싱 함.
# (열을 indexing 하는게 아니라는 점에서 pandas와 다름)
arr2[[0,2]]

# array([[ 1,  2,  3,  4],
#        [ 9, 10, 11, 12]])
--------------------------
arr2[[0,2],-1]

# array([ 4, 12])
```
- boolean을 이용한 인덱싱
```py
# 조건에 맞는 (= 필터링된) 값 불러오기
arr1[arr1 > 5]

# array([6, 7, 8, 9])
```
```py
# 조건에 맞는 값들의 인덱스값만 반환
np.where(arr1 > 5)

# (array([6, 7, 8, 9]),)
```
```py
np.where(arr1 > 5, 'O', 'X') 


# array(['X', 'X', 'X', 'X', 'X', 'X', 'O', 'O', 'O', 'O'], dtype='<U1')
```
### 4) 자주 쓰이는 Numpy 함수
- concatenate-결합
```py
array1_

# array([[1, 2, 3],
#        [4, 5, 6]])

array2_

# array([[10, 11, 12],
#        [13, 14, 15]])
```
```py
# 행을 기준으로 결합
array_concat = np.concatenate((array1_, array2_))

print(array_concat)

# [[ 1  2  3]
#  [ 4  5  6]
#  [10 11 12]
#  [13 14 15]]
```
```py
# 열을 기준으로 결합
array_con1 = np.concatenate((array1_, array2_), axis=1)

print(array_con1)

# [[ 1  2  3 10 11 12]
#  [ 4  5  6 13 14 15]]
```

- split 분할
```py
array1, array2 = np.split(array_concat,2) 

print(array1)
print(array2)

# [[1 2 3]
#  [4 5 6]]
# [[10 11 12]
#  [13 14 15]]
```
# Pandas
## 1. Pandas란?
파이썬에서 사용하는 데이터분석 라이브러리

행과 열로 이루어진 데이터 객체를 만들 수 있음
## 2. Pandas 기초
### 1) Seires
- series 생성
```py
series1_ = pd.Series([1, 6, 7, 3], index=["A","B","C","D"])

series1_

# A    1
# B    6
# C    7
# D    3
# dtype: int64

series2 = pd.Series({'A' : 90, 'B' : 80, 'C' : 70, 'D': 60})

series2

# A    90
# B    80
# C    70
# D    60
# dtype: int64
```
- data 확인 메소드
```py
series1.values

# 값만 확인
# array([1, 6, 7, 3])

series1.dtypes

# 자료형 확인
# dtype('int64')
```
- series 이름, 인덱스 이름 설정 가능
```py
series2.name = '학점 기준'
series2.index.name = '학점'
series2


# 학점
# A    90
# B    80
# C    70
# D    60
# Name: 학점 기준, dtype: int64
```
- Series 연산은 numpy와 거의 비슷함. 그러나 인덱스를 기준으로 연산. 
```py
#인덱스가 다른 두 series
series1 + series2

# 인데스가 다르면 NaN으로 출력
# A   NaN
# B   NaN
# C   NaN
# D   NaN
# W   NaN
# X   NaN
# Y   NaN
# Z   NaN
# dtype: float64
```
### 2) DataFrame
- 데이터 프레임 생성 및 데이터 삽입
```py
df1 = pd.DataFrame(columns=["Food", "Price"])

df1["Food"] = ["마라샹궈", "마라탕","아이스크림","아이셔"]
df1["Price"] = [15000, 8000,1000,"판매종료"]
df1

# 	Food	Price
# 0	마라샹궈	15000
# 1	마라탕	8000
# 2	아이스크림	1000
# 3	아이셔	판매종료
```
- 딕셔너리로 데이터 프레임 생성
```py
# 딕셔너리로 생성
food = ["꿔바로우", "순대국밥"]
price = [16000, 8000]
dic = {"Food": food, "Price": price}
df2 = pd.DataFrame(dic, index=['A','B'])

df2

# Food	Price
# A	꿔바로우	16000
# B	순대국밥	8000
```
- column을 index로 설정(set_index)
```py
df2.set_index("Food")

#        Price	
# Food   	
# 꿔바로우  16000
# 순대국밥  8000
```
- 데이터 프레임 인덱싱

    df2:

    ||Food|Price|
    |--|--|--|
    |A|꿔바로우|16000|
    |B|순대국밥|8000|

칼럼으로 접근

```py
df2['Food']

# A    꿔바로우
# B    순대국밥
# Name: Food, dtype: object

df2["A"] -> Error  # 데이터프레임은 index로 접근이 불가능함
```
  슬라이싱으로 접근

```py
df2["A":"A"]

#     Food	Price
# A	꿔바로우	16000
```
loc를 통해 index 접근
```py
df2.loc["A"]

# Food      꿔바로우
# Price    16000
# Name: A, dtype: object
```
iloc를 통해 접근
```py
df2.iloc[0]

# Food      꿔바로우
# Price    16000
# Name: A, dtype: object
```
조건으로 접근
```py
df2[df2.Price > 10000]

#     Food	Price
# A	꿔바로우	16000
```

- 데이터 프레임 추가, 삭제

    df1: 

    ||Food|Price|
    |--|--|--|
    |0|마라샹궈|15000|
    |1|마라탕|8000|
    |2|아이스크림|1000|
    |3|아이셔|판매종료|

loc를 이용해 추가, 병경 가능
```py
df1.loc[3] = ["꿔바로우", "12000"]
df1

#   Food	Price
# 0	마라샹궈	15000
# 1	마라탕	8000
# 2	아이스크림	1000
# 3	꿔바로우	12000

df1.loc[4] = ["볶음밥", "7000"]
df1

# 	Food	Price
# 0	마라샹궈	15000
# 1	마라탕	8000
# 2	아이스크림	1000
# 3	꿔바로우	12000
# 4	볶음밥	7000
```
Row 삭제
```py
df1_ = df1.drop(2)
df1_

# 	Food	Price
# 0	마라샹궈	15000
# 1	마라탕	8000
# 3	꿔바로우	12000
# 4	볶음밥	7000
```
Column 삭제
```py
df1__.drop("index", axis=1)


#     Food	Price
# 0	마라샹궈	15000
# 1	아이스크림	1200
# 2	꿔바로우	12000
# 3	볶음밥	7000
```
- 인덱스 재배정
```py
df1__.drop("index", axis=1)


#     Food	Price
# 0	마라샹궈	15000
# 1	아이스크림	1200
# 2	꿔바로우	12000
# 3	볶음밥	7000
```
- Concat

    df1:
     ||Food|Price|
    |--|--|--|
    |0|마라샹궈|15000|
    |1|아이스크림|1200|
    |2|꿔바로우|12000|
    |3|볶음밥|7000|

    df2:
    ||Food|Price|
    |--|--|--|
    |0|꿔바로우|16000|
    |1|순대국밥|8000|
```py
# default는 axis=0
df3 = pd.concat([df1, df2])
df3 

#     Food	Price
# 0	마라샹궈	15000
# 1	아이스크림	1200
# 2	꿔바로우	12000
# 3	볶음밥	7000
# A	꿔바로우	16000
# B	순대국밥	8000
```
```py
# axis=1로 수평축으로
pd.concat([df3,df2], axis=1)


#     Food	Price	Food	Price
# 0	마라샹궈	15000	NaN	    NaN
# 1	아이스크림	1200	NaN	    NaN
# 2	꿔바로우	12000	NaN 	NaN
# 3	볶음밥	7000	NaN	    NaN
# A	꿔바로우	16000	꿔바로우	16000.0
# B	순대국밥	8000	순대국밥	8000.0

# NAN 해결방법
# default는 outer로 되어있음
pd.concat([df3,df2], axis=1, join='inner')

# 	Food	Price	Food	Price
# A	꿔바로우	16000	꿔바로우	16000
# B	순대국밥	8000	순대국밥	8000
```
- merge

    df3:
     ||Food|Price|
    |--|--|--|
    |0|마라샹궈|15000|
    |1|아이스크림|1200|
    |2|꿔바로우|12000|
    |3|볶음밥|7000|
    |A|꿔바로우|16000|
    |B|순대국밥|8000|

    df2:
    ||Food|Price|
    |--|--|--|
    |0|꿔바로우|16000|
    |1|순대국밥|8000|

inner로 하면 key값이 둘 다 있는 것으로 merge
```py
pd.merge(df3, df5, how='inner', on='Food')

# 	Food	Price	Country
# 0	마라샹궈	15000	중국
# 1	순대국밥	8000	한국
```
outer로 하면 key값이 한 데이터프레임에만 있어도 병합
```py
pd.merge(df3, df5, how='outer', on='Food')

#     Food	Price	Country
# 0	마라샹궈	15000	중국
# 1	아이스크림	1200	NaN
# 2	꿔바로우	12000	NaN
# 3	꿔바로우	16000	NaN
# 4	볶음밥	7000	NaN
# 5	순대국밥	8000	한국
# 6	초밥	NaN	일본
# 7	부대찌개	NaN	한국
```
left로 하면 왼쪽에 있는 데이터 프레임의 'on' 기준
```py
pd.merge(df3, df5, how='left', on='Food')

# 	Food	Price	Country
# 0	마라샹궈	15000	중국
# 1	아이스크림	1200	NaN
# 2	꿔바로우	12000	NaN
# 3	볶음밥	7000	NaN
# 4	꿔바로우	16000	NaN
# 5	순대국밥	8000	한국
```
right로 하면 오른쪽에 있는 데이터 프레임의 'on' 기준
```py
pd.merge(df3, df5, how='right', on='Food')


#     Food	Price	Country
# 0	마라샹궈	15000	중국
# 1	초밥	NaN	일본
# 2	부대찌개	NaN	한국
# 3	순대국밥	8000	한국    
```
- apply

    함수를 데이터프레임에 적용할 수 있음.
```py
def is_adult(age):
    if age < 20 :
        return 'not adult'
    if age >= 20 :
        return 'adult'

titanic['is_adult'] = titanic['Age'].apply(is_adult)
titanic.tail()

# lambda로 적용가능. 더 간단해서 자주 사용
titanic['is_adult'] = titanic['Age'].apply(lambda x: 'adult' if x < 20 else 'not_adult')
```
- sort
    원하는 칼럼을 정렬
```py
# 원하는 칼럼을 선택해 그것을 기준으로 오름차순 정렬: .sort_values()
titanic.sort_values('Pclass').head()

# ascending=False로 하면 내림차순
titanic.sort_values('Pclass', ascending=False).head()
```
- null 값 확인
```py
# info()로도 확인 가능
titanic.info()

# null 값 확인: .isnull()
titanic[titanic["Age"].isnull()]

# null값 세기: .sum()
titanic.isnull().sum()

# null 값 아닌 것 확인: .notnull()
titanic[titanic["Age"].notnull()]
```
- dropna
```py
# 결측치 제거
titanic_notnull = titanic.dropna()

titanic[titanic['Age'].isnull()].head()

# 결측치 수 세기
titanic_notnull.isnull().sum()

# .dropna(axis=1) 하면 null 값인 column 자체를 다 삭제할 수 있음.
titanic_notnull = titanic.dropna(axis = 1)
```
- unique
```py
# embarked column에서 unique한 값을 출력
titanic.Embarked.unique()
```
- value_counts
```py
# 각 unique값들이 얼마나 있는지 value값의 count
titanic.Survived.value_counts()
```