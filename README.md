# IIPL_start_toy_project

#### ==================== step1 : load data ====================
##### ==================== step1 : load data ====================


### train dataset
train = pd.read_csv(r"PATH\train.csv") #file path
### test dataset
test = pd.read_csv(r"PATH\test.csv") #file path


### ==================== step2 : data preprocessing ====================
### 중복제거
train = train.drop_duplicates()
test = test.drop_duplicates()

### 알파벳과 공백 제외, 제거
train['comment'] = [' '.join(re.sub('[^a-zA-Z .\']', '', word) for word in row.split()) for row in train['comment']]
test['comment'] = [' '.join(re.sub('[^a-zA-Z ]', '', word) for word in row.split()) for row in test['comment']]

### stopword 제거 (stopword_ nltk stopwords 중 일부 사용)
stopwords=['i', "i'm", "i've", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'then', 'once', 'there', 'when', 'where', 'why', 'how']
train['comment'] = [' '.join(word for word in row.split() if word.lower() not in stopwords) for row in train['comment']]
test['comment'] = [' '.join(word for word in row.split() if word.lower() not in stopwords) for row in test['comment']]

### ' 제거
train['comment'] = train['comment'].replace("'","", regex=True)
test['comment'] = test['comment'].replace("'","", regex=True)

### positive -> 1, negative -> -1
train['label']=0
train.loc[train.sentiment=='positive','label']=1
train.loc[train.sentiment=='negative','label']=0
test['label']=0
test.loc[test.sentiment=='positive','label']=1
test.loc[test.sentiment=='negative','label']=0


### ==================== step3 : tokenizing & padding ====================
train['comment_tok'] = [text_to_word_sequence(comment) for comment in train['comment']]
train = train[['comment', 'comment_tok', 'sentiment', 'label']]
test['comment_tok'] = [text_to_word_sequence(comment) for comment in test['comment']]
test = test[['comment', 'comment_tok', 'sentiment', 'label']]

### 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['comment_tok']) #text -> list (데이터 입력)

word_dict = tokenizer.word_index #단어사전  (높은 index = 등장 빈도가 낮은 단어)
words_num = len(word_dict) #전체 단어 개수

only_one = 0 # 1회 등장 단어 개수 
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
only_one_freq = 0 # 1회 등장 단어의 등장 빈도수 총 합
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    if(value < 2): #1회 등장 단어 필터링
        only_one = only_one + 1
        only_one_freq = only_one_freq + value
### # 전체 단어집합(103717개) 중 등장 빈도가 1번인 단어(51391개)-> 단어 집합에서 49.55%, 전체 등장 빈도에서 1.49% 차지

### #
vocab_size = words_num - only_one + 1
tokenizer = Tokenizer(vocab_size)  #단어집합의 최대 크기 제한
tokenizer.fit_on_texts(train['comment_tok'])
sequences_train = tokenizer.texts_to_sequences(train['comment_tok']) #text -> index(seq)
sequences_test = tokenizer.texts_to_sequences(test['comment_tok'])

words_median = statistics.median([len(row) for row in train['comment']]) #단어 개수 median
words_max = max([len(row) for row in train['comment']]) #단어 개수 max

### 패딩
train_input = pad_sequences(sequences_train, maxlen=int(words_median), padding='post', truncating='post') #zero padding
test_input = pad_sequences(sequences_test, maxlen=int(words_median), padding='post', truncating='post')

train_label = np.array(train['label']).reshape(-1, 1).astype('float32')
test_label = np.array(test['label']).reshape(-1, 1).astype('float32')


### ==================== step4 : classification ====================
### # LSTM model #

