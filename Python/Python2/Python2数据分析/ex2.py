import nltk

#加载英语停用字语料
sw = set(nltk.corpus.stopwords.words('english'))
print "Stop words", list(sw)[:7]

#加载Gutenberg语料库
gb = nltk.corpus.gutenberg
print "Gutenberg files", gb.files()[-5:]

#提取milton-paradise.txt文件中的前两句内容
text_sent = gb.sents("milton-paradise.txt")[:2]
print "Unfiltered", text_sent

#过滤掉下面的停用字
for sent in text_sent:
    filtered = [w for w in sent if w.lower() not in sw]
    print "Filtered", filtered
    #接上for语句
    #获取文本内所含的标签
    tagged = nltk.pos_tag(filetered)
    print "Tagged", tagged
			
    #删除列表中的数字和姓名
    words= []
    for word in tagged:
    if word[1] != "NNP" and word[1] != 'CD':
    words.append(word[0])
