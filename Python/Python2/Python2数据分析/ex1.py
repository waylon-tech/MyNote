import nltk
from sklearn.feature_extraction.text import CountVectorizer

#加载gutenberg语料库
gb = nltk.corpus.gutenberg
#加载文档
hamlet = gb.raw("shakespeare-hamlet.txt")
macbeth = gb.raw("shakespeare-macbeth.txt")

#去掉英语停用词
cv = CountVectorzier(stop_words='english')
#并创建特征向量
print "Feature vector", cv.fit_transform([hamlet, macbeth]).toarray()
print "Features", cv.get_feature_names()[:5]
