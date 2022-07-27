import nltk
from sklearn.feature_extraction.text import CountVectorizer

#����gutenberg���Ͽ�
gb = nltk.corpus.gutenberg
#�����ĵ�
hamlet = gb.raw("shakespeare-hamlet.txt")
macbeth = gb.raw("shakespeare-macbeth.txt")

#ȥ��Ӣ��ͣ�ô�
cv = CountVectorzier(stop_words='english')
#��������������
print "Feature vector", cv.fit_transform([hamlet, macbeth]).toarray()
print "Features", cv.get_feature_names()[:5]
