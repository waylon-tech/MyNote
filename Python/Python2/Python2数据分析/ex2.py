import nltk

#����Ӣ��ͣ��������
sw = set(nltk.corpus.stopwords.words('english'))
print "Stop words", list(sw)[:7]

#����Gutenberg���Ͽ�
gb = nltk.corpus.gutenberg
print "Gutenberg files", gb.files()[-5:]

#��ȡmilton-paradise.txt�ļ��е�ǰ��������
text_sent = gb.sents("milton-paradise.txt")[:2]
print "Unfiltered", text_sent

#���˵������ͣ����
for sent in text_sent:
    filtered = [w for w in sent if w.lower() not in sw]
    print "Filtered", filtered
    #����for���
    #��ȡ�ı��������ı�ǩ
    tagged = nltk.pos_tag(filetered)
    print "Tagged", tagged
			
    #ɾ���б��е����ֺ�����
    words= []
    for word in tagged:
    if word[1] != "NNP" and word[1] != 'CD':
    words.append(word[0])
