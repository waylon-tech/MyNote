import urllib2
req = urllib2.urlopen(r'http://www.bilibili.com')
buf = req.read()
import re
urllist = re.findall(r'//.+\.jpg$', buf)
i = 0
for url in urllist:
    f = open(str(i)+'.jpg','w')
    req = urllib2.urlopen(url[2:])
    buf = req.read()
    f.write(buf)
    i+=1
print buf
print urllist
