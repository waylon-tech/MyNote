import networkx as nx
import matplotlib.pyplot as plt

#NetworkX�ṩ�����ʾ��ͼ�������г�
#print [s for s in dir(nx) if s.endswith('graph')]

G = nx.davis_southern_women_graph()
plt.figure(1)
plt.hist(nx.degree(G).values())

#���ƴ��нڵ��ǩ������ͼ
plt.figure(2)
pos = nx.spring_layout(G)
nx.draw(G, node_size=9)
nx.draw_networkx_labels(G, pos)
plt.show()

help(nx.spring_layout)
help(nx.draw)
help(nx.draw_networkx_labels)
