from graphviz import Digraph
import numpy as np
import math
import time
import itertools

ai_dna = ['cnn', 'act', 'pool', 'batchnorm', 'lstm', 'flatten']
NEURAL_NUMBER = 5#10
tst_matrix = np.zeros(shape=(NEURAL_NUMBER, NEURAL_NUMBER), dtype=np.uint64)
alphabet = list(map(chr, range(97, 123)))
random_seed = int(time.time()) % 1000
matrix_name_lst = []

for idx in range(NEURAL_NUMBER):
    if idx == 0:
        digit = 1
    else:
        digit = int(math.log(idx, len(alphabet))) + 1
    name_vec = []
    tmp_data = idx
    for digit_idx in range(digit):
        alphabet_idx = int(tmp_data / len(alphabet) ** (digit - digit_idx - 1)) % len(alphabet)
        name = alphabet[alphabet_idx]
        name_vec.append(name)
        tmp_data -= len(alphabet) ** (digit - digit_idx - 1) * alphabet_idx
    matrix_name_lst.append(''.join(name_vec))

for count, name in enumerate(matrix_name_lst):
    matrix_name_lst[count] = "-".join([name, ai_dna[np.random.randint(low=0, high=len(ai_dna))]])

dna_dict = list(itertools.product(ai_dna, ai_dna))

#tst_matrix = np.triu(np.random.randint(2, size=tst_matrix.shape), k=1)
tst_matrix = np.array([[0,1,1,1,1], [1,0,1,0,0], [1,1,0,0,0], [1,0,0,0,1], [1,0,0,1,0]], dtype = np.uint64)

dot = Digraph('G', filename='tst.gv')
dot.attr(size='6,6')
dot.node_attr.update(color='lightblue2', style='filled')

#Assume row is start point, column is end point
for column_idx in range(NEURAL_NUMBER):
    root_flag = True
    for row_idx in range(NEURAL_NUMBER):
        if tst_matrix[row_idx, column_idx] == 1:
            label = dna_dict.index((matrix_name_lst[row_idx].split('-')[1],
                                    (matrix_name_lst[column_idx].split('-')[1])))
            dot.edge(matrix_name_lst[row_idx], matrix_name_lst[column_idx], label=str(label))
            tst_matrix[row_idx, column_idx] = label
            root_flag = False
    if column_idx > 0 and root_flag:#make sure there is only one root
        label = dna_dict.index((matrix_name_lst[column_idx - 1].split('-')[1],
                                (matrix_name_lst[column_idx].split('-')[1])))
        dot.edge(matrix_name_lst[column_idx - 1], matrix_name_lst[column_idx], label=str(label))
        tst_matrix[column_idx - 1, column_idx] = label

dot.view()
print(tst_matrix)

# A fix operation

'''
u = Digraph('unix', filename='unix.gv')
u.attr(size='6,6')
u.node_attr.update(color='lightblue2', style='filled')

u.edge('5th Edition', '6th Edition')
u.edge('5th Edition', 'PWB 1.0')
u.edge('6th Edition', 'LSX')
u.edge('6th Edition', '1 BSD')
u.edge('6th Edition', 'Mini Unix')
u.edge('6th Edition', 'Wollongong')
u.edge('6th Edition', 'Interdata')
u.edge('Interdata', 'Unix/TS 3.0')
u.edge('Interdata', 'PWB 2.0')
u.edge('Interdata', '7th Edition')
u.edge('7th Edition', '8th Edition')
u.edge('7th Edition', '32V')
u.edge('7th Edition', 'V7M')
u.edge('7th Edition', 'Ultrix-11')
u.edge('7th Edition', 'Xenix')
u.edge('7th Edition', 'UniPlus+')
u.edge('V7M', 'Ultrix-11')
u.edge('8th Edition', '9th Edition')
u.edge('1 BSD', '2 BSD')
u.edge('2 BSD', '2.8 BSD')
u.edge('2.8 BSD', 'Ultrix-11')
u.edge('2.8 BSD', '2.9 BSD')
u.edge('32V', '3 BSD')
u.edge('3 BSD', '4 BSD')
u.edge('4 BSD', '4.1 BSD')
u.edge('4.1 BSD', '4.2 BSD')
u.edge('4.1 BSD', '2.8 BSD')
u.edge('4.1 BSD', '8th Edition')
u.edge('4.2 BSD', '4.3 BSD')
u.edge('4.2 BSD', 'Ultrix-32')
u.edge('PWB 1.0', 'PWB 1.2')
u.edge('PWB 1.0', 'USG 1.0')
u.edge('PWB 1.2', 'PWB 2.0')
u.edge('USG 1.0', 'CB Unix 1')
u.edge('USG 1.0', 'USG 2.0')
u.edge('CB Unix 1', 'CB Unix 2')
u.edge('CB Unix 2', 'CB Unix 3')
u.edge('CB Unix 3', 'Unix/TS++')
u.edge('CB Unix 3', 'PDP-11 Sys V')
u.edge('USG 2.0', 'USG 3.0')
u.edge('USG 3.0', 'Unix/TS 3.0')
u.edge('PWB 2.0', 'Unix/TS 3.0')
u.edge('Unix/TS 1.0', 'Unix/TS 3.0')
u.edge('Unix/TS 3.0', 'TS 4.0')
u.edge('Unix/TS++', 'TS 4.0')
u.edge('CB Unix 3', 'TS 4.0')
u.edge('TS 4.0', 'System V.0')
u.edge('System V.0', 'System V.2')
u.edge('System V.2', 'System V.3')

u.view()
'''