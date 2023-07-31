import numpy as np
import argparse
import os
import re
from collections import deque


BLOSUM62_MATRIX = [[4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
                 [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
                 [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
                 [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
                 [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
                 [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3],
                 [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
                 [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
                 [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
                 [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
                 [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
                 [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
                 [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
                 [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
                 [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
                 [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
                 [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
                 [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
                 [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
                 [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7]]

CHAR_INDEX = [0, -1, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10,
              11, -1, 12, 13, 14, 15, 16, -1, 17, 18, -1, 19, -1]

def blosum62(first, second):
    first = first.upper()
    second = second.upper()
    f_index = ord(first) - ord('A')
    s_index = ord(second) - ord('A')
    return BLOSUM62_MATRIX[CHAR_INDEX[f_index]][CHAR_INDEX[s_index]]

protsub_matrix = [[ 4,-1,-1, 0,-1,-1,-3, 0,-1, 1,-2,-1, 1, 0,-3,-2, 0,-2,-1,-1],
                     [ 4, 3, 3, 1, 0,-3,-4,-3,-2,-3,-3,-2,-1,-1,-3,-4,-3,-3,-4],
                        [ 6, 3, 2, 0,-2,-4,-3,-1,-4,-2,-3,-1,-1,-4,-3,-2,-3,-3],
                           [ 4, 1, 0,-3,-3,-3,-1,-3,-2,-2, 0,-1,-3,-2,-2,-3,-2],
                              [ 8, 1,-3,-3,-3,-1,-2, 0,-1,-1, 0,-3,-2,-1,-2,-2],
                                 [ 6, 1,-4,-4,-2,-3,-3,-2,-2, 3,-3,-4,-2,-1,-3],
                                    [13,-2,-4,-1,-4,-2,-3,-2, 3,-4,-3,-2,-2,-3],
                                       [ 9,-2,-3, 0,-2, 0,-2,-3,-1,-3,-2,-3,-2],
                                          [ 8,-3,-2,-1,-1,-1,-3,-1,-1,-2,-2,-1],
                                             [14,-3,-2,-1,-1,-2,-3,-3,-4,-3,-3],
                                                [ 7, 2, 1, 0,-2, 1,-1, 0, 1, 0],
                                                   [ 4, 0,-1, 0, 0, 1, 1, 0, 1],
                                                      [ 4, 2,-1, 1,-1, 0,-1, 0],
                                                         [ 7,-2,-2,-4,-1,-2,-1],
                                                            [ 8,-3,-2,-2, 2,-2],
                                                               [ 9, 3,-2, 1,-1],
                                                                  [ 4, 0, 0, 0],
                                                                     [ 6, 1, 2],
                                                                        [ 9,-1],
                                                                           [ 6]]
def protsub(first,second):
    res="AILVMFWGPCNQSTYDERHK"
    l = len(res)
    first = res.find(first.upper())
    second = res.find(second.upper())
    if second <= first:
        t = first
        first = second
        second = t
    return protsub_matrix[first][second-first]

def load_contacts(filename):
    cntc_file = open(filename, 'r')
    cntc_lines = cntc_file.readlines()
    cntc = np.zeros(len(cntc_lines)*2, dtype=int).reshape(len(cntc_lines), 2)
    for i in range(0,len(cntc_lines)):
        l = cntc_lines[i].split(" ")
        cntc[i][0] = int(l[0])
        cntc[i][1] = int(l[1])
    return cntc

def load_caomat(filename):
    res  = "ARNDCQEGHILKMFPSTWYV"
    cao_file = open(filename, 'r')
    cao_lines = cao_file.readlines()

    #should be alphabet length (26)
    caomat = np.zeros(26*26*26*26).reshape(26,26,26,26)
    caoval = np.array([])
    for i in range(0,len(cao_lines)):
        l = np.fromstring( cao_lines[i], dtype=float, sep=', ' )
        caoval = np.concatenate([caoval,l])
    z = 0
    for i in range(0,len(res)):
        a = ord(res[i]) - ord('A')
        for j in range(0,len(res)):
            b = ord(res[j]) - ord('A')
            for k in range(0,len(res)):
                c = ord(res[k]) - ord('A')
                for l in range(0,len(res)):
                    d = ord(res[l]) - ord('A')
                    caomat[a][b][c][d] = caoval[z] + 2.8
                    z += 1
    #non residue's are already zero, we begin with np.zeros
    return caomat


def calc_dpmat(caomat, seq1, seq2, cntc, wcao, matrix):
    h = len(seq1)
    w = len(seq2)
    dpmat = np.zeros(h*w).reshape(h, w)

    for i in range(0,h):
        for j in range(0,w):
            if seq1[i] == '.' or seq2[j] == '.':
                dpmat[i][j] = 0
            else:
                if matrix == 'blosum62':
                    dpmat[i][j] = blosum62(seq1[i],seq2[j])
                elif matrix == 'protsub': 
                    dpmat[i][j] = protsub(seq1[i],seq2[j])
                else:
                    print("Error: Unspecified matrix: " + matrix)
                    exit()
    #print("PAM Matrix")
    #print(dpmat)
    for con in cntc:
        cref0 = con[0]-1
        cref1 = con[1]-1
        rref0 = ord(seq1[cref0]) - ord('A')
        rref1 = ord(seq1[cref1]) - ord('A')
        for que in range(0,len(seq2)):
            cque0 = que
            cque1 = cque0 + con[1]-con[0]
            if cque0 < 0 or cque1 >= len(seq2):
                continue
            rque0 = ord(seq2[cque0]) - ord('A')
            rque1 = ord(seq2[cque1]) - ord('A')
            inc_val = wcao * caomat[rref0][rref1][rque0][rque1]
            #print(seq1[cref0] + seq1[cref1] + ":" + seq2[cque0] + seq2[cque1] + " = " + str(inc_val*10))
            dpmat[cref0][cque0] += inc_val 
            dpmat[cref1][cque1] += inc_val 
    return dpmat

best_score = -1
best_x = -1
best_y = -1
def update_best(matrix, y, x):
    global best_score
    global best_x
    global best_y
    if best_score < matrix[y][x]:
        best_score = matrix[y][x]
        best_x = x
        best_y = y

def local_align(dpmat,seq1,seq2,gap_open,gap_extend):
    h = len(seq1)
    w = len(seq2)
    matrix = np.zeros(h*w).reshape(h, w)
    max_x = np.zeros(w)
    max_y = 0
    
    for x in range(0,w):
        matrix[0][x] = max(dpmat[0][x], 0)
        max_x[x] = 0

        update_best(matrix,0,x)

    for y in range(1,h):
        max_y = 0
        matrix[y][0] = max(dpmat[y][0],0)

        update_best(matrix,y,x)

        for x in range(1,w):
            diag = matrix[y-1][x-1]
            matrix[y][x] = max(max(max_y,diag,max_x[x])+dpmat[y][x],0)
            diag -= gap_open
            max_y = max(diag,max_y) - gap_extend
            max_x[x] = max(diag,max_x[x]) - gap_extend

            update_best(matrix,y,x)

    return matrix

def global_align(dpmat,seq1,seq2,gap_open,gap_extend):
    global best_score
    h = len(seq1)
    w = len(seq2)
    matrix = np.zeros(h*w).reshape(h, w)
    max_x = np.zeros(w)
    max_y = 0

    for x in range(0,w):
        matrix[0][x] = dpmat[0][x]
        max_x[x] = -99999

    for y in range(1,h):
        max_y = -99999
        matrix[y][0] = dpmat[y][0]

        update_best(matrix,y,0)

        for x in range(1,w):
            diag = matrix[y-1][x-1]
            matrix[y][x] = max(max_y,diag,max_x[x])+dpmat[y][x]
            diag -= gap_open
            max_y = max(diag, max_y) - gap_extend
            max_x[x] = max(diag,max_x[x]) - gap_extend

    best_score = -99999
    for x in reversed(range(0,w)):
        update_best(matrix,h-1,x)
    for y in reversed(range(0,h)):
        update_best(matrix,y,w-1)

    return matrix

def traceback(matrix, dpmat, seq1, seq2, glb, gap_open, gap_extend):
    global best_score
    global best_x
    global best_y
    h = len(seq1)
    w = len(seq2)
    score = best_score
    x = best_x
    y = best_y
    gap = 0
    p0 = deque()
    p1 = deque()

    if glb:
        if w-x-1 > 0:
            gap = w-x-1
            p1.appendleft(seq2[len(seq2)-gap:len(seq2)])
            p0.appendleft('.'*gap)
        if h-y-1 > 0:
            gap = h-y-1
            p0.appendleft(seq1[len(seq1)-gap:len(seq1)])
            p1.appendleft('.'*gap)
    
    while x >= 0 and y >= 0:
        p0.appendleft(seq1[y])
        p1.appendleft(seq2[x])
        score = matrix[y][x] - dpmat[y][x]
        x -= 1
        y -= 1
        if not glb and np.isclose(score, 0):
            break;

        if x<0:
            gap = y+1
            y -= gap
            p0.appendleft(seq1[0:gap])
            p1.appendleft('.'*gap)
            continue

        if y<0:
            gap = x+1
            x -= gap
            p1.appendleft(seq2[0:gap])
            p0.appendleft('.'*gap)
            continue

        if not np.isclose(score, matrix[y][x]):
            gap = 1
            score += gap_open + gap_extend
            while True:
                if x >=gap and np.isclose(score, matrix[y][x-gap]):
                    x -= gap
                    p1.appendleft(seq2[x+1:x+1+gap])
                    p0.appendleft('.'*gap)
                    break
                if y>= gap and np.isclose(score, matrix[y-gap][x]):
                    y -= gap
                    p0.appendleft(seq1[y+1:y+1+gap])
                    p1.appendleft('.'*gap)
                    break


                gap += 1
                score += gap_extend
    return (score,p0,p1)

def read_fasta(filename):
    fasta = open(filename, 'r')
    seq = ''
    start = False
    for l in fasta.readlines():
        if l.startswith('>'):
            if start:
                break
            else:
                start = True
                continue
        if start:
            seq = seq + l.strip()
    return seq

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s1", help="Query sequence")
parser.add_argument("-c1", help="Contact map of query seqeunce")
parser.add_argument("-s2", help="Target sequence")
parser.add_argument("-o", help="Gap open score", default=14.0)
parser.add_argument("-e", help="Gap extend score", default=1.0)
parser.add_argument("-g", help="Global alignment, default local", action='store_true')
parser.add_argument("-c", help="CAO matrix, default cao120", default="cao120")
parser.add_argument("-w", help="CAO matrix weight", default=0.1)
parser.add_argument("-m", help="Matrix [blosum62,protsub]", default='blosum62')
args = parser.parse_args()

gap_open = float(args.o)
gap_extend = float(args.e)

if '.fasta' in args.s1: seq1 = read_fasta(args.s1)
else: seq1 = args.s1
if '.fasta' in args.s2: seq2 = read_fasta(args.s2)
else: seq2 = args.s2


print(seq1)
print(seq2)
cntc = load_contacts(args.c1)
caomat = load_caomat(args.c)
dpmat = calc_dpmat(caomat, seq1, seq2, cntc, float(args.w), args.m)
if args.g:
    matrix = global_align(dpmat,seq1,seq2,gap_open,gap_extend)
else:
    matrix = local_align(dpmat,seq1,seq2,gap_open,gap_extend)
(score,p0,p1) = traceback(matrix,dpmat,seq1,seq2,args.g,gap_open,gap_extend)

seq1 = ''.join(p0)
seq2 = ''.join(p1)

#identity calculation
idy = ''
idc = 0
gapc = 0
simc = 0
for i in range(len(seq1)):
    if seq1[i] == '.' or seq2[i] == '.':
        idy = idy + ' '
        gapc += 1
        continue
    if seq1[i] == seq2[i]:
        idy = idy + '|'
        idc += 1
    else:
        score = 0
        if args.m == 'blosum62':
            score = blosum62(seq1[i],seq2[i])
        elif args.m == 'protsub':
            score = protsub(seq1[i],seq2[i])
        #print(seq1[i]+":"+seq2[i]+"="+str(score))
        if score >= 1:
            idy = idy + ':'
        else:
            idy = idy + '.'
        simc += 1

l = len(idy)
print("score %.2f, w:%s matrix:%s, gap open:%.2f gap extend %.2f"%
        (best_score,args.w,args.m,gap_open, gap_extend))
print("Length %d, Identity: %d/%d (%.2f%%), Similarity: %d/%d (%.2f%%), gaps: %d/%d (%.2f%%)" %
        (l, idc, l, idc/l*100, simc,l,simc/l*100,gapc,l,gapc/l*100))
print(seq1)
print(idy)
print(seq2)

