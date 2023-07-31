import click
import scipy.stats as st
from pickle import load,loads,dump,dumps
import blosc
import torch
from esm.data import Alphabet
import numpy as np
import multiprocessing
from scipy.fftpack import dct, idct

def iDCTquant(v,n):
    f = dct(v.T, type=2, norm='ortho')
    trans = idct(f[:,:n], type=2, norm='ortho')
    for i in range(len(trans)):
        trans[i] = scale(trans[i])
    return trans.T

def scale(v):
    M = np.max(v)
    m = np.min(v)
    return (v - m) / float(M - m)

def quant2D(emb,n=5,m=44):
    dct = iDCTquant(emb[1:len(emb)-1],n)
    ddct = iDCTquant(dct.T,m).T
    ddct = ddct.reshape(n*m)
    return (ddct*127).astype('int8')


esm1b = torch.jit.load('traced_esm1b_25_13_cnt.pt').eval()
alphabet = Alphabet.from_architecture("ESM-1b")
batch_converter = alphabet.get_batch_converter()

for param in esm1b.parameters():
    param.grad = None
    param.requires_grad = False

if torch.cuda.is_available():
    esm1b = esm1b.cuda()
else:
    torch._C._jit_set_profiling_mode(False)
    torch.set_num_threads(multiprocessing.cpu_count())
    esm1b = torch.jit.freeze(esm1b)
    esm1b = torch.jit.optimize_for_inference(esm1b)

def _embed(seq):
    _, _, toks = batch_converter([("prot",seq)])
    if torch.cuda.is_available():
        toks = toks.to(device="cuda", non_blocking=True)
    results = esm1b(toks)
    for i in range(len(results)):
        results[i] = results[i].to(device="cpu")[0].detach().numpy()
    return results

from itertools import groupby
def fasta_iter(fastafile):
    fh = open(fastafile)
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq

def distMat2indList(mat,thr=0.1):
    math_thr = np.where(mat >= thr, 1, 0)

    # Remove self-referencing indices and duplicates
    filtered_indices = set()
    for index in np.argwhere(math_thr == 1):
        if index[0] != index[1]:
            filtered_indices.add((min(index)+1, max(index)+1))
    ret_lst = list(filtered_indices)
    ret_lst.sort()
    return ret_lst

@click.command()
@click.option('-s','--seq', help='Sequence itself or a FASTA file')
@click.option('-o','--out', default='out', help='Output file prefix')
@click.option('-t','--thr', default=0.02, help='Threshold for contact')
def embed(seq, out,thr):
    """A program that generates PROST embeddings and extracts contacts based on attention maps"""
    names = []
    seqs = []
    if '.fasta' in seq or '.fa' in seq:
        for f in fasta_iter(seq):
            names.append(f[0])
            seqs.append(f[1])
    else:
        seqs.append(seq)
        names.append('seq1')
    for i in range(len(seqs)):
        n = names[i]
        s = seqs[i]

        l13,l25,cnt = _embed(s)
        q25_544 = quant2D(l25,5,44)
        q13_385 = quant2D(l13,3,85)
        emb = np.concatenate([q25_544,q13_385])

        with open(out+'.%d.prdb'%i,'wb') as f:
            f.write(blosc.compress(dumps([np.array([n]), np.array([emb])])))

        cnt_lst = distMat2indList(cnt,float(thr))

        with open(out+'.%d.cnt.txt'%i,'w') as f:
            for c in cnt_lst:
                f.write('%d %d\n'%(c[0],c[1]))
    with open(out+'.names.txt','w') as f:
        for i in range(len(names)):
            f.write('%d\t%s\n'%(i, names[i]))

if __name__ == '__main__':
    embed()
