### Embeding & Attention Correlation 

* `embed.py` embeds given sequence and also creates attention based contact map with given threshold. This threshold is a score threshold and ranges between 0 and 1. The higher the score the more corelated the residues are. 0.02 and above is a good indicator of correlation.
* It generates two files `out.prdb` and `out.cnt.txt`. The `prdb` can be used with PROST to search a database like this `prost searchsp --thr 0.05 out.prdb results.tsv`
* ESM1b only accepts sequences upto 1022 aa length. Due to this contact map generation is limited to lenght of 1022 aa.
* If a fasta file wiht multiple sequences given, `embed.py` will generate a set of `[$(out).$(i).prdb, $(out).$(i).cnt.txt]` files for each sequence and a name mapping file `$(out).names.txt`
* Use `embed.py` to embed sequences in one go by giving multiple sequences with a fasta file because loading the language model will take at least 30sec. 
If you embed one by one, you'll have an overhead of 30sec for each.
```
Usage: embed.py [OPTIONS]

  A program that generates PROST embeddings and extracts contacts based on
  attention maps

Options:
  -s, --seq TEXT   Sequence itself or a FASTA file
  -o, --out TEXT   Output file prefix
  -t, --thr FLOAT  Threshold for contact
  --help           Show this message and exit.

python embed.py  -s YLGLPVTPEDMALLKEQLFAELAILPENTRINKVGENSFQIWVASENVKNQITETYPSGQITLSNAVTKVEFIFGD -t 0.02
```

### Alignment with Contact Map and 400x400 matrix

* `align.py` takes two sequences `s1` and `s2` and a contact map `c1` to produce contact based alignment.
* The 400x400 matrix can be selected with `-c` option.
* The score is calcualted like this `w*(400x400matrixScore) + (20x20matrixScore)`.
* 20x20 matrix can be selected via `-m` switch.
* The weight of 400x400 matrix can be selected with `-w` option.

```
usage: align.py [-h] [-s1 S1] [-c1 C1] [-s2 S2] [-o O] [-e E] [-g] [-c C] [-w W] [-m M]

optional arguments:
  -h, --help  show this help message and exit
  -s1 S1      Query sequence
  -c1 C1      Contact map of query seqeunce
  -s2 S2      Target sequence
  -o O        Gap open score
  -e E        Gap extend score
  -g          Global alignment, default local
  -c C        CAO matrix, default cao120
  -w W        CAO matrix weight
  -m M        Matrix [blosum62,protsub]

python align.py -s1 YLGLPVTPEDMALLKEQLFAELAILPENTRINKVGENSFQIWVASENVKNQITETYPSGQITLSNAVTKVEFIFGD -s2 FSGNCTMEDAKLAQDFLDSQNLSAYNTRLFKEVDGEGKPYYEVRLASVLGSESEVTSKLKSYEFRGSPFQVTRGD -c1 out.cnt.txt -g -w 0.1 -m protsub -o 10.5 -e 0.5
```

`3csk.cont.txt` is the real contact map from the PDB and produces this alignment: 
```
score 498.68, dcut:10 scut:0, pcut:0 w:0.1 matrix:protsub, gap open:10.50 gap extend 0.50
Length 78, Identity: 18/78 (23.08%), Similarity: 55/78 (70.51%), gaps: 5/78 (6.41%)
YLGLPVTPEDMALLKEQLFAELAILPENTRINKV..GENSFQIWVASENVKNQITETYPSGQITLSNAVTKVEFIFGD
:.| ..|.||..|.::.|.:: .:...|||:.|.  ||......|...:|....:| ..|...:.....:..:...||
FSG.NCTMEDAKLAQDFLDSQ.NLSAYNTRLFKEVDGEGKPYYEVRLASVLGSESE.VTSKLKSYEFRGSPFQVTRGD
```

while `out.cont.txt` is the contact map generated with `ESM1b` and `0.02` threshold and it also produces the same result:
```
score 248.83, w:0.1 matrix:protsub, gap open:10.50 gap extend 0.50
Length 78, Identity: 18/78 (23.08%), Similarity: 55/78 (70.51%), gaps: 5/78 (6.41%)
YLGLPVTPEDMALLKEQLFAELAILPENTRINKV..GENSFQIWVASENVKNQITETYPSGQITLSNAVTKVEFIFGD
:.| ..|.||..|.::.|.:: .:...|||:.|.  ||......|...:|....:|.. |...:.....:..:...||
FSG.NCTMEDAKLAQDFLDSQ.NLSAYNTRLFKEVDGEGKPYYEVRLASVLGSESEVT.SKLKSYEFRGSPFQVTRGD

```
These examples sequences are from Protsub paper. `3csk` and `3fvy`.
