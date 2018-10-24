from __future__ import absolute_import, division, print_function

import itertools

# These test cases were adapted from opt_einsum
# https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/tests/test_contract.py
# Copyright (c) 2014 Daniel Smith
EQUATIONS = [
    # Test hadamard-like products
    'a,ab,abc->abc',
    'a,b,ab->ab',

    # Test index-transformations
    'ea,fb,gc,hd,abcd->efgh',
    'ea,fb,abcd,gc,hd->efgh',
    'abcd,ea,fb,gc,hd->efgh',

    # Test complex contractions
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac->',
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac->',
    'cd,bdhe,aidb,hgca,gc,hgibcd,hgac->',
    'abhe,hidj,jgba,hiab,gab->',
    'bde,cdh,agdb,hica,ibd,hgicd,hiac->',
    'chd,bde,agbc,hiad,hgc,hgi,hiad->',
    'chd,bde,agbc,hiad,bdi,cgh,agdb->',
    'bdhe,acad,hiab,agac,hibd->',

    # Test collapse
    'ab,ab,c->',
    'ab,ab,c->c',
    'ab,ab,cd,cd->',
    'ab,ab,cd,cd->ac',
    'ab,ab,cd,cd->cd',
    'ab,ab,cd,cd,ef,ef->',

    # Test outer prodcuts
    'ab,cd,ef->abcdef',
    'ab,cd,ef->acdf',
    'ab,cd,de->abcde',
    'ab,cd,de->be',
    'ab,bcd,cd->abcd',
    'ab,bcd,cd->abd',

    # Random test cases that have previously failed
    'eb,cb,fb->cef',
    'dd,fb,be,cdb->cef',
    'bca,cdb,dbf,afc->',
    'dcc,fce,ea,dbf->ab',
    'fdf,cdd,ccd,afe->ae',
    'abcd,ad->bc',
    'ed,fcd,ff,bcf->be',
    'baa,dcf,af,cde->be',
    'bd,db,eac->ace',
    'fff,fae,bef,def->abd',
    'efc,dbc,acf,fd->abe',

    # Inner products
    'ab,ab->',
    'ab,ba->',
    'abc,abc->',
    'abc,bac->',
    'abc,cba->',

    # GEMM test cases
    'ab,bc->ac',
    'ab,cb->ac',
    'ba,bc->ac',
    'ba,cb->ac',
    'abcd,cd->ac',
    'abcd,ab->cd',
    'abcd,cdef->abef',
    'abcd,cdef->feba',
    'abcd,efdc->abef',

    # Inner than dot
    'aab,bc->ac',
    'ab,bcc->ac',
    'aab,bcc->ac',
    'baa,bcc->ac',
    'aab,ccb->ac',

    # Randomly build test caes
    'aab,fa,df,ecc->bde',
    'ecb,fef,bad,ed->ac',
    'bcf,bbb,fbf,fc->',
    'bb,ff,be->e',
    'bcb,bb,fc,fff->',
    'fbb,dfd,fc,fc->',
    'afd,ba,cc,dc->bf',
    'adb,bc,fa,cfc->d',
    'bbd,bda,fc,db->acf',
    'dba,ead,cad->bce',
    'aef,fbc,dca->bde',
]

# These test cases were added for Pyro
EQUATIONS.extend([
    '->',
    ',,,,->',
    'a->',
    'a->a',
    'a,a->a',
    ',a,a->a',
    'ij,jk,kl->il',
    'ea,fb,abcd,gc,hd->efgh',
])

# These are intended to be used by make_sahpes() below.
SIZES = [
    [2],
    [2, 3],
    [3, 2],
    [2, 3, 4],
    [2, 4, 3],
    [3, 2, 4],
    [3, 4, 2],
    [4, 2, 3],
    [4, 3, 2],
]


def make_shapes(equation, sizes):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    symbols = sorted(set(output).union(*inputs))
    sizes = {dim: size for dim, size in zip(symbols, itertools.cycle(sizes))}
    shapes = [tuple(sizes[dim] for dim in input_) for input_ in inputs]
    return shapes
