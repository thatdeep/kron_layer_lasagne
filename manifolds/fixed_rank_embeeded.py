import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))

import warnings
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from manifolds.manifold import Manifold

import copy

import theano
from theano import tensor


class FixedRankEmbeeded(Manifold):
    """
    Manifold of m-by-n real matrices of fixed rank k. This follows the
    embedded geometry described in Bart Vandereycken's 2013 paper:
    "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    A point X on the manifold is represented as a structure with three
    fields: U, S and V. The matrices U (mxk) and V (kxn) are orthonormal,
    while the matrix S (kxk) is any /diagonal/, full rank matrix.
    Following the SVD formalism, X = U*S*V. Note that the diagonal entries
    of S are not constrained to be nonnegative.

    Tangent vectors are represented as a structure with three fields: Up, M
    and Vp. The matrices Up (mxn) and Vp (kxn) obey Up*U = 0 and Vp*V = 0.
    The matrix M (kxk) is arbitrary. Such a structure corresponds to the
    following tangent vector in the ambient space of mxn matrices:
      Z = U*M*V + Up*V + U*Vp
    where (U, S, V) is the current point and (Up, M, Vp) is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as mxn matrices. If
    these are low-rank, they may also be represented as structures with
    U, S, V fields, such that Z = U*S*V. Their are no resitrictions on what
    U, S and V are, as long as their product as indicated yields a real, mxn
    matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space R^(mxn) equipped with the usual trace (Frobenius) inner product.


    Please cite the Manopt paper as well as the research paper:
        @Article{vandereycken2013lowrank,
          Title   = {Low-rank matrix completion by {Riemannian} optimization},
          Author  = {Vandereycken, B.},
          Journal = {SIAM Journal on Optimization},
          Year    = {2013},
          Number  = {2},
          Pages   = {1214--1236},
          Volume  = {23},
          Doi     = {10.1137/110845768}
        }
    """
    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k
        #self.stiefelm = Stiefel(self._m, self._k)
        #self.stiefeln = Stiefel(self._n, self._k)
        self._name = ('Manifold of {:d}x{:d} matrices of rank {:d}'.format(m, n, k))

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return (self._m + self._n - self._k) * self._k

    @property
    def typicaldist(self):
        return self.dim

    def inner(self, X, G, H):
        return G.M.ravel().dot(H.M.ravel()) + \
               G.Up.ravel().dot(H.Up.ravel()) + \
               G.Vp.ravel().dot(H.Vp.ravel())

    def norm(self, X, G):
        return tensor.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        raise NotImplementedError

    def tangent(self, X, Z):
        # can we process Up and Vp in-place ?
        U, S, V = X
        Up, M, Vp = Z
        Up = Up - U.dot(U.T.dot(Up))
        Vp = Vp - (Vp.dot(V.T)).dot(V)
        return Up, M, Vp

    def apply_ambient(self, Z, W, type="mat"):
        if type == "man_elem":
            U, S, V = Z
            return U.dot(S.dot(V.dot(W)))
        elif type == "tan_vec":
            Up, M, Vp = Z
            return Up.dot(M.dot(Vp.dot(W)))
        elif type == "mat":
            return Z.dot(W)
        else:
            raise TypeError("'type' must be 'mat', 'man_elem' or 'tan_vec'")

    def apply_ambient_transpose(self, Z, W, type="mat"):
        if type == "man_elem":
            U, S, V = Z
            return V.T.dot(S.T.dot(U.T.dot(W)))
        elif type == "tan_vec":
            Up, M, Vp = Z
            return Vp.T.dot(M.T.dot(Up.T.dot(W)))
        elif type == "mat":
            return Z.T.dot(W)
        else:
            raise TypeError("'type' must be 'mat', 'man_elem' or 'tan_vec'")

    def proj(self, X, Z, type='mat'):
        U, S, V = X
        ZV = self.apply_ambient(Z, V.T, type=type)
        UtZV = U.T.dot(ZV)
        ZtU = self.apply_ambient_transpose(Z, U, type=type).T

        Zproj = (ZV - U.dot(UtZV), UtZV, ZtU - (UtZV.dot(V)))
        return Zproj

    def from_partial(self, X, dX):
        U, S, V = X
        dU, dS, dV = dX

        ZV = dU.dot(tensor.diag(1.0 / tensor.diag(S)))
        UtZV = dS
        ZtU = tensor.diag(1.0 / tensor.diag(S)).dot(dV)

        Zproj = (ZV - U.dot(UtZV), UtZV, ZtU - (UtZV.dot(V)))
        return Zproj

    def egrad2rgrad(self, X, Z):
        return self.proj(X, Z, type='mat')

    def ehess2rhess(self, X, egrad, ehess, H):
        # TODO same problem as tangent
        """
        # Euclidean part
        rhess = self.proj(X, ehess)
        Sinv = tensor.diag(1.0 / tensor.diag(X.S))

        # Curvature part
        T = self.apply_ambient(egrad, H.Vp.T).dot(Sinv)
        rhess.Up += (T - X.U.dot(X.U.T.dot(T)))
        T = self.apply_ambient_transpose(egrad, H.Up).dot(Sinv)
        rhess.Vp += (T - X.V.T.dot(X.V.dot(T))).T
        return rhess
        """
        raise NotImplementedError("method is not imlemented")

    def tangent2ambient(self, X, Z):
        XU, XS, XV = X
        ZUp, ZM, ZVp = Z
        U = tensor.stack((XU.dot(ZM) + ZUp, XU), 1).reshape((XU.shape[0], -1))
        #U = np.hstack((X.U.dot(Z.M) + Z.Up, X.U))
        S = tensor.eye(2*self._k)
        V = tensor.stack((XV, ZVp), 0).reshape((-1, XV.shape[1]))
        #V = np.vstack((X.V, Z.Vp))
        return (U, S, V)

    def retr(self, X, Z, t=None):
        U, S, V = X
        Up, M, Vp = Z
        if t is None:
            t = 1.0
        Qu, Ru = tensor.nlinalg.qr(Up)

        # we need rq decomposition here
        Qv, Rv = tensor.nlinalg.qr(Vp[::-1].T)
        Rv = Rv.T[::-1]
        Rv = Rv[:, ::-1]
        Qv = Qv.T[::-1]

        # now we have rq decomposition (Rv @ Qv = Z.Vp)
        #Rv, Qv = rq(Z.Vp, mode='economic')


        zero_block = tensor.zeros((Ru.shape[0], Rv.shape[1]))
        block_mat = tensor.stack(
            (
                tensor.stack((S + t * M, t * Rv), 1).reshape((Rv.shape[0], -1)),
                tensor.stack((t * Ru, zero_block), 1).reshape((Ru.shape[0], -1))
            )
        ).reshape((-1, Ru.shape[1] + Rv.shape[1]))

        Ut, St, Vt = tensor.nlinalg.svd(block_mat, full_matrices=False)

        U_res = tensor.stack((U, Qu), 1).reshape((Qu.shape[0], -1)).dot(Ut[:, :self._k])
        V_res = Vt[:self._k, :].dot(tensor.stack((V, Qv), 0).reshape((-1, Qv.shape[1])))
        # add some machinery eps to get a slightly perturbed element of a manifold
        # even if we have some zeros in S
        S_res = tensor.diag(St[:self._k]) + tensor.diag(np.spacing(1) * tensor.ones(self._k))
        return (U_res, S_res, V_res)

    def exp(self, X, U, t=None):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U, t)

    def np_rand(self, shape):
        X = np.random.randn(*shape)
        q, r = np.linalg.qr(X)
        return q

    def rand_np(self):
        U = self.np_rand((self._m, self._k))
        V = self.np_rand((self._n, self._k)).T
        #U = rnd.randn(self._m, self._k)
        #V = rnd.randn(self._k, self._n)
        S = rnd.randn(self._k,)
        S = np.diag(S / la.norm(S) + np.spacing(1) * np.ones(self._k))
        return (U, S, V)

    def rand(self):
        U = srnd.normal(size=(self._m, self._k))
        V = srnd.normal(size=(self._k, self._n))
        S = srnd.normal(size=(self._k,))
        S = tensor.diag(S / S.norm(L=2) + np.spacing(1) * tensor.ones(self._k))
        return (U, S, V)

    def randvec(self, X):
        H = self.rand()
        P = self.proj(X, H)
        return self._normalize(P)

    def zerovec(self, X):
        return (tensor.zeros((self._m, self._k)),
                tensor.zeros((self._k, self._k)),
                tensor.zeros((self._k, self._n)))

    def vec(self, X, Z):
        Zamb = self.tangent2ambient(X, Z)
        U, S, V = Zamb
        Zamb_mat = U.dot(S).dot(V)
        Zvec = Zamb_mat.T.ravel()
        return Zvec

    def _normalize(self, P):
        Up = P.Up
        M = P.M / tensor.nlinalg.norm(P.M)
        Vp = P.Vp
        return (Up, M, Vp)

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d), type='tan_vec')

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        Up1, M1, Vp1 = u1
        if u2 is None and a2 is None:
            Up = a1 * Up1
            Vp = a1 * Vp1
            M = a1 * M1
            return (Up, M, Vp)
        elif None not in [a1, u1, a2, u2]:
            Up2, M2, Vp2 = u2
            Up = a1 * Up1 + a2 * Up2
            Vp = a1 * Vp1 + a2 * Vp2
            M = a1 * M1 + a2 * M2
            return (Up, M, Vp)
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')


