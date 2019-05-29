import scipy as sp
import scipy.sparse as sparse
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize
import multiprocessing as mp
import cvxpy as cvx

__author__ = "shuaicheng zhang"
# python implementation of the AptRank algorithm 
# see https://github.rcac.purdue.edu/mgribsko/aptrank for the original matlab implementation
# full citation:
#Jiang, B., Kloster, K., Gleich, D. F., & Gribskov, M., Aptrank: an adaptive pagerank model for protein function prediction on bi-relational graphs, Bioinformatics, 33(12), 1829-1836 (2017).  http://dx.doi.org/10.1093/bioinformatics/btx029

class AptRank:

    def __init__(self, G, Rtrain, dH, K, S, T, NCores, diffusion_type):

        self.G = G
        self.Rtrain = Rtrain
        self.dH = dH
        self.K = K
        self.S = S
        self.T = T
        self.NCores = NCores
        self.diffusion_type = diffusion_type

        self.data_shape = Rtrain.shape
        self.xc = sp.zeros((self.K-1, self.S))

    def algorithm(self):

        # python version of pmm
        # Matrix multiplication using parallel computing.
        # Split columns of Z into pieces.
        def pmm(p, z):
            """
            parallel matrix multiplication to solve Ax=b
            *p*: matrix representing A
            *z*: matrix representing x
            """

            def par_func(num, outter):
                outter.put((sparse.csc_matrix.dot(p,zc[num]), num))
                #print("%dth worker has finished" % num)
                return

            ncores = self.NCores
            # number of columns
            n = np.shape(z)[1]
            # bin size
            bs = np.ceil(n / ncores)
            nblocks = int(np.ceil(n / bs))

            zc = np.empty(nblocks, dtype=object)
            # zc = []
            for i in np.arange(int(nblocks)):
                start = i * bs
                subset = np.arange(start, min(start + bs, n), dtype=int)
                extracted = z[:, subset]
                extracted = sparse.csc_matrix(extracted)
                # this an arbitrary threshold to only keep scores > 1e-4
                # helps the algorithm run faster
                #extracted.data[extracted.data < 0.0001] = 0
                extracted.eliminate_zeros()
                zc[i] = extracted

            tc = np.empty(nblocks, dtype=object)
            out_man = mp.Manager()
            out_queue = out_man.Queue()
            # args = [(tc, zc, p, n) for n in np.arange(0, nblocks)]
            # pool = multiprocessing.Pool(nblocks)
            args = np.arange(0, nblocks)
            processes = []
            for arg in args:
                processes.append(mp.Process(target=par_func, args=(arg, out_queue)))
            [x.start() for x in processes]
            [x.join() for x in processes]
            # pool.close()
            # pool.join()

            for j in range(nblocks):
                tc[j] = out_queue.get()

            tc = [x[0] for x in sorted(tc, key=lambda x: x[1])]
            t = sparse.hstack(tc)

            return t

        m, n = self.data_shape
        for i in np.arange(self.S):

            print("Shuffle for iteration %d" % i)
            Rfit, Reval = self.shuffleSplit()
            print("Split Rfit and Reval done")
            # splits = loadmat("%d_iter.mat" % (i + 1))
            # Rfit = splits['Rfit']
            # Reval = splits['Reval']

            p = []
            if "oneway" in self.diffusion_type:
                p = sparse.vstack((sparse.hstack((self.G, sparse.csc_matrix(sp.zeros(self.data_shape)))),
                                   sparse.hstack((Rfit.T, self.dH)))).tocsc()
            elif "twoway" in self.diffusion_type:
                p = sparse.vstack((sparse.hstack((self.G, Rfit)),
                                    sparse.hstack((Rfit.T, self.dH)))).tocsc()
            else:
                print("No diffusion type matched! Input oneway or twoway only!")
                exit(1)
            # python version normalizing
            p = normalize(p, norm='l1', axis=0)
            z = sparse.vstack((sparse.eye(m), sparse.csc_matrix((n, m), dtype=float))).tocsc()
            #a = np.empty((self.K - 1), dtype=object)
            a = []
            print("Col normalized Done ! Markov chain starts")
            print("running the pool function")
            for k in np.arange(0, self.K):
                # parallel matrix multiplication
                z = pmm(p, z)
                # store the lower block of z
                zh = z[m:z.shape[0], :].T

                # Do not save zh[0] since it is all-zeros.
                if k > 0:
                    print("\tstoring results for k %d" % (k))
                    #zh_arr = zh.toarray()
                    #a[k - 1] = np.reshape(zh_arr, (m*n, 1), order='F')
                    zh_arr = zh.reshape((m*n, 1), order='F')
                    #a = sparse.hstack([a, zh_arr])
                    a.append(zh_arr)
            print("Done")
            # a = np.concatenate(a)
            # a[0].ravel()
            a = sparse.hstack(a)
            print("Doing qr")
            # need sparse version
            qa, ra = np.linalg.qr(a.A)
            d = ra.shape[1]
            # Solving Adaptive PageRank coefficients using CVX
            print("Done with qr. Enter cvx solver python to finsih the convex optimization")
            x = cvx.Variable(d)
            expre = ra*x
            # expre_flat = np.array(expre).flatten()
            expre_2 = qa.T.dot(np.reshape(Reval.toarray(), (Reval.shape[0] * Reval.shape[1], 1), order='F'))
            expre_2 = np.reshape(expre_2, expre_2.shape[0])
            expre_2_cvx = cvx.Constant(expre_2)
            opt_prob = expre - expre_2_cvx
            objectives = cvx.Minimize(cvx.norm(opt_prob, 2))
            constrains = [np.ones((1, d))*x == 1, x >= 0]
            print("cvx solver has finished setting paramters")

            prob = cvx.Problem(objectives, constrains)
            result = prob.solve()
            print("cvx solver has finished its job")
            self.xc[:, i] = x.value
            print("finished iteration %d" % i)

        # compute prediction matrix Xa using the whole Rtrain
        xopt = np.median(self.xc, axis=1)
        xopt = xopt/np.sum(xopt)

        p = []
        if "oneway" in self.diffusion_type:
            p = sparse.vstack((sparse.hstack((self.G, sparse.csc_matrix(sp.zeros(self.data_shape)))),
                               sparse.hstack((self.Rtrain.T, self.dH)))).tocsc()
        elif "twoway" in self.diffusion_type:
            p = sparse.vstack((sparse.hstack((self.G, self.Rtrain)),
                               sparse.hstack((self.Rtrain.T, self.dH)))).tocsc()
        else:
            print("No diffusion type matched! Input oneway or twoway only!")
            exit(1)

        print("Compute prediction matrix Xa using the whole Rtrain.")
        # python version normalizing
        p = normalize(p, norm='l1', axis=0)
        z = sparse.vstack((sparse.eye(m), sparse.csc_matrix((n, m), dtype=float))).tocsc()
        #a = np.empty((self.K - 1), dtype=object)
        a = []

        for k in np.arange(0, self.K):
            z = pmm(p, z)
            zh = z[m:z.shape[0], :].T

            if k > 0:
                #zh_arr = zh.toarray()
                #a[k - 1] = np.reshape(zh_arr, (m * n, 1), order='F')
                zh_arr = zh.reshape((m*n, 1), order='F')
                a.append(zh_arr)

        #a = np.hstack(a)
        a = sparse.hstack(a)
        xa = a.dot(xopt)
        xa = xa.reshape((m, n), order='F')
        print("prediction for whole Rtrain is finished")

        return xa


    # The equivalent function for splitRT.m
    def shuffleSplit(self):

        if self.T < 0 or self.T > 1:
            print("T must be in [0,1]")
        m, n = self.data_shape
        ri, rj, rv = sparse.find(self.Rtrain)
        rlen = len(rv)
        a = np.random.permutation(rlen)
        nz = int(self.T * rlen)

        RR = sparse.csc_matrix((a, (ri, rj)), shape=(m, n), dtype=float)
        p = sp.zeros(nz)
        n_p = 0

        # matlab code for loop is inclusive
        for i in range(0, m):

            ir = RR[i, :]
            if ir.getnnz() > 0:
                mr = ir.data.min()
                if n_p < nz:
                    p[n_p] = mr
                    n_p = n_p + 1
                else:
                    break

        if n_p < nz:

            cp2 = np.setdiff1d(a, p)
            ip2 = np.random.permutation(len(cp2))
            # for randomly choosing nz-n_p amount of indices
            if len(cp2) > nz - n_p:
                dlen = len(cp2) - (nz - n_p)
                ip2 = np.delete(ip2, range(dlen))

            p2 = cp2[ip2].T
            p = p[p > 0]
            p = np.union1d(p, p2)
            p = np.unique(p)
            p = np.array(p, dtype=int)

        p = p.astype(int)
        cp = np.setdiff1d(a, p)

        Rtrain = sparse.csc_matrix((rv[p], (ri[p], rj[p])), shape=(m, n), dtype=float)
        Rtest = sparse.csc_matrix((rv[cp], (ri[cp], rj[cp])), shape=(m, n), dtype=float)

        # spliter = ShuffleSplit(n_splits=1, test_size=self.T, random_state=None)
        # indices = list(spliter.split(self.Rtrain))
        # train_indices = indices[0]
        # test_indices = indices[1]

        return Rtrain, Rtest
