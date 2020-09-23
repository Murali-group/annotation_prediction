
# Script to setup network and annotations files as sparse matrices
# also used for weighting the networks with the
# Simultaneous Weight with Specific Negatives (SWSN) method

from collections import defaultdict
import os, sys, time
from optparse import OptionParser, OptionGroup
#import src.version_settings as v_settings
import src.utils.file_utils as utils
from src.utils.string_utils import full_column_names, \
    STRING_NETWORKS, NON_TRANSFERRED_STRING_NETWORKS, \
    CORE_STRING_NETWORKS
import src.algorithms.alg_utils as alg_utils
import src.go_term_prediction_examples.go_term_prediction_examples as go_examples
from src.weight_networks.findKernelWeights import findKernelWeights
from src.weight_networks.combineNetworksSWSN import combineNetworksSWSN
from networkx.algorithms.dag import descendants
import networkx as nx
import numpy as np
from scipy.io import savemat, loadmat
from scipy import sparse as sp
from tqdm import tqdm
import gzip
# this warning prints out a lot when normalizing the networks due to nodes having no edges.
# RuntimeWarning: divide by zero encountered in true_divide
# Ignore it for now
import warnings
warnings.simplefilter('ignore', RuntimeWarning)


class Sparse_Networks:
    """
    An object to hold the sparse network (or sparse networks if they are to be joined later), 
        the list of nodes giving the index of each node, and a mapping from node name to index
    *sparse_networks*: either a list of scipy sparse matrices, or a single sparse matrix
    *weight_method*: method to combine the networks if multiple sparse networks are given.
        Possible values: 'swsn', or 'gmw'
        'swsn': Simultaneous Weighting with Specific Negatives (all terms)
        'gmw': GeneMANIA Weighting (term-by-term). Also called gm2008
    *unweighted*: set the edge weights to 1 for all given networks.
    *term_weights*: a dictionary of tuples containing the weights and indices to use for each term.
        Would be used instead of running 'gmw'
    """
    def __init__(self, sparse_networks, nodes, net_names=None,
                 weight_method='swsn', unweighted=False, term_weights=None, verbose=False):
        self.multi_net = False
        if isinstance(sparse_networks, list):
            if len(sparse_networks) > 1:
                self.sparse_networks = sparse_networks
                self.multi_net = True
            else:
                self.W = sparse_networks[0]
        else:
            self.W = sparse_networks
        self.nodes = nodes
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(nodes)}
        self.net_names = net_names
        self.weight_method = weight_method
        self.unweighted = unweighted
        self.verbose = verbose
        # make sure the values are correct
        if self.multi_net is True:
            self.weight_swsn = True if weight_method.lower() == 'swsn' else False
            self.weight_gmw = True if weight_method.lower() in ['gmw', 'gm2008'] else False
            num_weight_methods = sum([self.weight_swsn, self.weight_gmw])
            if num_weight_methods == 0 or num_weight_methods > 1:
                raise("must specify exactly one method to combine networks when multiple networks are passed in. Given method: '%s'" % (weight_method))
            self.term_weights = term_weights
        else:
            self.weight_swsn = False
            self.weight_gmw = False

        # set a weight str for writing output files
        self.weight_str = '%s%s%s' % (
            '-unw' if self.unweighted else '', 
            '-gmw' if self.weight_gmw else '',
            '-swsn' if self.weight_swsn else '')

        if self.unweighted is True:
            print("\tsetting all edge weights to 1 (unweighted)")
            if self.multi_net is False:
                # convert all of the entries to 1s to "unweight" the network
                self.W = (self.W > 0).astype(int) 
            else:
                new_sparse_networks = []
                for i in range(len(self.sparse_networks)):
                    net = self.sparse_networks[i]
                    # convert all of the entries to 1s to "unweight" the network
                    net = (net > 0).astype(int) 
                    new_sparse_networks.append(net)
                self.sparse_networks = new_sparse_networks

        if self.multi_net is True:
            print("\tnormalizing the networks")
            self.normalized_nets = []
            for net in self.sparse_networks:
                self.normalized_nets.append(_net_normalize(net))

    def weight_SWSN(self, ann_matrix):
        self.W, self.swsn_time, self.swsn_weights = weight_SWSN(
            ann_matrix, normalized_nets=self.normalized_nets, 
            net_names=self.net_names, nodes=self.nodes, verbose=self.verbose)
        return self.W, self.swsn_time

    def combine_using_weights(self, weights):
        """ Combine the different networks using the specified weights
        *weights*: list of weights, one for each network
        """
        assert len(weights) == len(self.normalized_nets), \
            "%d weights supplied not enough for %d nets" % (len(weights), len(self.normalized_nets))
        combined_network = weights[0]*self.normalized_nets[0]
        for i, w in enumerate(weights):
            combined_network += w*self.normalized_nets[i] 
        return combined_network

    def weight_GMW(self, y, goid=None):
        if self.term_weights and goid in self.term_weights:
            weights = self.term_weights[goid]
            W = self.combine_using_weights(weights)
            process_time = 0
        else:
            W, process_time, weights = weight_GMW(y, self.normalized_nets, self.net_names, goid=goid) 
        return W, process_time, weights

    def save_net(self, out_file):
        print("Writing %s" % (out_file))
        utils.checkDir(os.path.dirname(out_file))
        if out_file.endswith('.npz'):
            # when the net was loaded, the idx file was already written
            # so no need to write it again
            sp.save_npz(out_file, self.W_SWSN)
        else:
            # convert the adjacency matrix to an edgelist
            G = nx.from_scipy_sparse_matrix(self.W_SWSN)
            idx2node = {i: n for i, n in enumerate(self.nodes)}
            # see also convert_node_labels_to_integers
            G = nx.relabel_nodes(G, idx2node, copy=False)
            delimiter = '\t'
            if out_file.endswith('.csv'):
                delimiter = ','
            nx.write_weighted_edgelist(G, out_file, delimiter=delimiter)


class Sparse_Annotations:
    """
    An object to hold the sparse annotations (including negative examples as -1),
        the list of GO term IDs giving the index of each term, and a mapping from term to index
    """
    # TODO add the DAG matrix
    def __init__(self, dag_matrix, ann_matrix, goids, prots):
        self.dag_matrix = dag_matrix
        self.ann_matrix = ann_matrix
        self.goids = goids
        # used to map from index to goid and vice versa
        self.goid2idx = {g: i for i, g in enumerate(goids)}
        self.prots = prots
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(prots)}

        #self.eval_ann_matrix = None 
        #if pos_neg_file_eval is not None:
        #    self.add_eval_ann_matrix(pos_neg_file_eval)

    #def add_eval_ann_matrix(self, pos_neg_file_eval):
    #    self.ann_matrix, self.goids, self.eval_ann_matrix = setup_eval_ann(
    #        pos_neg_file_eval, self.ann_matrix, self.goids, self.prots)
    #    # update the goid2idx mapping
    #    self.goid2idx = {g: i for i, g in enumerate(self.goids)}

    def reshape_to_prots(self, new_prots):
        """ *new_prots*: list of prots to which the cols should be changed (for example, to align to a network)
        """
        print("\treshaping %d prots to %d prots (%d in common)" % (
            len(self.prots), len(new_prots), len(set(self.prots) & set(new_prots))))
        # reshape the matrix cols to the new prots
        # put the prots on the rows to make the switch
        new_ann_mat = sp.lil_matrix((len(new_prots), self.ann_matrix.shape[0]))
        ann_matrix = self.ann_matrix.T.tocsr()
        for i, p in enumerate(new_prots):
            idx = self.node2idx.get(p)
            if idx is not None:
                new_ann_mat[i] = ann_matrix[idx]
        # now transpose back to term rows and prot cols
        self.ann_matrix = new_ann_mat.tocsc().T.tocsr()
        self.prots = new_prots

    def limit_to_terms(self, terms_list):
        """ *terms_list*: list of terms. Data from rows not in this list of terms will be removed
        """
        terms_idx = [self.goid2idx[t] for t in terms_list if t in self.goid2idx]
        print("\tlimiting data in annotation matrix from %d terms to %d" % (len(self.goids), len(terms_idx)))
        num_pos = len((self.ann_matrix > 0).astype(int).data)
        terms = np.zeros(len(self.goids))
        terms[terms_idx] = 1
        diag = sp.diags(terms)
        self.ann_matrix = diag.dot(self.ann_matrix)
        print("\t%d pos annotations reduced to %d" % (
            num_pos, len((self.ann_matrix > 0).astype(int).data)))

    def reshape_to_terms(self, terms_list, dag_mat):
        """ 
        *terms_list*: ordered list of terms to which the rows should be changed (e.g., COMP aligned with EXPC)
        *dag_mat*: new dag matrix. Required since the terms list could contain terms which are not in this DAG
        """
        assert len(terms_list) == dag_mat.shape[0], \
            "ERROR: # terms given to reshape != the shape of the given dag matrix"
        if len(terms_list) < len(self.goids):
            # remove the extra data first to speed up indexing
            self.limit_to_terms(terms_list)
        # now move each row to the correct position in the new matrix
        new_ann_mat = sp.lil_matrix((len(terms_list), len(self.prots)))
        #terms_idx = [self.goid2idx[t] for t in terms_list if t in self.goid2idx]
        for idx, term in enumerate(terms_list):
            idx2 = self.goid2idx.get(term)
            if idx2 is None:
                continue
            new_ann_mat[idx] = self.ann_matrix[idx2]
            #new_dag_mat[idx] = self.dag_matrix[idx2][:,terms_idx]
        self.ann_matrix = new_ann_mat.tocsr()
        self.dag_matrix = dag_mat.tocsr()
        self.goids = terms_list
        self.goid2idx = {g: i for i, g in enumerate(self.goids)}

    def limit_to_prots(self, prots):
        """ *prots*: array with 1s at selected prots, 0s at other indices
        """
        diag = sp.diags(prots)
        self.ann_matrix = self.ann_matrix.dot(diag)


def permute_sparse_matrix(M, new_row_order=None, new_col_order=None):
    """
    Reorders the rows and/or columns in a scipy sparse matrix 
        using the specified array(s) of indexes
        e.g., [1,0,2,3,...] would swap the first and second row/col.
    """
    if new_row_order is None and new_col_order is None:
        return M

    new_M = M
    if new_row_order is not None:
        permute_mat = sp.eye(M.shape[0]).tocoo()
        permute_mat.row = permute_mat.row[new_row_order]
        new_M = permute_mat.dot(new_M)
    if new_col_order is not None:
        permute_mat = sp.eye(M.shape[1]).tocoo()
        permute_mat.col = permute_mat.col[new_col_order]
        new_M = new_M.dot(permute_mat)
    return new_M


def propagate_ann_up_dag(pos_mat, dag_matrix):
    """ propagate all annotations up the DAG
    """
    # full_prop_ann_mat will get all the ancestors of every term
    full_prop_ann_mat = pos_mat.copy().T
    last_prop_ann_mat = pos_mat.T
    prop_ann_mat = pos_mat.copy().T
    # keep iterating until there are no more changes, or everything is 0
    # meaning there are no more edges upward
    while True:
        prop_ann_mat = prop_ann_mat.dot(dag_matrix)
        diff = (prop_ann_mat != last_prop_ann_mat)
        if diff is True or diff.nnz != 0:
            full_prop_ann_mat += prop_ann_mat
            last_prop_ann_mat = prop_ann_mat
        else:
            break
    # now change values > 1 in the full_prop_ann_mat to 1s 
    full_prop_ann_mat = (full_prop_ann_mat > 0).astype(int).T
    return full_prop_ann_mat
        

def create_sparse_net_file(
        out_pref, net_files=[], string_net_files=[], 
        string_nets=STRING_NETWORKS, string_cutoff=None, forcenet=False):
    if net_files is None:
        net_files = []
    # if there aren't any string net files, then set the string nets to empty
    if len(string_net_files) == 0:
        string_nets = [] 
    # if there are string_net_files, and string_nets is None, set it back to its default
    elif string_nets is None:
        string_nets = STRING_NETWORKS
    string_nets = list(string_nets)
    num_networks = len(net_files) + len(string_nets)
    # if there is only 1 string network, then write the name instead of the number
    if len(string_nets) == 1:
        num_networks = list(string_nets)[0] 
    sparse_nets_file = "%s%s-sparse-nets.mat" % (out_pref, num_networks)
    # the node IDs should be the same for each of the networks,
    # so no need to include the # in the ids file
    node_ids_file = "%snode-ids.txt" % (out_pref)
    net_names_file = "%s%s-net-names.txt" % (out_pref, num_networks)
    if forcenet is False \
       and os.path.isfile(sparse_nets_file) and os.path.isfile(node_ids_file) \
       and os.path.isfile(net_names_file):
        # read the files
        print("\treading sparse nets from %s" % (sparse_nets_file))
        sparse_networks = list(loadmat(sparse_nets_file)['Networks'][0])
        print("\treading node ids file from %s" % (node_ids_file))
        nodes = utils.readItemList(node_ids_file, 1)
        print("\treading network_names from %s" % (net_names_file))
        network_names = utils.readItemList(net_names_file, 1)

    else:
        print("\tcreating sparse nets and writing to %s" % (sparse_nets_file))
        sparse_networks, network_names, nodes = setup_sparse_networks(
            net_files=net_files, string_net_files=string_net_files, string_nets=string_nets, string_cutoff=string_cutoff)

        # now write them to a file
        write_sparse_net_file(
            sparse_networks, sparse_nets_file, network_names,
            net_names_file, nodes, node_ids_file)

    return sparse_networks, network_names, nodes


def write_sparse_net_file(
        sparse_networks, out_file, network_names,
        net_names_file, nodes, node_ids_file):
    #out_file = "%s/%s-net.mat" % (out_dir, version)
    # save this graph into its own matlab file because it doesn't change by GO category
    print("\twriting sparse networks to %s" % (out_file))
    mat_networks = np.zeros(len(sparse_networks), dtype=np.object)
    for i, net in enumerate(network_names):
        # convert to float otherwise matlab won't parse it correctly
        # see here: https://github.com/scipy/scipy/issues/5028
        mat_networks[i] = sparse_networks[i].astype(float)
    savemat(out_file, {"Networks":mat_networks}, do_compression=True)

    print("\twriting node2idx labels to %s" % (node_ids_file))
    with open(node_ids_file, 'w') as out:
        out.write(''.join(["%s\t%s\n" % (n, i) for i, n in enumerate(nodes)]))

    print("\twriting network names, which can be used to figure out " +
          "which network is at which index in the sparse-nets file, to %s" % (net_names_file))
    with open(net_names_file, 'w') as out:
        out.write(''.join(["%s\n" % (n) for n in network_names]))


def setup_sparse_networks(net_files=[], string_net_files=[], string_nets=[], string_cutoff=None):
    """
    Function to setup networks as sparse matrices 
    *net_files*: list of networks for which to make into a sparse
        matrix. The name of the file will be the name of the sparse matrix
    *string_net_files*: List of string files containing all 14 STRING network columns
    *string_nets*: List of STRING network column names for which to make a sparse matrix. 
    *string_cutoff*: Cutoff to use for the STRING combined network column (last)

    *returns*: List of sparse networks, list of network names, 
        list of proteins in the order they're in in the sparse networks
    """

    network_names = []
    # TODO build the sparse matrices without using networkx
    # I would need to make IDs for all proteins to ensure the IDs and
    # dimensions are the same for all of the matrices
    G = nx.Graph()
    for net_file in tqdm(net_files):
        name = os.path.basename(net_file)
        network_names.append(name)
        tqdm.write("Reading network from %s. Giving the name '%s'" % (net_file, name))
        open_func = gzip.open if '.gz' in net_file else open
        with open_func(net_file, 'r') as f:
            for line in f:
                line = line.decode() if '.gz' in net_file else line
                if line[0] == "#":
                    continue
                #u,v,w = line.rstrip().split('\t')[:3]
                line = line.rstrip().split('\t')
                u,v,w = line[:3]
                G.add_edge(u,v,**{name:float(w)})

    network_names += string_nets
    print("Reading %d STRING networks" % len(string_net_files))
    # for now, group all of the species string networks into a
    # massive network for each of the string_nets specified 
    for string_net_file in tqdm(string_net_files):
        tqdm.write("Reading network from %s" % (string_net_file))
        open_func = gzip.open if '.gz' in string_net_file else open
        with open_func(string_net_file, 'r') as f:
            for line in f:
                line = line.decode() if '.gz' in string_net_file else line
                if line[0] == "#":
                    continue
                #u,v,w = line.rstrip().split('\t')[:3]
                line = line.rstrip().split('\t')
                u,v = line[:2]
                attr_dict = {}
                combined_score = float(line[-1])
                # first check if the combined score is above the cutoff
                if string_cutoff is not None and combined_score < string_cutoff:
                    continue
                for net in string_nets:
                    w = float(line[full_column_names[net]-1])
                    if w > 0:
                        attr_dict[net] = w
                # if the edge already exists, 
                # the old attributes will still be retained
                G.add_edge(u,v,**attr_dict)
    print("\t%d nodes and %d edges" % (G.number_of_nodes(), G.number_of_edges()))

    print("\trelabeling node IDs with integers")
    G, node2idx, idx2node = convert_nodes_to_int(G)
    # keep track of the ordering for later
    nodes = [idx2node[n] for n in sorted(idx2node)]

    print("\tconverting graph to sparse matrices")
    sparse_networks = []
    net_names = []
    for i, net in enumerate(tqdm(network_names)):
        # all of the edges that don't have a weight for the specified network will be given a weight of 1
        # get a subnetwork with the edges that have a weight for this network
        print("\tgetting subnetwork for '%s'" % (net))
        netG = nx.Graph()
        netG.add_weighted_edges_from([(u,v,w) for u,v,w in G.edges(data=net) if w is not None])
        # skip this network if it has no edges, or leave it empty(?)
        if netG.number_of_edges() == 0:
            print("\t0 edges. skipping.")
            continue
        # now convert it to a sparse matrix. The nodelist will make sure they're all the same dimensions
        sparse_matrix = nx.to_scipy_sparse_matrix(netG, nodelist=sorted(idx2node))
        # convert to float otherwise matlab won't parse it correctly
        # see here: https://github.com/scipy/scipy/issues/5028
        sparse_matrix = sparse_matrix.astype(float) 
        sparse_networks.append(sparse_matrix)
        net_names.append(net) 

    return sparse_networks, net_names, nodes


def convert_nodes_to_int(G):
    index = 0 
    node2int = {}
    int2node = {}
    for n in sorted(G.nodes()):
        node2int[n] = index
        int2node[index] = n 
        index += 1
    # see also convert_node_labels_to_integers
    G = nx.relabel_nodes(G,node2int, copy=False)
    return G, node2int, int2node


def create_sparse_ann_and_align_to_net(
        obo_file, pos_neg_file, sparse_ann_file, net_prots,
        forced=False, verbose=False, **kwargs):
    """
    Wrapper around create_sparse_ann_file that also runs Youngs Negatives (potentially RAM heavy)
    and aligns the ann_matrix to a given network, both of which can be time consuming
    and stores those results to a file
    """ 
    if not kwargs.get('forcenet') and os.path.isfile(sparse_ann_file):
        print("Reading annotation matrix from %s" % (sparse_ann_file))
        loaded_data = np.load(sparse_ann_file, allow_pickle=True)
        dag_matrix = make_csr_from_components(loaded_data['arr_0'])
        ann_matrix = make_csr_from_components(loaded_data['arr_1'])
        goids, prots = loaded_data['arr_2'], loaded_data['arr_3']
        ann_obj = Sparse_Annotations(dag_matrix, ann_matrix, goids, prots)
    else:
        dag_matrix, ann_matrix, goids, ann_prots = create_sparse_ann_file(
                obo_file, pos_neg_file, **kwargs)
        #ann_matrix, goids = setup.setup_sparse_annotations(pos_neg_file, selected_terms, prots)
        ann_obj = Sparse_Annotations(dag_matrix, ann_matrix, goids, ann_prots)
        # so that limiting the terms won't make a difference, apply youngs_neg here
        if kwargs.get('youngs_neg'):
            ann_obj = youngs_neg(ann_obj, **kwargs)
        # align the ann_matrix prots with the prots in the network
        ann_obj.reshape_to_prots(net_prots)

        print("Writing sparse annotations to %s" % (sparse_ann_file))
        os.makedirs(os.path.dirname(sparse_ann_file), exist_ok=True)
        # store all the data in the same file
        dag_matrix_data = get_csr_components(ann_obj.dag_matrix)
        ann_matrix_data = get_csr_components(ann_obj.ann_matrix)
        np.savez_compressed(
            sparse_ann_file, dag_matrix_data, 
            ann_matrix_data, ann_obj.goids, ann_obj.prots)
    return ann_obj


def create_sparse_ann_file(
        obo_file, pos_neg_file, 
        forced=False, verbose=False, **kwargs):
    """
    Store/load the DAG, annotation matrix, goids and prots. 
    The DAG and annotation matrix will be aligned, and the prots will not be limitted to a network since the network can change.
    The DAG should be the same DAG that was used to generate the pos_neg_file
    *returns*:
        1) dag_matrix: A term by term matrix with the child -> parent relationships
        2) ann_matrix: A matrix with goterm rows, protein/node columns, and 1,0,-1 for pos,unk,neg values
        3) goterms: row labels
        4) prots: column labels
    """
    sparse_ann_file = pos_neg_file + '.npz'

    if forced or not os.path.isfile(sparse_ann_file):
        # load the pos_neg_file first. Should have only one hierarchy (e.g., BP)
        ann_matrix, goids, prots = setup_sparse_annotations(pos_neg_file)

        # now read the term hierarchy DAG
        # parse the go_dags first as it also sets up the goid_to_category dictionary
        dag_matrix, dag_goids = setup_obo_dag_matrix(obo_file, goids)
        dag_goids2idx = {g: i for i, g in enumerate(dag_goids)}
        # realign the terms in the dag_matrix to the terms in the annotation matrix
        # much faster than the other way around, since the DAG matrix is much smaller and sparser
        print("\taligning the DAG matrix (%d terms) and the ann_matrix (%d terms)" % (
            dag_matrix.shape[0], ann_matrix.shape[0]))
        # TODO for some reason this isn't working. 
        # The DAG is getting messed up (get_most_specific_ann is taking way too many iterations)
        #new_dag_mat = sp.lil_matrix(dag_matrix.shape)
        #for i, goid in enumerate(goids):
        #    new_dag_mat[i] = dag_matrix[dag_goids2idx[goid]]
        ## now add the leftover terms in the dag matrix
        #idx = len(goids)
        #for goid in set(dag_goids) - set(goids):
        #    new_dag_mat[idx] = dag_matrix[dag_goids2idx[goid]]
        #    goids.append(goid)
        #    idx += 1
        #dag_matrix = new_dag_mat.tocsr()
        ## add extra rows to the bottom of the matrix to match the size of the DAG
        #ann_matrix.resize((len(goids), len(prots)))
        #ann_matrix = ann_matrix.tocsr()
        #assert ann_matrix.shape[0] == dag_matrix.shape[0], \
        #        "Ann and DAG matrices do not have the same # terms"
        ann_matrix = alg_utils.align_mat(ann_matrix, (dag_matrix.shape[0], ann_matrix.shape[1]), goids, dag_goids2idx, verbose=verbose)
        goids = dag_goids

        print("\twriting sparse annotations to %s" % (sparse_ann_file))
        # store all the data in the same file
        dag_matrix_data = get_csr_components(dag_matrix)
        ann_matrix_data = get_csr_components(ann_matrix)
        #np.savez_compressed(
        #    sparse_ann_file, dag_matrix_data=dag_matrix_data, 
        #    ann_matrix_data=ann_matrix_data, goids=goids, prots=prots)
        np.savez_compressed(
            sparse_ann_file, dag_matrix_data, 
            ann_matrix_data, goids, prots)
    else:
        print("\nReading annotation matrix from %s" % (sparse_ann_file))
        loaded_data = np.load(sparse_ann_file, allow_pickle=True)
        dag_matrix = make_csr_from_components(loaded_data['arr_0'])
        ann_matrix = make_csr_from_components(loaded_data['arr_1'])
        goids, prots = loaded_data['arr_2'], loaded_data['arr_3']
        #dag_matrix = make_csr_from_components(loaded_data['dag_matrix_data'])
        #ann_matrix = make_csr_from_components(loaded_data['ann_matrix_data'])
        #goids, prots = loaded_data['goids'], loaded_data['prots']

    return dag_matrix, ann_matrix, goids, prots


def setup_obo_dag_matrix(obo_file, goterms):
    """
    *goterms*: if a set of goterms are given, then limit the dag to 
        the sub-ontology which has the given terms. Currently just returns the DAG for the first term. 
        TODO allow for multiple
    """
    go_dags = go_examples.parse_obo_file_and_build_dags(obo_file)
    dag_matrix = None
    for h, dag in go_dags.items():
        t = list(goterms)[0]
        if not dag.has_node(t):
            continue
        dag_matrix, goids = build_hierarchy_matrix(dag, goterms, h=h)
    if dag_matrix is None:
        print("ERROR: term %s not found in any of the sub-ontologies" % (t))
        sys.exit("Quitting")
    else:
        return dag_matrix, goids


def build_hierarchy_matrix(go_dag, goids, h=None):
    """
    *goids*: the leaf terms to use to get a sub-graph of the DAG.
        All ancestor terms will be included in the DAG
    """

    # UPDATE: limit to only the GO terms in R
    print("Limiting DAG to only the %d %s GO terms that have at least 1 annotation (assuming annotations already propagated up the DAG)" % (len(goids), h))
    ancestor_goids = set()
    for goid in goids:
        # if we already have the ancestors of this goid, then skip
        if goid in ancestor_goids:
            continue
        ancestor_goids.update(descendants(go_dag, goid))
    ancestor_goids.update(goids)
    goids_list = sorted(ancestor_goids)

    G = nx.subgraph(go_dag, ancestor_goids)
    if h is not None:
        print("\t%s DAG has %d nodes and %d edges" % (h, G.number_of_nodes(), G.number_of_edges()))
    else:
        print("\thierarchy DAG has %d nodes and %d edges" % (h, G.number_of_nodes(), G.number_of_edges()))

    # convert the GO DAG to a sparse matrix, while maintaining the order of goids so it matches with the annotation matrix
    dag_matrix = nx.to_scipy_sparse_matrix(G, nodelist=goids_list, weight=None)

    return dag_matrix, goids_list


def setup_sparse_annotations(pos_neg_file):
    """
    
    *returns*: 1) A matrix with goterm rows, protein/node columns, and 1,0,-1 for pos,unk,neg values
        2) List of goterms in the order in which they appear in the matrix
        3) List of prots in the order in which they appear in the matrix
    """
    print("\nSetting up annotation matrix")

    print("Reading positive and negative annotations for each protein from %s" % (pos_neg_file))
    if '-list' in pos_neg_file:
        ann_matrix, goids, prots = read_pos_neg_list_file(pos_neg_file) 
    else:
        ann_matrix, goids, prots = read_pos_neg_table_file(pos_neg_file) 
    num_pos = len((ann_matrix > 0).astype(int).data)
    num_neg = len(ann_matrix.data) - num_pos
    print("\t%d terms, %d prots, %d annotations. %d positives, %d negatives" % (
        ann_matrix.shape[0], ann_matrix.shape[1], len(ann_matrix.data), num_pos, num_neg))

    return ann_matrix, goids, prots


def read_pos_neg_table_file(pos_neg_file):
    """
    Reads a tab-delimited file with prots on rows, terms on columns, and 0,1,-1 as values
    *returns*: 1) A matrix with goterm rows, protein/node columns, and 1,0,-1 for pos,unk,neg values
        2) List of goterms in the order in which they appear in the matrix
        3) List of prots in the order in which they appear in the matrix
    """
    # rather than explicity building the matrix, use the indices to build a coordinate matrix
    # rows are prots, cols are goids
    i_list = []
    j_list = []
    data = []
    goids = []
    prots = []
    i = 0

    # read the file to build the matrix
    open_func = gzip.open if '.gz' in pos_neg_file else open
    with open_func(pos_neg_file, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.decode() if '.gz' in pos_neg_file else line
            if line[0] == '#':
                continue
            line = line.rstrip().split('\t')
            if line_idx == 0:
                # this is the header line
                goids = line[1:]
                continue

            prot, vals = line[0], line[1:]
            prots.append(prot)
            for j, val in enumerate(vals):
                # don't need to store 0 values
                if int(val) != 0:
                    i_list.append(i)
                    j_list.append(j)
                    data.append(int(val))
            i += 1

    # convert it to a sparse matrix 
    print("Building a sparse matrix of annotations")
    ann_matrix = sp.coo_matrix((data, (i_list, j_list)), shape=(len(prots), len(goids)), dtype=float).tocsr()
    ann_matrix = ann_matrix.transpose()
    return ann_matrix, goids, prots


# keeping this for backwards compatibility
def read_pos_neg_list_file(pos_neg_file):
    """
    Reads a tab-delimited file with two lines per goterm. A positives line, and a negatives line
        Each line has 3 columns: goid, pos/neg assignment (1 or -1), and a comma-separated list of prots
    *returns*: 1) A matrix with goterm rows, protein/node columns, and
        1,0,-1 for pos,unk,neg values
        2) List of goterms in the order in which they appear in the matrix
        3) List of prots in the order in which they appear in the matrix
    """
    # rather than explicity building the matrix, use the indices to build a coordinate matrix
    # rows are prots, cols are goids
    i_list = []
    j_list = []
    data = []
    goids = []
    prots = []
    # this will track the index of the prots
    node2idx = {}
    i = 0
    j = 0
    # read the file to build the matrix
    open_func = gzip.open if '.gz' in pos_neg_file else open
    with open_func(pos_neg_file, 'r') as f:
        for line in f:
            line = line.decode() if '.gz' in pos_neg_file else line
            if line[0] == '#':
                continue
            goid, pos_neg_assignment, curr_prots = line.rstrip().split('\t')[:3]
            curr_idx_list = []
            for prot in curr_prots.split(','):
                prot_idx = node2idx.get(prot)
                if prot_idx is None:
                    prot_idx = i 
                    node2idx[prot] = i
                    prots.append(prot)
                    i += 1
                curr_idx_list.append(prot_idx)
            # the file has two lines per goterm. A positives line, and a negatives line
            for idx in curr_idx_list:
                i_list.append(idx)
                j_list.append(j)
                data.append(int(pos_neg_assignment))
            if int(pos_neg_assignment) == -1:
                goids.append(goid)
                j += 1

    # convert it to a sparse matrix 
    print("Building a sparse matrix of annotations")
    ann_matrix = sp.coo_matrix((data, (i_list, j_list)), shape=(len(prots), len(goids)), dtype=float).tocsr()
    ann_matrix = ann_matrix.transpose()
    return ann_matrix, goids, prots


def youngs_neg(ann_obj, **kwargs):
    """
    for a term t, a gene g cannot be a negative for t if g shares an annotation with any gene annotated to t  
    *ann_obj*: contains the terms x genes matrix with 1 (positive), -1 (negative) and 0 (unknown) assignments, as well as the list of goids, and prots
    *cat*: the GO category to get (either P, F, or C)
    *returns*: The ann_obj modified inplace
    """
    print("Running the Youngs 2013 method for better negative examples")
    goids, prots = ann_obj.goids, ann_obj.prots  
    pos_mat = (ann_obj.ann_matrix > 0).astype(int)
    neg_mat = -(ann_obj.ann_matrix < 0).astype(int)
    num_pos = len(pos_mat.data) 
    num_neg = len(neg_mat.data) 
    # need to limit the annotations to only the most specific terms
    leaf_ann_mat = get_most_specific_ann(
            pos_mat, ann_obj.dag_matrix, verbose=kwargs.get('verbose'))
    if kwargs.get('verbose'):
        utils.print_memory_usage()
    # multiply genes*terms with terms*genes to get the genes x genes co-annotation matrix
    # TODO this is giving a memory error for IEA. I may be able to temporarily fix this by reducing the # terms
    #gene_co_ann_mat = leaf_ann_mat.T.dot(leaf_ann_mat)
    ## then multiply the annotations with the co-ann matrix to "give" each gene the annotations of its co-ann genes
    #co_ann_mat = leaf_ann_mat.dot(gene_co_ann_mat)
    # UPDATE: Instead of a gene x gene matrix, I can use a term x term matrix, which would be much smaller
    co_ann_mat = (leaf_ann_mat.dot(leaf_ann_mat.T)).dot(pos_mat)
    # the negatives which are coannoated with another gene will become 0 or greater
    neg_mat = neg_mat + co_ann_mat
    # get the matrix of only negative examples
    neg_mat = (neg_mat < 0).astype(int)
    new_num_neg = len(neg_mat.data)
    # and add them back together
    ann_obj.ann_matrix = pos_mat - neg_mat
    ann_obj.ann_matrix.eliminate_zeros() 
    if kwargs.get('verbose'):
        utils.print_memory_usage()
    # store them in the original ann_matrix instead of making a copy
    #new_ann_obj = Sparse_Annotations(
    #        ann_obj.dag_matrix, new_ann_mat, goids, prots)
    print("\t%d (%0.2f%%) negative examples relabeled to unknown examples (%d negative examples before, %d after)." % (
        num_neg - new_num_neg, (num_neg - new_num_neg) / float(num_neg)*100, num_neg, new_num_neg))
    print("\t%d total positive examples" % (num_pos))
    return ann_obj


def get_most_specific_ann(pos_mat, dag_matrix, verbose=False):
    # full_prop_ann_mat will have the annotations of the ancestor terms of all prot-term pairs
    # thus, we can remove the ancestor annotations to get only the most specific annotations 
    full_prop_ann_mat = sp.csr_matrix(pos_mat.shape).T
    #last_prop_ann_mat = sp.csr_matrix(pos_mat.shape)
    last_prop_ann_mat = pos_mat.T
    prop_ann_mat = pos_mat.copy().T
    # propagate all of the annotations up the DAG
    i = 0
    while True:
        i += 1
        if verbose:
            print("\tpropagating annotations (%s)..." % (i))
        prop_ann_mat = prop_ann_mat.dot(dag_matrix)
        diff = prop_ann_mat != last_prop_ann_mat
        if diff is True or diff.nnz != 0:
            # full_prop_ann_mat doesn't get the initial values of prop_ann_mat. Only the next level up
            full_prop_ann_mat += prop_ann_mat
            last_prop_ann_mat = prop_ann_mat
        else:
            break
    #print("\tdone!")
    # now change values > 1 in the full_prop_ann_mat to 1s 
    full_prop_ann_mat = (full_prop_ann_mat > 0).astype(int).T
    # and subtract them from the pos_mat to get the most specific ones
    spec_ann_mat = pos_mat - full_prop_ann_mat
    spec_ann_mat = (spec_ann_mat > 0).astype(int) 
    spec_ann_mat.eliminate_zeros() 
    if verbose:
        print("\tdone! %d most specific annotations" % (len(spec_ann_mat.data)))
    return spec_ann_mat


def weight_GMW(y, normalized_nets, net_names=None, goid=None):
    """ TODO DOC
    """
    start_time = time.process_time()
    if goid is not None:
        print("\tgoid %s: %d positives, %d negatives" % (goid, len(np.where(y == 1)[0]), len(np.where(y == -1)[0])))
    alphas, indices = findKernelWeights(y, normalized_nets)
    # print out the computed weights for each network
    if net_names is not None:
        print("\tnetwork weights: %s\n" % (', '.join(
            "%s: %s" % (net_names[x], alphas[i]) for
            i, x in enumerate(indices))))

    weights_list = [0]*len(normalized_nets)
    weights_list[indices[0]] = alphas[0]
    # now add the networks together with the alpha weight applied
    combined_network = alphas[0]*normalized_nets[indices[0]]
    for i in range(1,len(alphas)):
        combined_network += alphas[i]*normalized_nets[indices[i]] 
        weights_list[indices[i]] = alphas[i] 
    total_time = time.process_time() - start_time

    # don't write each goterm's combined network to a file
    return combined_network, total_time, weights_list


def weight_SWSN(ann_matrix, sparse_nets=None, normalized_nets=None, net_names=None,
                out_file=None, nodes=None, verbose=False):
    """ 
    *normalized_nets*: list of networks stored as scipy sparse matrices. Should already be normalized
    """
    # UPDATED: normalize the networks
    if sparse_nets is not None:
        print("Normalizing the networks")
        normalized_nets = []
        for net in sparse_nets:
            normalized_nets.append(_net_normalize(net))
    elif normalized_nets is None:
        print("No networks given. Nothing to do")
        return None, 0
    if len(normalized_nets) == 1:
        print("Only one network given to weight_SWSN. Nothing to do.")
        total_time = 0
        return sparse_nets[0], total_time
    if verbose:
        print("Removing rows with 0 annotations/positives")
        utils.print_memory_usage()
    # remove rows with 0 annotations/positives
    empty_rows = []
    for i in range(ann_matrix.shape[0]):
        pos, neg = alg_utils.get_goid_pos_neg(ann_matrix, i)
        # the combineWeightsSWSN method doesn't seem to
        # work if there's only 1 positive
        if len(pos) <= 1 or len(neg) <= 1:
            empty_rows.append(i)
    # don't modify the original annotation matrix to keep the rows matching the GO ids
    curr_ann_mat = delete_rows_csr(ann_matrix.tocsr(), empty_rows)

    if verbose:
        utils.print_memory_usage()
    print("Weighting networks for %d different GO terms" % (curr_ann_mat.shape[0]))
    print("Running simultaneous weights with specific negatives")
    start_time = time.process_time()
    alpha, indices = combineNetworksSWSN(curr_ann_mat, normalized_nets, verbose=verbose) 
    # print out the computed weights for each network
    if net_names is not None:
        print("network weights:")
        #print("\tnetworks chosen: %s" % (', '.join([net_names[i] for i in indices])))
        weights = defaultdict(int)
        for i in range(len(alpha)):
            weights[net_names[indices[i]]] = alpha[i]
        weights_table = ["%0.3e"%weights[net] for net in net_names]
        print('\t'.join(net_names))
        print('\t'.join(weights_table))

    # now add the networks together with the alpha weight applied
    weights_list = [0]*len(normalized_nets)
    weights_list[indices[0]] = alpha[0]
    combined_network = alpha[0]*normalized_nets[indices[0]]
    for i in range(1,len(alpha)):
        combined_network += alpha[i]*normalized_nets[indices[i]] 
        weights_list[indices[i]] = alpha[i] 
    total_time = time.process_time() - start_time

    if out_file is not None:
        # replace the .txt if present 
        out_file = out_file.replace('.txt', '.npz')
        utils.checkDir(os.path.dirname(out_file))
        print("\twriting combined network to %s" % (out_file))
        sp.save_npz(out_file, combined_network)
        # also write the node ids so it's easier to access
        # TODO figure out a better way to store this
        node2idx_file = out_file + "-node-ids.txt"
        print("\twriting node ids to %s" % (node2idx_file)) 
        with open(node2idx_file, 'w') as out:
            out.write(''.join("%s\t%s\n" % (n, i) for i, n in enumerate(nodes)))

        # write the alpha/weight of the networks as well
        net_weight_file = out_file + "-net-weights.txt"
        print("\twriting network weights to %s" % (net_weight_file)) 
        with open(net_weight_file, 'w') as out:
            out.write(''.join("%s\t%s\n" % (net_names[idx], str(alpha[i])) for i, idx in enumerate(indices)))

    return combined_network, total_time, weights_list


# copied from here: https://stackoverflow.com/a/26504995
def delete_rows_csr(mat, indices):
    """ 
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


# this was mostly copied from the deepNF preprocessing script
def _net_normalize(X):
    """ 
    Normalizing networks according to node degrees.
    """
    #if X.min() < 0:
    #    print("### Negative entries in the matrix are not allowed!")
    #    X[X < 0] = 0 
    #    print("### Matrix converted to nonnegative matrix.")
    # for now assume the network is symmetric
    #if (X.T != X).all():
    #    pass
    #else:
    #    print("### Matrix not symmetric.")
    #    X = X + X.T - np.diag(np.diag(X))
    #    print("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).A.flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0 
    # sparse matrix function to make a diagonal matrix
    D = sp.spdiags(deg, 0, X.shape[0], X.shape[1], format="csr")
    X = D.dot(X.dot(D))

    return X


# small utility functions for working with the pieces of
# sparse matrices when saving to or loading from a file
def get_csr_components(A):
    all_data = np.asarray([A.data, A.indices, A.indptr, A.shape], dtype=object)
    return all_data


def make_csr_from_components(all_data):
    return sp.csr_matrix((all_data[0], all_data[1], all_data[2]), shape=all_data[3])
