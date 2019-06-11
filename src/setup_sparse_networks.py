
# Script to setup network and annotations files as sparse matrices
# also used for weighting the networks with the
# Simultaneous Weight with Specific Negatives (SWSN) method

import os, sys, time
from optparse import OptionParser, OptionGroup
#import src.version_settings as v_settings
import src.utils.file_utils as utils
from src.utils.string_utils import full_column_names, \
    STRING_NETWORKS, NON_TRANSFERRED_STRING_NETWORKS, \
    CORE_STRING_NETWORKS
import src.algorithms.alg_utils as alg_utils
from src.weight_networks.findKernelWeights import findKernelWeights
from src.weight_networks.combineNetworksSWSN import combineNetworksSWSN
import networkx as nx
import numpy as np
from scipy.io import savemat, loadmat
from scipy import sparse
from tqdm import tqdm
import gzip


class Sparse_Networks:
    """
    An object to hold the sparse network (or sparse networks if they are to be joined later), 
        the list of nodes giving the index of each node, and a mapping from node name to index
    """
    def __init__(self, sparse_networks, nodes, net_names=None,
                 weight_swsn=False, weight_gm2008=False, unweighted=False):
        if isinstance(sparse_networks, list):
            self.sparse_networks = sparse_networks
            self.multi_net = True
        else:
            self.W = sparse_networks
            self.multi_net = False
        self.nodes = nodes
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(nodes)}
        self.net_names = net_names
        self.weight_swsn = weight_swsn
        self.weight_gm2008 = weight_gm2008
        self.unweighted = unweighted

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
        return weight_SWSN(ann_matrix, self.normalized_nets,
                           net_names=self.net_names, nodes=self.nodes)

    def weight_GM2008(self, y, goid):
        return weight_GM2008(y, self.normalized_nets, self.net_names, goid)


class Sparse_Annotations:
    """
    An object to hold the sparse annotations (including negative examples as -1),
        the list of GO term IDs giving the index of each term, and a mapping from term to index
    """
    def __init__(self, ann_matrix, goids, prots):
        self.ann_matrix = ann_matrix
        self.goids = goids
        # used to map from index to goid and vice versa
        self.goid2idx = {g: i for i, g in enumerate(goids)}
        self.prots = prots
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(prots)}


#def parse_args(args):
#    ## Parse command line args.
#    usage = '%s [options]\n' % (sys.argv[0])
#    parser = OptionParser(usage=usage)
#
#    # general parameters
#    group = OptionGroup(parser, 'Main Options')
#    group.add_option('','--version',type='string', action='append',
#            help="Version of the PPI to run. Can specify multiple versions and they will run one after the other. " +
#            "Default = '2017_10-string'" +
#            "Options are: %s." % (', '.join(v_settings.ALLOWEDVERSIONS)))
#    group.add_option('', '--pos-neg-file', type='string', action='append',
#            help="File containing positive and negative examples for each GO term. Required")
#    group.add_option('-o', '--out-pref-net', type='string',
#            help="Output prefix for the network file to create. " +
#                     "Default: %s" % ('TODO'))
#    group.add_option('', '--out-pref-ann', type='string',
#            help="Output prefix for the annotations file to create. " +
#                     "Default: %s" % ('TODO'))
#    parser.add_option_group(group)
#
#    # options to limit the # of GO terms or species
#    group = OptionGroup(parser, 'Limitting options')
#    group.add_option('', '--only-functions', type='string',
#            help="Run GAIN using only the functions in a specified " +
#            "file. If not specified, all functions will be used.")
#    group.add_option('-G', '--goterm', type='string', action="append",
#            help="Specify the GO terms to use (should be in GO:00XX format)")
#    group.add_option('-T', '--taxon', type='string', action="append",
#            help="Specify the species taxonomy ID to use.")
#    parser.add_option_group(group)
#
#    # parameters for STRING networks
#    group = OptionGroup(parser, 'STRING options')
#    group.add_option('', '--only-combined', action="store_true", default=False,
#            help="Use only the STRING combined network: \n\tcombined_score")
#    group.add_option('', '--only-core', action="store_true", default=False,
#            help="Use only the 6 core networks: \n\t%s" % (', '.join(CORE_STRING_NETWORKS)))
#    group.add_option('', '--non-transferred', action="store_true", default=False,
#            help="Use all non-transferred networks: \n\t%s" % (', '.join(NON_TRANSFERRED_STRING_NETWORKS)))
#    group.add_option('', '--all-string', action="store_true", default=False,
#            help="Use all individual 13 STRING networks: \n\t%s" % (', '.join(STRING_NETWORKS)))
#    group.add_option('-S', '--string-networks', type='string', 
#            help="Comma-separated list of string networks to use. " +
#                 "If specified, other STRING options will be ignored.")
#    parser.add_option_group(group)
#
#    # additional parameters
#    group = OptionGroup(parser, 'Additional options')
#    group.add_option('', '--weight-SWSN', type='string', 
#            help="Weight and combine the networks using the" +
#                 "Simultaneous Weights with Specific Negatives method." +
#                 "Must specify a prefix for the network")
#    group.add_option('', '--forcenet', action="store_true", default=False,
#                     help="Force re-building network matrix from scratch")
#    #group.add_option('', '--verbose', action="store_true", default=False,
#    #                 help="Print additional info about running times and such")
#    parser.add_option_group(group)
#
#    (opts, args) = parser.parse_args(args)
#
#    if opts.pos_neg_file is None:
#        print("--pos-neg-file required")
#        sys.exit(1)
#
#    for f in opts.pos_neg_file:
#        if not os.path.isfile(f):
#            print("ERROR: --pos-neg-file '%s' not found" % (f))
#            sys.exit(1)
#
#    if opts.version is None:
#        opts.version = ["2018_06-seq-sim-e0_1-string"]
#
#    for version in opts.version:
#        if version not in v_settings.ALLOWEDVERSIONS:
#            print("ERROR: '%s' not an allowed version. Options are: %s." % (version, ', '.join(v_settings.ALLOWEDVERSIONS)))
#            sys.exit(1)
#
#    return opts
#
#
#def main(versions, pos_neg_files, goterms=None, taxons=None,
#         out_pref_net=None, out_pref_ann=None,
#         string_networks=["combined_score"],
#         string_cutoff=400,
#         weight_swsn=None, forcenet=False,
#):
#
#    for version in versions:
#        INPUTSPREFIX, RESULTSPREFIX, network_file, selected_strains = v_settings.set_version(version)
#
#        if out_pref_net is None:
#            out_pref_net = '%s/sparse_nets/' % (INPUTSPREFIX)
#        utils.checkDir(os.path.dirname(out_pref_net))
#        if out_pref_ann is None:
#            out_pref_ann = '%s/sparse_nets/' % (INPUTSPREFIX)
#        utils.checkDir(os.path.dirname(out_pref_ann))
#
#        if taxons is not None:
#            for taxon in taxons:
#                out_pref = "%s%s-" % (out_pref_net, taxon)
#                sparse_networks, network_names, nodes = create_sparse_net_file(
#                    version, out_pref, taxon=taxon,
#                    string_nets=string_networks, string_cutoff=string_cutoff,
#                    forcenet=forcenet)
#
#                out_pref = "%s%s-" % (out_pref_ann, taxon)
#                ann_matrix, goids = create_sparse_ann_file(
#                    out_pref, pos_neg_files, goterms, nodes,
#                    selected_strains=selected_strains, taxon=taxon)
#
#                if weight_swsn:
#                    out_file = "%s%s-%d-nets-combined-SWSN.npz" % (
#                        weight_swsn, taxon, len(sparse_networks))
#                    weight_SWSN(ann_matrix, sparse_networks,
#                                net_names=network_names, out_file=out_file,
#                                nodes=nodes)
#        else:
#            sparse_networks, network_names, nodes = create_sparse_net_file(
#                version, out_pref_net, selected_strains=selected_strains,
#                string_nets=string_networks, string_cutoff=string_cutoff,
#                forcenet=forcenet)
#
#            # now write the annotations
#            ann_matrix, goids = create_sparse_ann_file(
#                out_pref_ann, pos_neg_files, goterms, nodes,
#                selected_strains=selected_strains)
#
#            if weight_swsn:
#                out_file = "%s%d-nets-combined-SWSN.npz" % (
#                    weight_swsn, len(sparse_networks))
#                weight_SWSN(ann_matrix, sparse_networks,
#                            net_names=network_names, out_file=out_file,
#                            nodes=nodes)


## TODO need to set this up again for FunGCAT
#def create_sparse_net_file(
#        version, out_pref, selected_strains=None, taxon=None, string_nets=[],
#        string_file_cutoff=400, string_cutoff=None, forcenet=False):
#    """
#    *string_file_cutoff*: cutoff for the combined score in the name of the string file
#    *string_cutoff*: This option allows you to use a higher cutoff than what the file has.
#        If None, then all edges in the string file will be used.  
#    """
#
#    net_files = []
#    string_net_files = []
#    if taxon is not None:
#        network_file = v_settings.STRING_TAXON_UNIPROT_FULL % (taxon, taxon, string_file_cutoff)
#        #ann_file = v_settings.FUN_FILE % (self.taxon, self.taxon)
#        string_net_files.append(network_file)
#
#    else:
#        # use all of the networks available to this version by default
#        if 'SEQ_SIM' in v_settings.NETWORK_VERSION_INPUTS[version]:
#            # load the seq sim network for this version
#            net_files.append(v_settings.SEQ_SIM_NETWORKS[version])
#        if 'STRING' in v_settings.NETWORK_VERSION_INPUTS[version]:
#            # and all of the string networks available
#            for taxon in selected_strains:
#                net_file = v_settings.STRING_TAXON_UNIPROT_FULL % (taxon, taxon, string_file_cutoff)
#                string_net_files.append(net_file)


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
    num_networks = len(net_files) + len(string_nets)
    # if there is only 1 string network, then write the name instead of the number
    if num_networks == 1 and len(string_nets) == 1:
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
        mat_networks[i] = sparse_networks[i]
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
    for i, net in enumerate(tqdm(network_names)):
        # all of the edges that don't have a weight for the specified network will be given a weight of 1
        # get a subnetwork with the edges that have a weight for this network
        print("\tgetting subnetwork for '%s'" % (net))
        netG = nx.Graph()
        netG.add_weighted_edges_from([(u,v,w) for u,v,w in G.edges(data=net) if w is not None])
        # now convert it to a sparse matrix. The nodelist will make sure they're all the same dimensions
        sparse_matrix = nx.to_scipy_sparse_matrix(netG, nodelist=sorted(idx2node))
        # convert to float otherwise matlab won't parse it correctly
        # see here: https://github.com/scipy/scipy/issues/5028
        sparse_matrix = sparse_matrix.astype(float) 
        sparse_networks.append(sparse_matrix)

    return sparse_networks, network_names, nodes


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


def create_sparse_ann_file(out_pref, pos_neg_files, goids, prots,
                           selected_strains=None, taxon=None):
    # setup the annotations
    ann_matrix, goids = setup_sparse_annotations(
        pos_neg_files, goids, prots, selected_species=selected_strains, taxon=taxon)

    out_file = "%s%d-annotations.mat" % (out_pref, len(goids))
    # convert the GO DAG to a sparse matrix, while maintaining the order of goids so it matches with the annotation matrix
    #dag_matrix = nx.to_scipy_sparse_matrix(G, nodelist=goids, weight=None)
    #print("\twriting sparse annotations to %s" % (out_file))
    #savemat(out_file, {'annotations': ann_matrix})
    # also save it as a scipy file
    out_file = out_file.replace('.mat', '.npz')
    print("\twriting sparse annotations to %s" % (out_file))
    sparse.save_npz(out_file, ann_matrix)

    # write this order of the GO IDs to a file so they can easily be accessed (to get the rows of the sparse matrix)
    # without having to perform these operations again
    goids_file = "%s%d-goids.txt" % (out_pref, len(goids))
    print("Writing %s" % (goids_file))
    with open(goids_file, 'w') as out:
        out.write(''.join("%s\t%d\n" % (goid, i) for i, goid in enumerate(goids)))

    return ann_matrix, goids


def setup_sparse_annotations(pos_neg_file, goterms, prots,
                             selected_species=None, taxon_prots=None):
    """
    *prots*: list of proteins used to set the index. 
        If proteins in the pos_neg_file are not in the list of specified prots, they will be ignored.
    
    *returns*: 1) A matrix with goterm rows, protein/node columns, and
        1,0,-1 for pos,unk,neg values
        2) List of goterms in the order in which they appear in the matrix
    """
    # TODO store the annotations in sparse matrix form, then load the sparse matrix directly
    # In the case of specific goterms passed in, load the matrix, then keep only the specified goterms 
    #num_goterms = 6
    #h = "bp"
    #ann_file = "%s/%s-%d-annotations.npz" % (out_dir, h, num_goterms)
    #
    #if os.path.isfile(ann_file):
    #    print("\nReading annotations from %s" % (ann_file))
    #    ann_matrix = sparse.load_npz(ann_file)
    #    print("\t%d goterms, %d pos/neg annotations" %
    #          (ann_matrix.shape[0], len(ann_matrix.data)))
    #    goids_file = "%s/%s-%d-goids.txt" % (out_dir, h, num_goterms)
    #    goids = utils.readItemList(goids_file, 1)
    #else:
    print("\nSetting up annotation matrix")

    # UPDATE 2018-10-22: build the matrix while parsing the file
    #goid_pos, goid_neg = alg_utils.parse_pos_neg_files(pos_neg_files, goterms=goterms) 
    node2idx = {prot: i for i, prot in enumerate(prots)}

    print("Reading positive and negative annotations for each protein from %s" % (pos_neg_file))
#def build_sparse_matrix(data, rows, cols):
    # rather than explicity building the matrix, use the indices to build a coordinate matrix
    # rows are prots, cols are goids
    i_list = []
    j_list = []
    data = []
    num_pos = 0
    num_neg = 0
    # limit the annotations to the proteins which are in the networks
    prots_set = set(prots)
    if taxon_prots is not None:
        # limit the annotations to the taxon of interest
        prots_set = prots_set & taxon_prots
#    for j, goid in enumerate(goids):
#        for prot in goid_pos[goid] & prots_set:
#            i_list.append(node2idx[prot])
#            j_list.append(j)
#            data.append(1)
#            num_pos += 1
#        for prot in goid_neg[goid] & prots_set:
#            i_list.append(node2idx[prot])
#            j_list.append(j)
#            data.append(-1)
#            num_neg += 1
    goids = []
    j = 0
    # TODO get the GO terms from the summary matrix if they're not available here.
    # first estimate the number of lines
    total = 0
    with open(pos_neg_file, 'r') as f:
        for line in f:
            total += 1
    # read the file to build the matrix
    with open(pos_neg_file, 'r') as f:
        for line in tqdm(f, total=total, disable=True if total < 250 else False):
            if line[0] == '#':
                continue
            goid, pos_neg_assignment, curr_prots = line.rstrip().split('\t')[:3]
            if goterms and goid not in goterms:
                continue
            # the file has two lines per goterm. A positives line, and a negatives line
            curr_prots = set(curr_prots.split(','))
            if int(pos_neg_assignment) == 1:
                for prot in curr_prots & prots_set:
                    i_list.append(node2idx[prot])
                    j_list.append(j)
                    data.append(1)
                    num_pos += 1
            elif int(pos_neg_assignment) == -1:
                for prot in curr_prots & prots_set:
                    i_list.append(node2idx[prot])
                    j_list.append(j)
                    data.append(-1)
                    num_neg += 1
                goids.append(goid)
                j += 1

    print("\t%d annotations. %d positive, %d negatives" % (len(data), num_pos, num_neg))

    # convert it to a sparse matrix 
    print("Building a sparse matrix of annotations")
    ann_matrix = sparse.coo_matrix((data, (i_list, j_list)), shape=(len(prots), len(goids)), dtype=float).tocsr()
    print("\t%d pos/neg annotations" % (len(ann_matrix.data)))
    ann_matrix = ann_matrix.transpose()

    return ann_matrix, goids


def weight_GM2008(y, normalized_nets, net_names=None, goid=None):
    """ TODO DOC
    """
    start_time = time.process_time()
    if goid is not None:
        print("\tgoid %s: %d positives, %d negatives" % (goid, len(np.where(y > 0)[0]), len(np.where(y < 0)[0])))
    alphas, indices = findKernelWeights(y, normalized_nets)
    if net_names is not None:
        print("\tnetwork weights: %s\n" % (', '.join(
            "%s: %s" % (net_names[x], alphas[i]) for
             i, x in enumerate(indices))))

    # now add the networks together with the alpha weight applied
    combined_network = alphas[0]*normalized_nets[indices[0]]
    for i in range(1,len(alphas)):
        combined_network += alphas[i]*normalized_nets[indices[i]] 
    total_time = time.process_time() - start_time

    # don't write each goterm's combined network to a file
    return combined_network, total_time


def weight_SWSN(ann_matrix, sparse_nets, net_names=None, out_file=None, nodes=None):
    """ TODO DOC
    """
    if len(sparse_nets) == 1:
        print("Only one network given to weight_SWSN. Nothing to do.")
        total_time = 0
        return sparse_nets[0], total_time
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

    # normalize the networks
    print("Normalizing the networks")
    normalized_nets = []
    for net in sparse_nets:
        normalized_nets.append(_net_normalize(net))
    print("Weighting networks for %d different GO terms" % (curr_ann_mat.shape[0]))
    print("Running simultaneous weights with specific negatives")
    start_time = time.process_time()
    alpha, indices = combineNetworksSWSN(curr_ann_mat, normalized_nets) 
    if net_names is not None:
        print("\tnetworks chosen: %s" % (', '.join([net_names[i] for i in indices])))

    # now add the networks together with the alpha weight applied
    combined_network = alpha[0]*sparse_nets[indices[0]]
    for i in range(1,len(alpha)):
        combined_network += alpha[i]*sparse_nets[indices[i]] 
    total_time = time.process_time() - start_time

    if out_file is not None:
        # replace the .txt if present 
        out_file = out_file.replace('.txt', '.npz')
        utils.checkDir(os.path.dirname(out_file))
        print("\twriting combined network to %s" % (out_file))
        sparse.save_npz(out_file, combined_network)
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

    return combined_network, total_time


# copied from here: https://stackoverflow.com/a/26504995
def delete_rows_csr(mat, indices):
    """ 
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sparse.csr_matrix):
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
    if X.min() < 0:
        print("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0 
        print("### Matrix converted to nonnegative matrix.")
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
    D = sparse.spdiags(deg, 0, X.shape[0], X.shape[1], format="csr")
    X = D.dot(X.dot(D))

    return X


#def run():
#    #versions = ["2017_10-seq-sim", "2017_10-seq-sim-x5-string"]
#    opts = parse_args(sys.argv)
#    goterms = alg_utils.select_goterms(
#            only_functions_file=opts.only_functions, goterms=opts.goterm) 
#    if goterms is not None:
#        print("%d GO terms from only_functions_file and/or specified GO terms" % (len(goterms)))
#
#    #goid_pos, goid_neg = alg_utils.parse_pos_neg_files(opts.pos_neg_file) 
#
#    # setup the selection of string networks 
#    string_networks = []
#    if opts.string_networks:
#        string_networks = opts.string_networks.split(',')
#        for net in string_networks:
#            if net not in STRING_NETWORKS:
#                print("ERROR: STRING network '%s' not one of the" +
#                      "available choices which are: \n\t%s" % (net, ', '.join(STRING_NETWORKS)))
#                sys.exit(1)
#    elif opts.only_combined:
#        string_networks = ['combined_score']
#    elif opts.only_core:
#        string_networks = CORE_STRING_NETWORKS
#    elif opts.non_transferred:
#        string_networks = NON_TRANSFERRED_STRING_NETWORKS
#    elif opts.all_string:
#        string_networks = STRING_NETWORKS
#
#    main(opts.version, opts.pos_neg_file, goterms=goterms, taxons=opts.taxon,
#         out_pref_net=opts.out_pref_net, out_pref_ann=opts.out_pref_ann,
#         string_networks=string_networks, string_cutoff=v_settings.VERSION_STRING_CUTOFF[opts.version],
#         weight_swsn=opts.weight_SWSN, forcenet=opts.forcenet
#    )
#
#
#if __name__ == "__main__":
#    run()
