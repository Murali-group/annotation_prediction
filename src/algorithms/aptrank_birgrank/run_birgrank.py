
import sys, os
from optparse import OptionParser, OptionGroup
import src.version_settings as v_settings
import src.utils.file_utils as utils
import src.setup_sparse_networks as ssn
import src.algorithms.alg_utils as alg_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from birgrank import birgRank
import src.go_term_prediction_examples.go_term_prediction_examples as go_examples
from scipy import sparse
from scipy.io import savemat, loadmat
from networkx.algorithms.dag import descendants
import networkx as nx


# can be run directly if desired
def parse_args(args):
    usage = '%s [options]\n' % (sys.argv[0])
    parser = OptionParser(usage=usage)

    group = OptionGroup(parser, 'Main Options')
    group.add_option('-N','--net-file',type='string',
                     help="Network file to use.")
    group.add_option('', '--pos-neg-file', type='string', action='append',
                     help="File containing positive and negative examples for each GO term. Must use either this or the --gaf-file option")
    group.add_option('-g', '--gaf-file', type='string',
                     help="File containing GO annotations in GAF format. Annotations will not be propagated")
    group.add_option('-b', '--obo-file', type='string', default=v_settings.GO_FILE,
                     help="GO OBO file which contains the GO DAG. Default: %s" % (v_settings.GO_FILE))
    group.add_option('-H', '--hierarchy', type='string', default='bp',
                     help="Hierarchy to use when creating the matrices and running BirgRank. Default: 'bp'")
    group.add_option('', '--ignore-ec', type='string',
                     help="Comma-separated list of evidence codes where annotations with the specified codes will be ignored when parsing the GAF file. " +
                     "For example, specifying 'IEA' will skip all annotations with an evidence code 'IEA'. ")
    group.add_option('-o', '--out-pref', type='string',
                     help="Output prefix for the annotations matrix and heirarchy matrix to create. " +
                     "Default: %s" % ('TODO'))
    parser.add_option_group(group)

    #group = OptionGroup(parser, 'BirgRank Options')
    #group.add_option('-a', '--alpha', type=float, action="append",
    #                 help="Alpha insulation parameter. Default=0.8")
    #group.add_option('-W', '--num-pred-to-write', type='int', default=100,
    #                 help="Number of predictions to write to the file. If 0, none will be written. If -1, all will be written. Default=100")
    #group.add_option('', '--only-cv', action="store_true", default=False,
    #                 help="Perform cross-validation only")
    #group.add_option('-C', '--cross-validation-folds', type='int',
    #                 help="Perform cross validation using the specified # of folds. Usually 5")

    (opts, args) = parser.parse_args(args)
    return opts


# take in a network, and annotation matrix
# also take in a GO heirarchy and make it into a matrix
def main(sparse_net_file, obo_file, pos_neg_file=None, gaf_file=None, ignore_ec=["IEA"],
         alpha=.5, theta=.5, mu=.5, h="bp", out_pref=None):

    W, prots = alg_utils.setup_sparse_network(sparse_net_file)
    # parse the go_dags first as it also sets up the goid_to_category dictionary
    go_dags = go_examples.parse_obo_file_and_build_dags(obo_file)

    dag_matrix, ann_matrix, goids = build_h_ann_matrices(prots, go_dags, pos_neg_file=pos_neg_file, gaf_file=gaf_file, h='bp')
    # make sure they're type float so matlab will parse them correctly
    sparse_net = W.astype('float') 
    ann_matrix = ann_matrix.astype('float') 
    dag_matrix = dag_matrix.astype('float')

    if out_pref is not None:
        out_file = "%s%s-annotations-and-go-dag.mat" % (out_pref, h)
        utils.checkDir(os.path.dirname(out_file))

        print("\twriting graph, annotation, and hierarchy matrices to %s" % (out_file))
        # write these to a file to run the matlab BirgRank 
        savemat(out_file, {"G": sparse_net, "R": ann_matrix, "H": dag_matrix}, do_compression=True)

        goids_file = "%s%s-goids.txt" % (out_pref, h)
        print("\twriting goids to %s" % (goids_file))
        with open(goids_file, 'w') as out:
            out.write(''.join("%s\n" % (goid) for goid in goids))

    run_birgrank = True 
    if run_birgrank is True:
        Xh = birgRank(sparse_net, ann_matrix.transpose(), dag_matrix, alpha=.5, theta=.5, mu=.5, eps=0.0001, max_iters=1000, verbose=True)
        Xh = Xh.T
        print(Xh.shape)

        out_file = "%s%s-pred-scores.txt" % (out_pref, h)
        print("\twriting scores to %s" % (out_file))
        # write the results for a single GO term
        with open(out_file, 'w') as out:
            for i in range(Xh.shape[0]):
                print("writing results for goterm %s" % (goids[i]))
                out.write(''.join("%s\t%s\t%s\n" % (goids[i], prots[j], score) for j, score in enumerate(Xh[i].toarray().flatten())))
                break
    return


def build_h_ann_matrices(
        prots, go_dags, pos_neg_file=None, gaf_file=None, h='bp',
        goterms=None):
    """
    """
    category_to_name = {"C": "cc", "P": "bp", "F": "mf"}
    name_to_category = {val: key for key, val in category_to_name.items()}
    cat = name_to_category[h]

    # mapping from a GO term ID and the category it belongs to ('C', 'F' or 'P')
    goid_to_category = go_examples.goid_to_category

    # aptrank doesn't use negatives, so just get the positives
    if pos_neg_file is not None:
        goid_prots, _ = alg_utils.parse_pos_neg_file(pos_neg_file, goterms=goterms) 
    elif gaf_file is not None:
        prot_goids_by_c, goid_prots, _, _ = go_examples.parse_gaf_file(
            gaf_file, ignore_ec=ignore_ec) 
        goid_prots = {goid: prots for goid, prots in goid_prots.items()
                      if goid in goid_to_category and goid_to_category[goid] == cat}
    else:
        print("ERROR: must specify a pos-neg-file or gaf-file")
        sys.exit(1)
    print("\t%d GO terms" % (len(goid_prots)))
    if len(goid_prots) == 0:
        print("\tskipping")
        return sparse.csr_matrix((0,0)), sparse.csr_matrix((0,0)), []

    dag_matrix, goids = build_hierarchy_matrix(go_dags[cat], goid_prots.keys(), h=h)
    ann_matrix = setup_sparse_annotations(goid_prots, prots, goids, h=h)

    return dag_matrix, ann_matrix, goids


def setup_sparse_annotations(
        goid_prots, prots, goids, h="bp", taxon=None):
    """
    *goid_prots*: dictionary containing the proteins annotated to each go term
    *prots*: list of proteins in the same order as the matrix
    *goids*: list of goids in the same order as the hierarchy matrix
    *returns*: A matrix with goterm rows, protein/node columns, and 1,0 for pos,unk values
    """
    node2idx = {prot: i for i, prot in enumerate(prots)}
    goid2idx = {goid: i for i, goid in enumerate(goids)}
    # rather than explicity building the matrix, use the indices to build a coordinate matrix
    # rows are prots, cols are goids
    i_list = []
    j_list = []
    data = []
    # limit the annotations to the proteins which are in the networks
    prots_set = set(prots)
    print("%d goids in Hierarchy, %d goids with annotations" %
          (len(goids), len(goid_prots)))
    for goid in goid_prots:
        for prot in goid_prots[goid] & prots_set:
            i_list.append(node2idx[prot])
            j_list.append(goid2idx[goid])
            data.append(1)


    # convert it to a sparse matrix 
    print("Building a sparse matrix of annotations")
    ann_matrix = sparse.coo_matrix((data, (i_list, j_list)), shape=(len(prots), len(goids)), dtype=float).tocsr()
    print("\t%d pos/neg annotations" % (len(ann_matrix.data)))
    ann_matrix = ann_matrix.transpose()

    return ann_matrix


def build_hierarchy_matrix(go_dag, goids, h="bp"):
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
    print("\t%s DAG has %d nodes and %d edges" % (h, G.number_of_nodes(), G.number_of_edges()))

    # convert the GO DAG to a sparse matrix, while maintaining the order of goids so it matches with the annotation matrix
    dag_matrix = nx.to_scipy_sparse_matrix(G, nodelist=goids_list, weight=None)

    return dag_matrix, goids_list


# this function is called from other scripts that already have the prots list
def setup_h_ann_matrices(prots, obo_file, pos_neg_file, goterms=None):
    # parse the go_dags first as it also sets up the goid_to_category dictionary
    go_dags = go_examples.parse_obo_file_and_build_dags(obo_file)

    goids = []
    # TODO build a matrix with the direct annotations (i.e., from the gaf file)
    # for now, just use all of the propagated annotations
    # and then evaluate using the scores
    #for pos_neg_file in pos_neg_files:
    if 'bp' in pos_neg_file:
        h = 'bp'
    elif 'mf' in pos_neg_file:
        h = 'mf'
    elif 'cc' in pos_neg_file:
        h = 'cc'
    dag_matrix, ann_matrix, goids = build_h_ann_matrices(
        prots, go_dags, pos_neg_file=pos_neg_file, h=h, goterms=goterms)

    return dag_matrix, ann_matrix, goids


if __name__ == "__main__":
    opts = parse_args(sys.argv)
    ignore_ec = [] if opts.ignore_ec is None else opts.ignore_ec.split(',') 
    main(opts.net_file, opts.obo_file, opts.pos_neg_file, opts.gaf_file,
         ignore_ec=ignore_ec, out_pref=opts.out_pref)
