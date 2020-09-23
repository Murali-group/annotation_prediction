#!/usr/bin/python

# code to post toxcast results to graphspace

#print("Importing Libraries"))

import os, sys
import yaml
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
# TODO give the path to this repo
from graphspace_python.api.client import GraphSpace
from graphspace_python.graphs.classes.gsgraph import GSGraph
import pandas as pd
import numpy as np
from scipy import sparse as sp
# GSGraph already implements networkx
import networkx as nx
# local imports (4 folders up)
fss_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0,fss_dir)
from src.utils import file_utils as utils
#import fungcat_settings as f_settings
# for string networks
from src import setup_sparse_networks as setup
from src.plot.graphspace import post_to_graphspace as gs
from src.algorithms import alg_utils as alg_utils
from src.algorithms import runner as runner
from src.plot import plot_utils as plot_utils
from src.evaluate import eval_leave_one_species_out as eval_loso
import run_eval_algs
# TODO this shouldn't be required
from scripts import experiments as net_exp


evidence_code_name = {
    "EXP": "Inferred from Experiment",
    "IDA": "Inferred from Direct Assay",
    "IPI": "Inferred from Physical Interaction",
    "IMP": "Inferred from Mutant Phenotype",
    "IGI": "Inferred from Genetic Interaction",
    "IEP": "Inferred from Expression Pattern",
    "ISS": "Inferred from Sequence or structural Similarity",
    "ISO": "Inferred from Sequence Orthology",
    "ISA": "Inferred from Sequence Alignment",
    "ISM": "Inferred from Sequence Model",
    "IGC": "Inferred from Genomic Context",
    "IBA": "Inferred from Biological aspect of Ancestor",
    "IBD": "Inferred from Biological aspect of Descendant",
    "IKR": "Inferred from Key Residues",
    "IRD": "Inferred from Rapid Divergence",
    "RCA": "Inferred from Reviewed Computational Analysis",
    "TAS": "Traceable Author Statement",
    "NAS": "Non-traceable Author Statement",
    "IC" : "Inferred by Curator",
    "ND" : "No biological Data available",
    "IEA": "Inferred from Electronic Annotation",
}

evidence_code_type = {"IEA": "electronic"}
for code in ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"]:
    evidence_code_type[code] = "experimental"
# Author Statement evidence codes are grouped in here as well
for code in ["ISS", "ISO", "ISA", "ISM", "IGC", "IBA", "IBD", "IKR", "IRD", "RCA", "NAS"]:
    evidence_code_type[code] = "computational"

# color nodes according to their type
node_type_color = {
    "prediction": "#d88c00",  # orange
    "annotation": "#40aff9",  # blue
    # also keep track of the negatives
    "neg-annotation": "#8f68a5",  # dark purple
    "non-taxon-annotation": "#8ec67b",  # green
    "non-taxon-neg-annotation": "#54575b",  # default grey 
    "default": "#D8D8D8",  # default grey - background-color
}
annotation_type_styles = {
    # all experimental evidence codes. Maroon double border, square
    "experimental": {"border-style": "double", "border-width": "10", "border-color": "#cc5800", "shape": "rectangle"},  
    # computational analysis evidence codes (still with some curator input). Black double border, square
    "computational": {"border-style": "double", "border-width": "10", "border-color": "#3a3835", "shape": "rectangle"},  
    # Inferred by Electronic Annotation. For now, just use the defaults
    "electronic": {"shape": "rectangle"},  
    #"electronic": {},  
}
edge_type_color = {
    "default": "#6b6b6b",
    "string": "#24bf1c",
}

alg_names = {
    'fastsinksource': "FSS",
    "sinksource": "SinkSource",
    "localplus": "Local+",
    "sinksourceplus": "SinkSource+",
    "genemania": "GeneMANIA",
    "birgrank": "BirgRank",
}


def get_mat_neighbors(A, n):
    neighbors = A[n].nonzero()[1]
    return neighbors


def load_net_ann_datasets(
        out_dir, taxon, 
        dataset, input_settings, alg_settings,
        uniprot_taxon_file, **kwargs):
    sparse_net_file = "%s/%s-net.npz" % (out_dir, taxon)
    node2idx_file = sparse_net_file + "-node-ids.txt"
    swsn_weights_file = sparse_net_file + "-swsn-weights.txt" 
    sparse_ann_file = "%s/ann.npz" % (out_dir)
    if not kwargs.get('forcenet') and \
            (os.path.isfile(sparse_net_file) and os.path.isfile(node2idx_file)) and \
            os.path.isfile(sparse_ann_file):
        print("Reading network from %s" % (sparse_net_file))
        W = sp.load_npz(sparse_net_file)
        print("\t%d nodes and %d edges" % (W.shape[0], len(W.data)/2))
        print("Reading node names from %s" % (node2idx_file))
        prots = utils.readItemList(node2idx_file, 1)
        new_net_obj = setup.Sparse_Networks(W, prots)
        if os.path.isfile(swsn_weights_file):
            print("Reading swsn weights file %s" % (swsn_weights_file))
            weights = [float(w) for w in utils.readItemList(swsn_weights_file, 1)]
            # also load the original networks to get the edge weights for the STRING networks
            net_obj = run_eval_algs.setup_net(input_settings['input_dir'], dataset, **kwargs)
            net_obj.swsn_weights = weights
        else:
            net_obj = new_net_obj
        print("\nReading annotation matrix from %s" % (sparse_ann_file))
        loaded_data = np.load(sparse_ann_file, allow_pickle=True)
        dag_matrix = setup.make_csr_from_components(loaded_data['arr_0'])
        ann_matrix = setup.make_csr_from_components(loaded_data['arr_1'])
        goids, prots = loaded_data['arr_2'], loaded_data['arr_3']
        ann_obj = setup.Sparse_Annotations(dag_matrix, ann_matrix, goids, prots)
        species_to_uniprot_idx = eval_loso.get_uniprot_species(uniprot_taxon_file, ann_obj)
        # TODO eval ann obj
        eval_ann_obj = None
    else:
        # load the network
        # TODO if a subset of the network was run, need to get that subset
        net_obj, ann_obj, eval_ann_obj = run_eval_algs.setup_dataset(
                dataset, input_settings['input_dir'], alg_settings, **kwargs) 
        species_to_uniprot_idx = eval_loso.get_uniprot_species(uniprot_taxon_file, ann_obj)
        new_net_obj = net_obj
        # run SWSN if needd
        #if net_obj.multi_net:
            # TODO if LOSO was run, need to leave out the taxon for edge weights to be accurate
        if taxon is not None:
            if kwargs.get('limit_to_taxons_file'):
                # limit the network to the specified species
                # read in the specified taxons from the file
                _, net_taxons = eval_loso.get_selected_species(species_to_uniprot_idx, kwargs['limit_to_taxons_file'])
                net_taxon_prots = net_exp.get_taxon_prots(net_obj.nodes, net_taxons, species_to_uniprot_idx)
                net_obj, ann_obj = net_exp.limit_to_taxons(net_taxon_prots, net_obj=net_obj, ann_obj=ann_obj, **kwargs)
            # leave out the annotations for this taxon ID
            train_ann_mat, test_ann_mat, sp_goterms = eval_loso.leave_out_taxon(
                taxon, ann_obj, species_to_uniprot_idx,
                eval_ann_obj=eval_ann_obj, **kwargs)
            taxon_prots = net_exp.get_taxon_prots(net_obj.nodes, [taxon], species_to_uniprot_idx)
            new_net_obj = net_exp.limit_net_to_target_taxon(
                train_ann_mat, taxon_prots, net_obj, ann_obj, **kwargs)
            W = new_net_obj.W
        #    else:
        #        W, _ = net_obj.weight_SWSN(ann_obj.ann_matrix)
        #        #new_net_obj =  
        else:
            W = net_obj.W
        print("\twriting sparse matrix to %s" % (sparse_net_file))
        sp.save_npz(sparse_net_file, W)
        print("\twriting node2idx labels to %s" % (node2idx_file))
        with open(node2idx_file, 'w') as out:
            out.write(''.join(["%s\t%d\n" % (prot,i) for i, prot in enumerate(net_obj.nodes)]))
        if net_obj.multi_net:
            print("\twriting swsn weights file to %s" % (swsn_weights_file))
            with open(swsn_weights_file, 'w') as out:
                out.write('\n'.join([str(w) for w in new_net_obj.swsn_weights])+'\n')
                net_obj.swsn_weights = new_net_obj.swsn_weights
        # now store them to a file
        print("\twriting sparse annotations to %s" % (sparse_ann_file))
        # store all the data in the same file
        dag_matrix_data = setup.get_csr_components(ann_obj.dag_matrix)
        ann_matrix_data = setup.get_csr_components(ann_obj.ann_matrix)
        #np.savez_compressed(
        #    sparse_ann_file, dag_matrix_data=dag_matrix_data, 
        #    ann_matrix_data=ann_matrix_data, goids=goids, prots=prots)
        np.savez_compressed(
            sparse_ann_file, dag_matrix_data, 
            ann_matrix_data, ann_obj.goids, ann_obj.prots)
    return net_obj, new_net_obj, ann_obj, eval_ann_obj, species_to_uniprot_idx


def setup_post_to_graphspace(
        config_map, selected_goid, alg='fastsinksource', 
        name_postfix='', tags=None,
        taxon=None, goid_summary_file=None,
        num_neighbors=1, nodes_to_post=None, **kwargs):

    input_settings, alg_settings, \
            output_settings, out_pref, kwargs = \
            plot_utils.setup_variables(
                    config_map, **kwargs)

    input_dir = input_settings['input_dir']
    dataset = input_settings['datasets'][0]
    for arg in ['ssn_target_only', 'ssn_target_ann_only', 'ssn_only', 
            'string_target_only', 'string_nontarget_only',
            'limit_to_taxons_file', 'add_target_taxon',
            'oracle_weights', 'rem_neg_neighbors', 'youngs_neg', 'sp_leaf_terms_only']:
        kwargs[arg] = dataset.get(arg)
    uniprot_taxon_file = "%s/%s" % (input_dir, dataset['taxon_file'])

# don't need it since we are re-running the alg anyway
#    # predictions file:
#    results_dir = "%s/%s/%s" % (
#        output_settings['output_dir'], dataset['net_version'], dataset['exp_name'])
#    alg_params = alg_settings[alg] 
#    combos = [dict(zip(alg_params.keys(), val))
#        for val in itertools.product(
#            *(alg_params[param] for param in alg_params))]
#    # TODO allow for multiple
#    if len(combos) > 1:
#        print("%d combinations for %s. Using the first one" % (len(combos), alg))
#    param_combo = combos[0]
#    # first get the parameter string for this runner
#    params_str = runner.get_runner_params_str(alg, dataset, param_combo)
#    prec_rec_str = "prec-rec%s-%s" % (taxon, selected_goid)
#    exp_type = 'loso'
#    pred_file = "%s/%s/%s%s%s%s.txt" % (results_dir, alg, exp_type, params_str, kwargs.get('postfix',''), prec_rec_str)
#    if not os.path.isfile(pred_file):
#        print("\tPredictions file not found: %s. Quitting" % (pred_file))
#        sys.exit(1)
#    print("\treading %s" % (pred_file))
#    df = pd.read_csv(pred_file, sep='\t')
#    print(df.head())

    out_dir = "outputs/viz/graphspace/%s-%s/" % (
            dataset['net_version'].split('/')[-1], dataset['exp_name'].split('/')[-1])
    os.makedirs(out_dir, exist_ok=True)
    print("storing net and ann files to %s" % (out_dir))

    # TODO allow posting without STRING
    net_obj, new_net_obj, ann_obj, eval_ann_obj, species_to_uniprot_idx = \
            load_net_ann_datasets(
        out_dir, taxon, 
        dataset, input_settings, alg_settings,
        uniprot_taxon_file, **kwargs)
    W = new_net_obj.W
    prots = ann_obj.prots

    # also run the alg to get the full prediction scores
    # TODO get them from a file?
    alg_settings = {alg: alg_settings[alg]}
    alg_settings[alg]['should_run'] = [True] 
    kwargs['verbose'] = True
    alg_runners = run_eval_algs.setup_runners(
            alg_settings, new_net_obj, ann_obj, 
            output_settings['output_dir'], **kwargs)
    run_obj = alg_runners[0]
    run_obj.goids_to_run = [selected_goid]

    train_ann_mat, test_ann_mat, sp_goterms = eval_loso.leave_out_taxon(
        taxon, ann_obj, species_to_uniprot_idx,
        eval_ann_obj=eval_ann_obj, **kwargs)
    # now run the loso evaluation for this term, and get the scores back 
    eval_loso.run_and_eval_algs(
        run_obj, ann_obj,
        train_ann_mat, test_ann_mat,
        taxon=taxon, **kwargs)
    term_scores = np.ravel(run_obj.goid_scores[ann_obj.goid2idx[selected_goid]].toarray())
    print("top 10 scores for %s, %s:" % (taxon, selected_goid))
    taxon_prots_idx = list(species_to_uniprot_idx[taxon])
    taxon_prots = [prots[i] for i in taxon_prots_idx]
    taxon_term_scores = term_scores[taxon_prots_idx] 
    print('\n'.join(["%s\t%0.4e" % ( 
        ann_obj.prots[taxon_prots_idx[i]], taxon_term_scores[i]) \
            for i in np.argsort(taxon_term_scores)[::-1][:10]]))

    pos_neg_file = "%s/%s" % (input_dir, dataset['pos_neg_file'])
    #selected_goid = "15643"  # toxic substance binding
    #selected_goid = "9405"  # pathogenesis
    #selected_goid = "98754"  # detoxification
    selected_goname = None
    # build a dictionary of the evidencecode for each prot
    uniprot_to_evidencecode = defaultdict(set)
    annotated_prots = set()
    neg_prots = set()
    if goid_summary_file is None:
        goid_summary_file = pos_neg_file.replace("bp-",'').replace("mf-",'')
        if '-list' in pos_neg_file:
            goid_summary_file = goid_summary_file.replace("-list","-summary-stats")
        elif '.gz' in pos_neg_file:
            goid_summary_file = goid_summary_file.replace(".tsv.gz","-summary-stats.tsv")
        else:
            goid_summary_file = goid_summary_file.replace(".tsv","-summary-stats.tsv")
    df_summary = pd.read_csv(goid_summary_file, sep='\t')
    goid_names = dict(zip(df_summary['GO term'], df_summary['GO term name']))
    #goid_num_anno = dict(zip(df_summary['GO term'], df_summary['# positive examples']))
    print("GO name: %s" % (goid_names[selected_goid]))
    selected_goname = goid_names[selected_goid].replace(' ','-')[0:20]
    # load the GAIN propagation to get the evidence code 
    ev_codes_file = dataset.get('ev_codes_file')
    if ev_codes_file is not None:
        for orf, goid, goname, hierarchy, evidencecode, annotation_type in utils.readColumns(ev_codes_file, 1,2,3,4,5,6):
            if selected_goid[:3] == "GO:":
                goid = "GO:" + "0"*(7-len(goid)) + goid
            if goid != selected_goid:
                continue
            selected_goname = goname.replace(' ','-')[0:20]
            if annotation_type != '1':
                continue 

            uniprot_to_evidencecode[orf].add(evidencecode)
    # limit it to the current taxon
    if taxon is not None:
        print("Getting species of each prot from %s" % (uniprot_taxon_file))
        #print("Limiting the prots to those for taxon %s (%s)" % (taxon, selected_species[taxon]))
        print("Limiting the prots to those for taxon %s" % (taxon))
        # for each of the 19 species, leave out their annotations 
        # and see how well we can retrieve them 
        uniprot_to_species = utils.readDict(uniprot_taxon_file, 1,2)
        if taxon not in species_to_uniprot_idx:
            print("Error: taxon ID '%d' not found" % (taxon))
            sys.exit()
        # also limit the proteins to those in the network
        print("\t%d prots for taxon %s." % (len(taxon_prots_idx), taxon))
        goid_idx = ann_obj.goid2idx[selected_goid]
        pos, neg = alg_utils.get_goid_pos_neg(train_ann_mat, goid_idx)
        non_taxon_annotated_prots = set([prots[i] for i in pos])
        non_taxon_neg_prots = set([prots[i] for i in neg])
        print("\t%d non-taxon pos, %d non-taxon neg" % (len(non_taxon_annotated_prots), len(non_taxon_neg_prots)))
        pos, neg = alg_utils.get_goid_pos_neg(test_ann_mat, goid_idx)
        annotated_prots = set([prots[i] for i in pos])
        neg_prots = set([prots[i] for i in neg])
        print("\t%d taxon pos, %d taxon neg" % (len(annotated_prots), len(neg_prots)))

    print("\t%d annotated prots for %s (%s)" % (len(annotated_prots), selected_goname, selected_goid))

    #conf_cutoff = 0.2
    conf_cutoff = -1
    predicted_prots = set() 
    ranks = {}
    scores = {}
    first_zero_rank = None
    for i, idx in enumerate(np.argsort(taxon_term_scores)[::-1]):
        rank = i + 1
        prot = prots[taxon_prots_idx[idx]]
        predicted_prots.add(prot) 
        score = taxon_term_scores[idx]
        scores[prot] = score
        if taxon is not None:
            ranks[prot] = rank
            if score == 0 and first_zero_rank is None:
                first_zero_rank = rank 
        else:
            ranks[prot] = rank

            # move the score between 0 and 1 if it's genemania (normally between -1 and 1)
            # as the score is used to set the opacity
            # TODO fix genemania
            #if alg == "genemania":
            #    pred_cut_conf[gene] = local_conf 
            #    local_conf = ((float(local_conf) - -1) / float(1--1)) * (1-0) + 0
            #pred_local_conf[gene] = local_conf 

    print("\t%d prots with a score" % (len(taxon_term_scores)))
    print("Rank of first zero score: %d" % (first_zero_rank))
    print("Ranks of left-out positives:")
    for gene in sorted(annotated_prots, key=ranks.get):
        print("%s\t%d" % (gene, ranks[gene]))
    print("Including top 30 ranked-proteins of left-out species")
    top_30 = sorted(set(taxon_prots) & set(ranks.keys()), key=ranks.get)[:30]
    if ev_codes_file is not None:
        print("Evidence codes of top 30:")
        for i, gene in enumerate(top_30):
            if gene in uniprot_to_evidencecode:
                print("%s\t%s\t%s" % (i, gene, uniprot_to_evidencecode[gene]))
    top_30 = set(top_30)

    if taxon is not None:
        print("Getting the induced subgraph of the neighbors of the %d annotated nodes" % (len(annotated_prots)))
        prededges = set()
        if nodes_to_post is not None: 
            print("Getting neighbors of %s" % (', '.join(nodes_to_post)))
            nodes_to_add_neighbors = set(nodes_to_post) 
        else:
            nodes_to_add_neighbors = annotated_prots.copy() | top_30
        node2idx = ann_obj.node2idx
        for i in range(opts.num_neighbors):
            #print("Adding neighbors %d" % (i+1))
            curr_nodes_to_add_neighbors = nodes_to_add_neighbors.copy()
            nodes_to_add_neighbors = set() 
            print("adding %sneighbors of %d nodes" % ("positive ", len(curr_nodes_to_add_neighbors)))
            for u in curr_nodes_to_add_neighbors:
                #neighbors = set(nx.all_neighbors(G, u))
                neighbors = set([prots[v] for v in get_mat_neighbors(W, node2idx[u])])
                if opts.node_to_post is None:
                    # UPDATE 2018-10: try adding just the positive neighbors of the node
                    # TODO make this a command-line option
                    neighbors = neighbors & (non_taxon_annotated_prots | annotated_prots | top_30)
                #if len(neighbors) > 15 and nodes_to_post is None:
                #    print("\tskipping adding neighbors of %s. len(neighbors): %d" % (u, len(neighbors)))
                #    continue
                nodes_to_add_neighbors.update(neighbors)
                prededges.update(set([(u,v) for v in neighbors]))
    else:
        print("Getting the induced subgraph of the %d annotated and %d predicted proteins" % (len(annotated_prots), len(predicted_prots)))
        print("not yet implemented. quitting")
        sys.exit()
    #    prededges = set(G.subgraph(annotated_prots.union(predicted_prots)).edges())
    prededges = set([tuple(sorted((u,v))) for u,v in prededges])
    # TODO I should also show the disconnected nodes
    prednodes = set([n for edge in prededges for n in edge])

    print("\t%d nodes, %d edges" % (len(prednodes), len(prededges)))
    if len(prededges) > 1000 or len(prednodes) > 500:
        print("\nToo many nodes/edges. Not posting to GraphSpace. Quitting")
        sys.exit()

    #graph_attr_file = ""
    #graph_attr, attr_desc = readGraphAttr()
    # add the edge weight from the network to attr_desc which will be used for the popup
    # set the edges as the neighbors of the annotated genes
    #prededges = set()
    # get the induced subgraph of the annotated nodes and predicted nodes
    #for n in func_prots:
    #    if not G.has_node(n):
    #        continue
    #    for neighbor in G.neighbors(n):
    #        prededges.add((n, neighbor))

    graph_attr = {n: {} for n in prednodes}
    attr_desc = {n: {} for n in prednodes}

    print("Reading gene names and species for each protein from %s" % (uniprot_taxon_file))
    #prot_species = utils.readDict(uniprot_taxon_file, 1, 2)
    uniprot_to_gene = utils.readDict(uniprot_taxon_file, 1, 4)
    # there can be multiple gene names. Just show the first one for now
    uniprot_to_gene = {n:gene.split(' ')[0] for n,gene in uniprot_to_gene.items()}
    node_labels = {} 

    print("building graphspace object")
    # get the abbreviation of the species names
    species_names, net_taxons = eval_loso.get_selected_species(species_to_uniprot_idx, kwargs['limit_to_taxons_file'])
    sp_abbrv = {t: ''.join(
        subs[0] for subs in sp_name.split(' ')[:2]) for t, sp_name in species_names.items()}
    # for each node, add the prediction values
    for n in tqdm(prednodes):
        # set the name of the node to be the gene name and add the k to the label
        gene_name = uniprot_to_gene.get(n,n)
        curr_taxon = uniprot_to_species[n]
        species_short_name = sp_abbrv[curr_taxon]
        # add the species to the front of the gene name
        label = "%s-%s" % (species_short_name, gene_name)
        uniprot_to_gene[n] = label
        #node_labels[n] = "%s\n%d" % (label, min(ranks[n], 43)) if n in annotated_prots else label
        node_labels[n] = "%s\n%d" % (label, ranks[n] if ranks[n] < first_zero_rank else first_zero_rank) if n in taxon_prots else label

        # maybe put the labels below the nodes?
        # helps with visualizing the background opacity
        graph_attr[n]['text-valign'] = 'bottom'
        # add the strain name to the popup
        attr_desc[n]['Strain'] = species_names[curr_taxon]
        if n in predicted_prots:
            # don't need to normalize because the confidence values are already between 0 and 1
            if taxon and (n in non_taxon_annotated_prots or n in non_taxon_neg_prots):
                pass
            else:
                # UPDATE: use the node rank instead of the node score
                #graph_attr[n]['background-opacity'] = pred_local_conf[n]
                if n not in ranks:
                    graph_attr[n]['background-opacity'] = scores[n]
                else:
                    #graph_attr[n]['background-opacity'] = scores[n]
                    graph_attr[n]['background-opacity'] = max([0.9 - (ranks[n] / float(first_zero_rank)), float(scores[n])])
                    attr_desc[n]["%s rank"%(alg_names[alg])] = ranks[n]
            attr_desc[n]["%s prediction score"%(alg_names[alg])] = "%0.4f" % (scores[n])
        #elif n in annotated_prots or (taxon and (n in non_taxon_annotated_prots or n in non_taxon_neg_prots)) \
        #     or n in neg_prots:
            #if n in pred_local_conf:
            #    graph_attr[n]['background-opacity'] = pred_local_conf[n]
            #    attr_desc[n]["Local prediction confidence"] = pred_local_conf[n] 
        # also add the annotation to the popup
        if n in uniprot_to_evidencecode:
            codes = uniprot_to_evidencecode[n]
            # TODO add bullet points to the list
            #attr_desc[n]["Evidence code"] = ''.join(["%s (%s)\n" % (c, evidence_code_name[c]) for c in codes])
            # order it by exp, comp, then elec
            evidence_codes = ''.join(["<li>%s (%s)</li>" % (c, evidence_code_name[c]) for c in codes if evidence_code_type[c] == 'experimental'])
            evidence_codes += ''.join(["<li>%s (%s)</li>" % (c, evidence_code_name[c]) for c in codes if evidence_code_type[c] == 'computational'])
            evidence_codes += ''.join(["<li>%s (%s)</li>" % (c, evidence_code_name[c]) for c in codes if evidence_code_type[c] == 'electronic'])
            attr_desc[n]["Evidence code"] = "<ul>%s</ul>" % (evidence_codes)

    # set the width of the edges by the network weight
    edge_weights = defaultdict(float)
    for u,v in tqdm(prededges):
        e = (u,v)
        if e not in attr_desc:
            attr_desc[e] = {}
        if e not in graph_attr:
            graph_attr[e] = {}
        #attr_desc[e]["edge weight"] = G.adj[u][v]]['weight']
        if net_obj.multi_net:
            #attr_desc[e]["Final edge weight"] = "%0.1f" % (W[node2idx[u]][:,node2idx[v]].A.flatten()[0])
            edge_type_weights = []
            # add the weights for the individual string networks
            for i in range(len(net_obj.net_names)):
                net_name = net_obj.net_names[i]
                net_name = "SSN (E-value <= 0.1)" if 'eval-e0_1' in net_name else net_name
                net = net_obj.sparse_networks[i]
                w = net[node2idx[u]][:,node2idx[v]].A.flatten()[0]

                if w != 0:
                    #attr_desc[e][net_name] = "%0.1f" % (w)
                    edge_type_weights.append("<li>%s: %0.1f</li>" % (net_name, w))
                    edge_weights[e] += w *net_obj.swsn_weights[i]
            attr_desc[e]["Edge weights by type"] = "<ul>%s</ul>" % (''.join(sorted(edge_type_weights)))
        else:
            attr_desc[e]["Edge weight"] = "%0.1f" % (W[node2idx[u]][:,node2idx[v]].A.flatten()[0])
        # make the edges somewhat opaque for a better visual style
        graph_attr[e]['opacity'] = 0.7

    # set the width of the edges by the network weight
    #edge_weights = {(u,v): float(W[node2idx[u]][:,node2idx[v]].A.flatten()[0]) for u,v in prededges}
    for e,w in edge_weights.items():
        attr_desc[e]["Final edge weight"] = "%0.1f" % (w)
    # TODO set the min and max as parameters or something
    #max_weight = 180 
    if net_obj.multi_net:
        max_weight = net_obj.swsn_weights[0]*180
        print(max_weight)
    else:
        max_weight = 180 
    for e in edge_weights:
        if edge_weights[e] > max_weight:
            edge_weights[e] = max_weight 
    graph_attr = gs.set_edge_width(prededges, edge_weights, graph_attr,
                                   a=1, b=12, min_weight=1, max_weight=max_weight)

    H = nx.Graph()
    H.add_edges_from(prededges)

    # see which DB the edge came from to set the edge color
    print("Getting the edge type from networks")
    if net_obj.multi_net:
        print("\tFrom both STRING and SEQ_SIM")
        seq_sim_edges = set()
        for u,v in prededges:
            # get the SSN weight of this edge. Should be the first network
            net = net_obj.sparse_networks[0]
            w = net[node2idx[u]][:,node2idx[v]].A.flatten()[0]
            if w != 0:
                # these are all undirected, so just store the sorted version
                u,v = tuple(sorted((u,v)))
                # give these the default color
                graph_attr[(u,v)]['color'] = edge_type_color['default']
                seq_sim_edges.add((u,v))

#        string_edges = set()
#        temp_version = '2017_10-string'
#        net = f_settings.NETWORK_template % (temp_version, temp_version) 
#        for u,v in utils.readColumns(net, 1, 2):
#            #if (u,v) not in prededges:
#            if not H.has_edge(u,v):
#                continue
#            # give these the default color
#            u,v = tuple(sorted((u,v)))
#            graph_attr[(u,v)]['color'] = edge_type_color['string']
#            string_edges.add((u,v))
        string_edges = prededges.difference(seq_sim_edges)
        print("\t%d edges from seq-sim, %d edges from STRING" % (len(seq_sim_edges), len(string_edges)))
        # set the color to STRING if it didn't come from sequence similarity
        for e in string_edges:
            #if 'color' not in graph_attr[e]:
            graph_attr[e]['color'] = edge_type_color['string']

    #elif 'STRING' in f_settings.NETWORK_VERSION_INPUTS[version]:
    #    for e in graph_attr:
    #        graph_attr[e]['color'] = edge_type_color['string']
    else:
        for e in graph_attr:
            graph_attr[e]['color'] = edge_type_color['default']

    # apply the evidence code style to each protein
    for n in prednodes:
        if n in annotated_prots:
            graph_attr[n]['color'] = node_type_color['annotation']
        elif taxon and n in non_taxon_annotated_prots:
            graph_attr[n]['color'] = node_type_color['non-taxon-annotation']
        elif taxon and n in non_taxon_neg_prots:
            graph_attr[n]['color'] = node_type_color['non-taxon-neg-annotation']
        elif n in neg_prots:
            graph_attr[n]['color'] = node_type_color['neg-annotation']
        elif n in predicted_prots:
            graph_attr[n]['color'] = node_type_color['prediction']
        if n in uniprot_to_evidencecode:
            curr_style = ""
            for evidencecode in uniprot_to_evidencecode[n]:
                curr_type = evidence_code_type[evidencecode]
                if curr_type == "experimental":
                    curr_style = annotation_type_styles[curr_type]
                    break
                elif curr_style == "computational":
                    continue
                else:
                    curr_style = annotation_type_styles[curr_type]
            graph_attr[n].update(curr_style)
        # temporary fix to get the non-target positive examples
        if n in non_taxon_annotated_prots:
            graph_attr[n].update(annotation_type_styles['experimental'])

    # TODO build the popups here. That way the popup building logic can be separated from the
    # GSGraph building logic
    popups = {}
    prednodes = set([n for edge in prededges for n in edge])
    for n in prednodes:
        popups[n] = gs.buildNodePopup(n, attr_val=attr_desc)
    for u,v in prededges:
        popups[(u,v)] = gs.buildEdgePopup(u,v, node_labels=uniprot_to_gene, attr_val=attr_desc)

    # Now post to graphspace!
    print("Building GraphSpace graph")
    G = gs.constructGraph(prededges, node_labels=node_labels, graph_attr=graph_attr, popups=popups)

    # TODO add an option to build the 'graph information' tab legend/info
    # build the 'Graph Information' metadata
    #desc = gs.buildGraphDescription(opts.edges, opts.net)
    desc = ''
    metadata = {'description':desc,'tags':[], 'title':''}
    if tags is not None:
        metadata['tags'] = tags
    G.set_data(metadata)
    if 'graph_exp_name' in dataset:
        graph_exp_name = dataset['graph_exp_name']
    else:
        graph_exp_name = "%s-%s" % (dataset['exp_name'].split('/')[-1], dataset['net_version'].split('/')[-1])
    graph_name = "%s-%s-%s-%s%s" % (
        selected_goname, selected_goid, alg, graph_exp_name, name_postfix)
    G.set_name(graph_name)

    # rather than call it from here and repeat all the options, return G, and then call this after 
    #post_graph_to_graphspace(G, opts.username, opts.password, opts.graph_name, apply_layout=opts.apply_layout, layout_name=opts.layout_name,
    #                         group=opts.group, make_public=opts.make_public)
    return G, graph_name


# TODO modify the utils post_to_graphspace.py parseArgs so I can call it instead
def parseArgs():
    ## Parse command line args.
    parser = argparse.ArgumentParser()

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, required=True,
                     help="Configuration file")
    group.add_argument('--alg', type=str,
                      help="Algorithm for which to get predictions. Must be in the config file.") 
    group.add_argument('--goterm', '-G', type=str,
                      help='GO-term ID for which annotations and precitions will be posted')
    group.add_argument('--goid-summary-file', '-S', type=str, 
                      help="File containing GO term names and # of annotations for each GO term")
    group.add_argument('--taxon', '-T', type=str, 
                      help="Specify the species taxonomy ID to use. Otherwise, all species will be used")
    group.add_argument('--num-neighbors', type=int, default=1,
                      help="Number of neighbors to add around taxon positives. Default: 1")
    group.add_argument('--node-to-post', type=str, action="append",
                      help="UniProt ID of a taxon node for which to get neighbors. Can specify multiple")

    # posting options
    group = parser.add_argument_group('GraphSpace Options')
    group.add_argument('--username', '-U', type=str, 
                      help='GraphSpace account username to post graph to. Required')
    group.add_argument('--password', '-P', type=str,
                      help='Username\'s GraphSpace account password. Required')
    #group.add_argument('', '--graph-name', type=str, metavar='STR', default='test',
    #                  help='Graph name for posting to GraphSpace. Default = "test".')
    #group.add_argument('', '--outprefix', type=str, metavar='STR', default='test',
    #                  help='Prefix of name to place output files. Required.')
    group.add_argument('--name-postfix', type=str, default='',
                      help='Postfix of graph name to post to graphspace.')
    group.add_argument('--group', type=str,
                      help='Name of group to share the graph with.')
    group.add_argument('--make-public', action="store_true", default=False,
                      help='Option to make the uploaded graph public')
    # TODO implement and test this option
    #group.add_argument('--group-id', type=str, metavar='STR',
    #                  help='ID of the group. Could be useful to share a graph with a group that is not owned by the person posting')
    group.add_argument('--tag', type=str, action="append",
                      help='Tag to put on the graph. Can list multiple tags (for example --tag tag1 --tag tag2)')
    group.add_argument('--apply-layout', type=str,
                      help='Specify the name of a graph from which to apply a layout. Layout name specified by the --layout-name option. ' + 
                      'If left blank and the graph is being updated, it will attempt to apply the --layout-name layout.')
    group.add_argument('--layout-name', type=str, default='layout1',
                      help="Name of the layout of the graph specified by the --apply-layout option to apply. Default: 'layout1'")
    # TODO implement parent nodes
    #group.add_argument('--include-parent-nodes', action="store_true", default=False,
    #                  help="Include source, target, intermediate parent nodes, and whatever parent nodes are in the --graph-attr file")

    # extra options
    #parser.add_option('', '--graph-attr', type=str, metavar='STR',
    #        help='Tab-separated File used to specify graph attributes 1st column: style, 2nd: style attribute, 3rd: list of uniprot ids (nodes/edges) separated by |, 4th: Description.')
    #parser.add_option('', '--set-edge-width', action="store_true", default=False,
    #                  help='Set edge widths according to the weight in the network file')

    opts = parser.parse_args()

    if opts.goterm is None or opts.alg is None:
        print("--goterm, --alg, required")
        sys.exit(1)

    if opts.username is None or opts.password is None:
        #parser.print_help()
        sys.exit("\nERROR: --username and --password required")

    return opts


def load_config_file(config_file):
    with open(config_file, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    return config_map


if __name__ == '__main__':
    opts = parseArgs()
    kwargs = vars(opts)
    config_map = load_config_file(opts.config)
    # TODO allow for multiple versions. Will need to allow for multiple exp_names as well(?)
    G, graph_name = setup_post_to_graphspace(
            config_map, opts.goterm, **kwargs)
    if opts.taxon:
        graph_name += "-%s"%opts.taxon
    if opts.node_to_post is not None:
        graph_name += '-'.join(opts.node_to_post)
    graph_name += opts.name_postfix
    G.set_name(graph_name)
    # example command: python src/post_to_graphspace.py --edges toxic-sub-bind-edges.txt --net inputs/2017_10-seq-sim/2017_10-seq-sim-net.txt --username jeffl@vt.edu --password f1fan --graph-name toxic-substance-binding-cc-test5 --graph-attr graph_attr.txt --tag test1 --tag test2 --set-edge-width
    gs.post_graph_to_graphspace(
            G, opts.username, opts.password, graph_name, 
            apply_layout=opts.apply_layout, layout_name=opts.layout_name,
            group=opts.group, make_public=opts.make_public)
