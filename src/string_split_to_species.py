#!/usr/bin/python

print("Importing libraries")

from optparse import OptionParser
import os
import sys
import gzip
#import subprocess
#import networkx as nx
from tqdm import tqdm
import src.version_settings as v_settings
import src.utils.file_utils as utils
import src.utils.string_utils as s_utils


#OUT_FILE_TEMPLATE = v_settings.STRING_TAXON_UNIPROT

# Some of the selected species are not found in STRING.
# For those, I manually selected the closest species in STRING.
# In those cases, I need to get the interactions and mappings from the closest STRING speices, but write the interactions file to the selected species
closest_species = {
    '242231': ['528354','528357'],
    '333849': ['565664'],
    '83333': ['511145','316385','316407'],
    '441771': ['929506','445337','592027','536233','445335'],
    '509170': ['575584'],
    '632': ['214092','187410'],
    '301447': ['160490'],
    '300267': ['198214'],
}

string_to_uniprot_species = {
        '528354': '242231',
        '528357': '242231',
        '565664': '333849',
        '511145': '83333',
        '316385': '83333',
        '316407': '83333',
        '929506': '441771',
        '445337': '441771',
        '592027': '441771',
        '536233': '441771',
        '445335': '441771',
        '575584': '509170',
        '214092': '632',
        '187410': '632',
        '160490': '301447',
        '198214': '300267'
}


def parse_args(args):
    ## Parse command line args.
    usage = '%s [options]\n' % (args[0])
    parser = OptionParser(usage=usage)
    parser.add_option('-i', '--string-file', type='string', default=v_settings.STRING_FILE,
            help="File containing interactions of all species downloaded from STRING. Will be split into each individual species. Default: %s." % (v_settings.STRING_FILE))
    parser.add_option('-o', '--out-dir', type='string', default=v_settings.STRING_TAXON_DIR,
            help="Directory to write the interactions of each species to. Default: %s." % (v_settings.STRING_TAXON_DIR))
    parser.add_option('-S', '--score-cutoff', type='int', default=400,
            help="Cutoff for the STRING interaction score. Scores range from 150-1000. Default is 400 (medium). Useful to save on file space")
    parser.add_option('-s', '--selected-species', type='string', default=v_settings.SELECTED_STRAINS,
            help="Uniprot reference proteome species to perform analyses on. Default: %s"% (v_settings.SELECTED_STRAINS))
    #parser.add_option('-u', '--map-to-uniprot', type='string', default=v_settings.STRING_TO_UNIPROT,
    #        help="Also write the file formatted to use as input for GAIN. Default: %s" % (v_settings.STRING_TO_UNIPROT))

    (opts, args) = parser.parse_args(args)

    if opts.string_file is None:
        sys.exit("--string-file (-i) and --out-dir (-o) required")

    return opts, args


def main(args):
    opts, args = parse_args(args)

    selected_species = utils.readItemSet(opts.selected_species, 1)

    string_species = set()
    for s in selected_species:
        if s in closest_species:
            print("\tmapping species %s to closest STRING species %s" % (s, ', '.join(closest_species[s])))
            string_species.update(set(closest_species[s]))
        else:
            string_species.add(s)

    print("%d selected species map to %d closest STRING species" % (len(selected_species), len(string_species)))

    species_to_split = []
    for species in string_species:
        if not os.path.isfile(v_settings.STRING_TAXON_FILE % (species, species, opts.score_cutoff)):
            species_to_split.append(species)

    # This can take a few hours, so only split the original file if needed
    if len(species_to_split) > 0:
        print("Splitting %d species from STRING file from %s to %s: %s" % (len(species_to_split), opts.string_file, opts.out_dir, ', '.join(species_to_split)))
        split_string_to_species(species_to_split, opts.string_file, opts.out_dir, score_cutoff=opts.score_cutoff)
    else:
        print("%d species have already been split from the main STRING file %s to %s." % (len(species_to_split), opts.string_file, opts.out_dir))

    # TODO For each of the uniprot reference selected_strains that aren't in STRING,  
    # I manually copied/concatenated the networks of the string_species 
    # to a network for the selected_species
    # This should be done here
    map_string_to_uniprot(string_species, score_cutoff=opts.score_cutoff) 


def map_string_to_uniprot(string_species, score_cutoff=150, only_combined_score=False):
    """
    *only_combined_score*: only write the combined score to the file. 
    """
    # now map those files to uniprot
    string_to_uniprot = {}
    #string_per_taxon = {s: set() for s in string_species}
    #string_to_uniprot_bitscore = {}
    uniprot_to_string = {} 

    # also see how many of the UniProt to STRING mappings there are
    uniprot_to_string = utils.readDict(v_settings.UNIPROT_TO_STRING, 1, 6)
    # take off the ';' at the end of the ID(??)
    uniprot_to_string = {p:uniprot_to_string[p].strip(';') for p in uniprot_to_string}
    # get the reverse mapping as well
    string_to_uniprot = {uniprot_to_string[p]:p for p in uniprot_to_string}
    print("\t%d total UniProt IDs map to %d total STRING IDs" % (len(uniprot_to_string), len(set(uniprot_to_string.values()))))
    # How many of these STRING IDs are in the STRING networks?
    print("\t%d out of %d STRING IDs from UniProt intersect with the %d STRING network proteins" % (len(set(uniprot_to_string.values()).intersection(set(string_to_uniprot.keys()))), len(set(uniprot_to_string.values())), len(string_to_uniprot)))

    # now map the interactinos from STRING to UniProt
    # -----------------------
    # -----------------------

    print("Deleting previously written networks")
    for species in string_species:
        out_species = species
        if species in string_to_uniprot_species:
            out_species = string_to_uniprot_species[species]
        out_file = v_settings.STRING_TAXON_UNIPROT % (out_species, out_species, score_cutoff)
        if os.path.isfile(out_file):
            print("\tremoving %s" %(out_file))
            os.remove(out_file)
        if only_combined_score is False:
            out_file = v_settings.STRING_TAXON_UNIPROT_FULL % (out_species, out_species, score_cutoff)
            if os.path.isfile(out_file):
                print("\tremoving %s" %(out_file))
                os.remove(out_file)

    species_mapping = {}
    for species in tqdm(string_species):
        tqdm.write("Mapping interactions for species '%s'" % (species))
        #map_counts = {x:0 for x in xrange(5000)}
        # interaction map counts
        #intx_map_counts = {x:0 for x in xrange(5000)}
        new_interactions = {}
        #prots_seen = set()
        string_edges = set()
        #intx_count = 0 
        string_file = v_settings.STRING_TAXON_FILE % (species, species, score_cutoff)
        # TODO check for overlapping interactions in cases where there are multiple species
        out_species = species
        if species in string_to_uniprot_species:
            out_species = string_to_uniprot_species[species]
        if not only_combined_score:
            out_file = v_settings.STRING_TAXON_UNIPROT_FULL % (out_species, out_species, score_cutoff)
            print("\twriting full links to %s" % (out_file))
            write_header = False 
            if not os.path.isfile(out_file):
                write_header = True
            out = open(out_file, 'a')
            if write_header:
                out.write("#" + '\t'.join(s_utils.full_column_names) + '\n')

        with open(string_file, 'r') as f:
            for line in f:
                if line[0] == "#":
                    continue
                #intx_count += 1 
                line = line.rstrip().split(' ')
                combined_score = int(line[-1])
                # each interaction has the taxon ID at the start. Remove that so we can map to uniprot
                # replacing by the species doesn't work for the string_species
                #a = line[0].replace(species+'.','')
                a = line[0]
                #a = a[a.index('.')+1:]
                b = line[1]
                #b = b[b.index('.')+1:]
                string_edges.add((a,b))
                if a in string_to_uniprot and b in string_to_uniprot:
                    u_a = string_to_uniprot[a]
                    u_b = string_to_uniprot[b]
                    new_interactions[(u_a, u_b)] = combined_score 
                    if not only_combined_score:
                        out.write("%s\t%s\t%s\n" % (u_a, u_b, '\t'.join(line[2:])))

        if not only_combined_score:
            out.close()
        string_ids = set([n for edge in string_edges for n in edge])
        uniprot_ids = set([n for edge in new_interactions for n in edge])
        tqdm.write("\t%d total STRING IDs map to %d total UniProt IDs" % (len(string_ids), len(uniprot_ids)))
        tqdm.write("\t%d STRING interactions map to %d UniProt interactions" % (len(string_edges), len(new_interactions)))

        species_mapping[(species,out_species)] = [len(string_ids), len(uniprot_ids), len(string_edges), len(new_interactions)]

        out_file = v_settings.STRING_TAXON_UNIPROT % (out_species, out_species, score_cutoff)
        tqdm.write("Writing %s" % (out_file))
        with open(out_file, 'a') as out:
            for (a,b), combined_score in new_interactions.items():
                out.write("%s\t%s\t%d\n" % (a, b, combined_score))

    # now write a table of the mappings
    print("Overview mapping statistics for each species:")
    print('\t'.join(['STRING_strain', 'UniProt_strain', 'STRING IDs', 'UniProt IDs', 'STRING Intx', 'UniProt Intx']))
    for species in species_mapping:
        print('\t'.join(list(species) + [str(x) for x in species_mapping[species]]))
    return


def split_string_to_species(species_to_split, string_file, out_dir, score_cutoff=150):
    """
    *score_cutoff*: Only include lines with a combined score >= *score_cutoff*. 150 is the lowest for STRING
    """
    #if score_cutoff > 150:
    print("\tonly including interactions with a combied score >= %d" % (score_cutoff))
    # estimate the number of lines
    # takes too long (~20 minutes)
    #num_lines = rawgencount(string_file)
    num_lines = 2800000000
    #print("%s has %d lines" % (string_file, num_lines))

    # files are very large, so use the gzip library 
    with gzip.open(string_file, 'rb') as f:
        header = f.readline()
        last_species = ""
        # now split the file by species
        # TODO the file appears to be organized by species, so only have to open one output file for writing at a time
        # otherwise this would have to either sort the file, open multiple files for writing, or append to all files
        for line in tqdm(f, total=num_lines):
            # needed for Python3
            line = line.decode('UTF-8')
            score = int(line.rstrip().split(' ')[-1])
            # only write the interactions with a score >= the score cutoff
            if score_cutoff is not None and score < score_cutoff:
                continue

            curr_species = line[:line.index('.')]

            if curr_species not in species_to_split:
                continue

            if curr_species != last_species:
                out_dir = "%s/%s" % (out_dir, curr_species)
                # analagous to mkdir -p directory from the command line
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                out_file = v_settings.STRING_TAXON_FILE % (curr_species, curr_species, score_cutoff)
                tqdm.write("Writing new species interactions to '%s'" % (out_file))
                if last_species != "":
                    # close the last file we had open
                    out.close()
                # open the new file for writing
                out = open(out_file, 'w')
                #out2 = open(out_file2, 'w')
                last_species = curr_species

            out.write(line)
            #interactors = ' '.join(line.split(' ')
            #out2.write("%s %s %d" % (line.split,,score)

    out.close()
    print("Finished splitting species string interactions")
    return


if __name__ == '__main__':
    main(sys.argv)
