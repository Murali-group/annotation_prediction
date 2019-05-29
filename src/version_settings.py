# file to contain settings used by multiple scripts

import utils.file_utils as utils
import socket

# Ouputs Structure:
# ouputs - version1 - all - method 1 - exp_name 1 - results
#       |          |      |          - exp_name 2 - results
#       |          |      - method 2 - exp_name 1 - results
#       |           - species1
#       |
#        - version2

SEQ_SIM_NETWORKS = {
    # these have a STRING cutoff of 0.4
    "2018_06-seq-sim-e0_1-string":       "inputs/2018_06-seq-sim-e0_1/2018_06-seq-sim-e0_1-net.txt",
    "2018_06-seq-sim-e0_1-string-150":   "inputs/2018_06-seq-sim-e0_1/2018_06-seq-sim-e0_1-net.txt",
    "2018_06-seq-sim-e0_1-string-700":   "inputs/2018_06-seq-sim-e0_1/2018_06-seq-sim-e0_1-net.txt",
    "2018_06-seq-sim-e0_1-string-900":   "inputs/2018_06-seq-sim-e0_1/2018_06-seq-sim-e0_1-net.txt",
    }
# I already built these using a little bash script in each version's directory
for cutoff in ["1e-25", "1e-15", "1e-6", "0_1", "5", "20", "50"]:
    SEQ_SIM_NETWORKS["2018_06-seq-sim-e%s"%cutoff] = "inputs/2018_06-seq-sim-e%s/2018_06-seq-sim-e%s-net.txt" % (cutoff, cutoff)

# 200 species networks
for cutoff in ["1e-25", "0_1"]:
    SEQ_SIM_NETWORKS["2018_09-s200-seq-sim-e%s"%cutoff] = "inputs/2018_09-s200-seq-sim-e%s/2018_09-s200-seq-sim-e%s-net.txt" % (cutoff, cutoff)


NETWORK_VERSION_INPUTS = {
    "2018_06-seq-sim-e1e-25": ['SEQ_SIM'],
    "2018_06-seq-sim-e1e-15": ['SEQ_SIM'],
    "2018_06-seq-sim-e1e-6": ['SEQ_SIM'],
    "2018_06-seq-sim-e0_1": ['SEQ_SIM'],
    "2018_06-seq-sim-e5": ['SEQ_SIM'],
    "2018_06-seq-sim-e20": ['SEQ_SIM'],
    "2018_06-seq-sim-e50": ['SEQ_SIM'],
    "2018_06-seq-sim-e0_1-string": ['SEQ_SIM', 'STRING'],
    "2018_06-seq-sim-e0_1-string-150": ['SEQ_SIM', 'STRING'],
    "2018_06-seq-sim-e0_1-string-700": ['SEQ_SIM', 'STRING'],
    "2018_06-seq-sim-e0_1-string-900": ['SEQ_SIM', 'STRING'],
    "2018_09-s200-seq-sim-e1e-25": ['SEQ_SIM'],
    "2018_09-s200-seq-sim-e0_1": ['SEQ_SIM'],
    }

ALLOWEDVERSIONS = sorted(NETWORK_VERSION_INPUTS.keys())

VERSION_SELECTED_STRAINS = {}
for version in ALLOWEDVERSIONS:
    VERSION_SELECTED_STRAINS[version] = 'inputs/selected-strains.txt' 
for version in ["2018_09-s200-seq-sim-e1e-25", "2018_09-s200-seq-sim-e0_1"]:
    VERSION_SELECTED_STRAINS[version] = 'inputs/selected-strains/2018-09-12-strains-200.txt'
# TODO only a couple scripts still use this. should be removed
SELECTED_STRAINS = "inputs/selected-strains.txt"

NAME_TO_SHORTNAME = {
    "Neisseria gonorrhoeae FA 1090"                             : "Ng",
    "Peptoclostridium difficile / Clostridioides difficile 630" : "Cd",
    "Helicobacter pylori 85962"                                 : "Hp",
    "Klebsiella pneumoniae"                                     : "Kp",
    "Enterococcus faecium DO"                                   : "Ef",
    "Escherichia coli K-12"                                     : "Ec",
    "Haemophilus influenzae RD KW20"                            : "Hi",
    "Bordetella pertussis Tohama I"                             : "Bp",
    "Burkholderia cepacia"                                      : "Bc",
    "Clostridium botulinum A str. Hall"                         : "Cb",
    "Acinetobacter baumannii"                                   : "Ab",
    "Staphylococcus aureus"                                     : "Sa",
    "Vibrio cholerae O1 biovar El Tor str. N16961"              : "Vc",
    "Yersinia pestis"                                           : "Yp",
    "Streptococcus pyogenes"                                    : "Sp",
    "Pseudomonas aeruginosa"                                    : "Pa",
    "Salmonella typhimurium / Salmonella enterica"              : "Se",
    "Shigella dysenteriae serotype 1 (strain Sd197)"            : "Sd",
    "Mycobacterium tuberculosis"                                : "Mt",
}
NAME_TO_SHORTNAME2 = {
    "Neisseria gonorrhoeae FA 1090"                             : "N. gonorrhoeae",
    "Peptoclostridium difficile / Clostridioides difficile 630" : "C. difficile",
    "Helicobacter pylori 85962"                                 : "H. pylori",
    "Klebsiella pneumoniae"                                     : "K. pneumoniae",
    "Enterococcus faecium DO"                                   : "E. faecium",
    "Escherichia coli K-12"                                     : "E. coli",
    "Haemophilus influenzae RD KW20"                            : "H. influenzae",
    "Bordetella pertussis Tohama I"                             : "B. pertussis",
    "Burkholderia cepacia"                                      : "B. cepacia",
    "Clostridium botulinum A str. Hall"                         : "C. botulinum",
    "Acinetobacter baumannii"                                   : "A. baumannii",
    "Staphylococcus aureus"                                     : "S. aureus",
    "Vibrio cholerae O1 biovar El Tor str. N16961"              : "V. cholerae",
    "Yersinia pestis"                                           : "Y. pestis",
    "Streptococcus pyogenes"                                    : "S. pyogenes",
    "Pseudomonas aeruginosa"                                    : "P. aeruginosa",
    "Salmonella typhimurium / Salmonella enterica"              : "S. typhimurium",
    "Shigella dysenteriae serotype 1 (strain Sd197)"            : "S. dysenteriae",
    "Mycobacterium tuberculosis"                                : "M. tuberculosis",
}
NAME_TO_TAX = {}
TAX_TO_NAME = {}

GOA_DIR = "inputs/goa"
GOA_TAXON_DIR = "%s/2017_09/taxon" % (GOA_DIR)

# input files
GO_FILE = "%s/2017_09/2017-09-26-go.obo" % (GOA_DIR)
GOA_FILE = "%s/2017_09/2017-09-26-goa_uniprot_all.gaf.gz" % (GOA_DIR)

# parsed input files
# for example: inputs/goa/taxon/22839/22839-goa.gaf
GOA_TAXON_FILE = "%s/%%s/%%s-goa.gaf" % (GOA_TAXON_DIR)
# file containing all annotations for the 19 species
GOA_ALL_FUN_FILE = "%s/all-taxon-goa.txt" % (GOA_TAXON_DIR)

# STRING directories and file templates
STRING_DIR = "inputs/string"
STRING_TAXON_DIR = "%s/taxon" % (STRING_DIR)
STRING_FILE = "%s/protein.links.full.v10.5.txt.gz" % (STRING_DIR)
# cutoff to be used for the STRING interactions
# Ranges from 150-1000
# 400 is considered a "Medium" cutoff
VERSION_STRING_CUTOFF = {}
for version in ALLOWEDVERSIONS:
    if 'STRING' in NETWORK_VERSION_INPUTS[version]:
        VERSION_STRING_CUTOFF[version] = 400
# To use a different cutoff, need to call string_split_to_species.py
for version in ["2018_06-seq-sim-e1e-25-string-150", "2018_06-seq-sim-e0_1-string-150"]:
    VERSION_STRING_CUTOFF[version] = 150
for version in ["2018_06-seq-sim-e1e-25-string-700", "2018_06-seq-sim-e0_1-string-700"]:
    VERSION_STRING_CUTOFF[version] = 700
# useful to use a higher cutoff from a different file
# I should really just keep a file with the lowest cutoff, and then use the cutoff to filter edges from that file
VERSION_STRING_FILE_CUTOFF = VERSION_STRING_CUTOFF.copy()
for version in ["2018_06-seq-sim-e1e-25-string-900", "2018_06-seq-sim-e0_1-string-900"]:
    VERSION_STRING_FILE_CUTOFF[version] = 400 
    VERSION_STRING_CUTOFF[version] = 900

# Template for a species/taxon STRING file
# Last number is the cutoff used on interactions in this file
# for example: inputs/string/taxon/9606/9606.links.full.v10.5-400.txt
STRING_TAXON_FILE = "%s/%%s/%%s.links.full.v10.5-%%d.txt" % (STRING_TAXON_DIR)
# STRING FILE mapped to uniprot IDs
STRING_TAXON_UNIPROT = "%s/%%s/%%s-uniprot-links-v10.5-%%d.txt" % (STRING_TAXON_DIR)
STRING_TAXON_UNIPROT_FULL = "%s/%%s/%%s-uniprot-full-links-v10.5-%%d.txt" % (STRING_TAXON_DIR)

# first column is uniprot ID, second is organism ID
UNIPROT_TO_SPECIES = "inputs/protein-similarity/uniprot-species/2017-10-17-uniprot-prots-19-species-plus-string.tab"
# first column is uniprot ID, last is STRING ID
# STRING is now included in the UNIPROT_TO_SPECIES file
# TODO update scripts to use the UNIPROT_TO_SPECIES variable
UNIPROT_TO_STRING = UNIPROT_TO_SPECIES
VERSION_UNIPROT_TO_SPECIES = {} 
for version in ALLOWEDVERSIONS:
    if '2018_06' in version:
        VERSION_UNIPROT_TO_SPECIES[version] = "inputs/protein-similarity/2018_06/2018-06-14-uniprot-19-strains-plus-string.tab"
    elif '2018_09' in version:
        VERSION_UNIPROT_TO_SPECIES[version] = "inputs/protein-similarity/2018_09/2018-09-12-uniprot-200-strains-prots-plus-string.tab"
VERSION_UNIPROT_TO_STRING = VERSION_UNIPROT_TO_SPECIES


VERSION = ''
INPUTSPREFIX = ''
RESULTSPREFIX = ''
# processed network file for each version
NETWORK_template = "inputs/%s/%s-net.txt"


def set_version(version):
    global VERSION, INPUTSPREFIX, RESULTSPREFIX, NETWORK
    global NAME_TO_TAX, TAX_TO_NAME
    global UNIPROT_TO_SPECIES, UNIPROT_TO_STRING

    VERSION = version
    print("Using version '%s'" % (VERSION))

    INPUTSPREFIX = "inputs/%s" % VERSION
    RESULTSPREFIX = "outputs/%s" % VERSION
    NETWORK = NETWORK_template % (VERSION, VERSION)

    selected_strains = utils.readItemSet(VERSION_SELECTED_STRAINS[VERSION], 1)
    TAX_TO_NAME = utils.readDict(VERSION_SELECTED_STRAINS[VERSION], 1,2)
    NAME_TO_TAX = utils.readDict(VERSION_SELECTED_STRAINS[VERSION], 2,1)
    if version in VERSION_UNIPROT_TO_SPECIES:
        UNIPROT_TO_SPECIES = VERSION_UNIPROT_TO_SPECIES[version]
        UNIPROT_TO_STRING = VERSION_UNIPROT_TO_STRING[version]
    else:
        # use the default 
        VERSION_UNIPROT_TO_SPECIES[version] = UNIPROT_TO_SPECIES
        VERSION_UNIPROT_TO_STRING[version] = UNIPROT_TO_STRING

    return INPUTSPREFIX, RESULTSPREFIX, NETWORK, selected_strains
