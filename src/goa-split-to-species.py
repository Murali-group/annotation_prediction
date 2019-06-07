#!/usr/bin/python

print("Importing libraries")

from optparse import OptionParser
import os
import sys
import gzip
from tqdm import tqdm
import utils.file_utils as utils


def open_file_to_write(out_file, out_handle, for_gain=False, write_gzip=True):
    out_file = out_file + '.gz' if write_gzip else out_file
    if not os.path.isfile(out_file):
        tqdm.write("Writing new species annotations to '%s'" % (out_file))
    else:
        tqdm.write("Appending species annotations to '%s'" % (out_file))
    if out_handle is not None: 
        # close the last file we had open
        out_handle.close()
        write_header = False 
    else:
        write_header = True

    # the file is not in order by species, so we will need to append to the currently existing file
    out_handle = gzip.open(out_file, 'ab') if write_gzip else open(out_file, 'a')

    if write_header is True:
        if for_gain is True:
            # header line for gain
            # the columns we want are described here: http://bioinformatics.cs.vt.edu/~murali/software/biorithm/gain.html
            # *orf* A systematic name for the gene.
            # *goid* The ID of the GO function. You can leave in the "GO:0+" prefix for a function. GAIN will strip it out.
            # *hierarchy* The GO category the function belongs to. You can use either the abbreviations "c", "f", and "p", the capitalised forms "C", "F", and "P", or the complete names "cellular_component", "molecular_function", and "biological_process".
            # *evidencecode* The evidence code for an annotation. GAIN currently ignores this information. Future versions of GAIN will use this information.
            # *annotation type* The value is either 1 (indicating that the gene is annotated with the function), 0 (indicating that the gene is unknown for the function), or -1 (indicating that the gene is not annotated with the function).
            gain_header_line = '\t'.join(['orf', 'goid', 'hierarchy', 'evidencecode', 'annotation type']) + "\n"
            out_handle.write(gain_header_line)
        #else:
            # header line for GAF file
            #header_line = '#' + '\t'.join(['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID', 'DB:Reference', 'Evidence_Code', 'With_(or)_From', 'Aspect', 'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 'Taxon_and_Interacting_taxon', 'Date', 'Assigned_By', 'Annotation_Extension', 'Gene_Product_Form_ID']) + '\n'
            #out_handle.write(header_line)

    return out_handle


def main(args):
    ## Parse command line args.
    usage = '%s [options]\n' % (sys.argv[0])
    parser = OptionParser(usage=usage)
    parser.add_option('-i', '--goa-annotations', 
                      help="File containing goa annotations downloaded from UniProt in GAF format. Should be compressed with gzip (i.e., ends with .gz)")
    parser.add_option('-o', '--out-dir', type='string', metavar='STR',
                      help="Directory to write the annotations of each species to.")
    parser.add_option('-s', '--selected-strains', type='string',
                      help="Uniprot reference proteome strains for which to write the individual files")
    parser.add_option('-z', '--gzip', action="store_true", default=False,
                      help="Write gzip files for each individual species.")
    parser.add_option('-g', '--for-gain', action="store_true", default=False,
                      help="Also write the file formatted to use as input for GAIN")

    (opts, args) = parser.parse_args(args)

    if opts.goa_annotations is None or opts.out_dir is None:
        sys.exit("--goa-annotations (-i) and --out-dir (-o) required")

    #selected_strains_file = "inputs/selected-strains.txt"
    selected_strains = None 
    if opts.selected_strains is not None:
        selected_strains = utils.readItemSet(opts.selected_strains, 1)

    last_taxon = ""
    # initialize output file handle variables
    out = None
    out_gain = None
    # now split the file by species
    with gzip.open(opts.goa_annotations, 'rb') as f:
        # it takes a while to count the number of lines, so I stored the line count from a previous run here
        if 'all' in opts.goa_annotations:
            total_lines = 408333951
        elif 'gcrp' in opts.goa_annotations:
            total_lines = 137704163
        for orig_line in tqdm(f, total=total_lines):
            line = orig_line.decode('UTF-8')
            if line.rstrip() == '' or line[0] == '!':
                continue

            # columns of this file are explained in the README: ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/README and http://geneontology.org/page/go-annotation-file-gaf-format-21
            # 
            #	Column  Contents
            #	1       DB
            #	2       DB_Object_ID
            #	3       DB_Object_Symbol
            #	4       Qualifier
            #	5       GO_ID
            #	6       DB:Reference
            #	7       Evidence Code
            #	8       With (or) From
            #	9       Aspect
            #	10      DB_Object_Name
            #	11      DB_Object_Synonym
            #	12      DB_Object_Type
            #	13      Taxon and Interacting taxon
            #	14      Date
            #	15      Assigned_By
            #	16      Annotation_Extension
            #	17      Gene_Product_Form_ID
            cols = line.rstrip().split('\t')
            #aspect = line[8]
            #if aspect != 'P':
            #    continue
            curr_taxon = cols[12].split(':')[-1]

            if selected_strains is not None and curr_taxon not in selected_strains:
                #print "Skipping taxon %s" % (curr_taxon)
                #print line
                #sys.exit()
                continue

            if curr_taxon != last_taxon:
                out_dir = "%s/%s" % (opts.out_dir, curr_taxon)
                utils.checkDir(out_dir)
                # write annotations of the new species to the given file
                out_file = "%s/%s-goa.gaf" % (out_dir, curr_taxon)
                out = open_file_to_write(out_file, out, write_gzip=opts.gzip)
                if opts.for_gain is True:
                    out_file_gain = "%s/%s-goa-for-gain.txt" % (out_dir, curr_taxon)
                    out_gain = open_file_to_write(out_file_gain, out_gain, for_gain=opts.for_gain)

                last_taxon = curr_taxon
                #count = 0

            # write the entire line to the species-specific file
            if opts.gzip:
                out.write(orig_line)
            else:
                out.write(line)

            if opts.for_gain:
                # columns described in open_file_to_write()
                orf = cols[1]  # orf is the uniprot ID
                goid = cols[4]
                hierarchy = cols[8]
                evidencecode = cols[6]
                if cols[3] == "":
                    annotation_type = 1 
                elif cols[3] == "NOT":
                    annotation_type = -1
                else:
                    annotation_type = 0

                out_gain.write('\t'.join([orf, goid, hierarchy, evidencecode, str(annotation_type)]) + '\n')
                #interactors = ' '.join(line.split(' ')
                #out2.write("%s %s %d" % (line.split,,score)

    out.close()
    print("Finished")


if __name__ == '__main__':
    main(sys.argv)
