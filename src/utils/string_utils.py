"""
Function utilities for the networks in STRING
"""

from collections import OrderedDict

# the names of the columns in the full links file
full_column_names = OrderedDict()
full_column_names["protein1"]                = 1
full_column_names["protein2"]                = 2
full_column_names["neighborhood"]            = 3
full_column_names["neighborhood_transferred"]= 4
full_column_names["fusion"]                  = 5
full_column_names["cooccurence"]             = 6
full_column_names["homology"]                = 7
full_column_names["coexpression"]            = 8
full_column_names["coexpression_transferred"]= 9
full_column_names["experiments"]             = 10
full_column_names["experiments_transferred"] = 11
full_column_names["database"]                = 12
full_column_names["database_transferred"]    = 13
full_column_names["textmining"]              = 14
full_column_names["textmining_transferred"]  = 15
full_column_names["combined_score"]          = 16

# the three different sets of STRING networks considered
STRING_NETWORKS = list(full_column_names.keys())[2:-1]
NON_TRANSFERRED_STRING_NETWORKS = [net for net in STRING_NETWORKS if 'transferred' not in net]
CORE_STRING_NETWORKS = ["neighborhood", "fusion", "cooccurence", "coexpression", "experiments", "database"]

