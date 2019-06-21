
import os
import sys
import src.plot.plot_utils as plot_utils


#def main():
#    plot_utils.parse_args(sys.argv)


if __name__ == "__main__":
    config_map, kwargs = plot_utils.parse_args(sys.argv)

    plot_utils.main(config_map, **kwargs)
    #main()
