import argparse
import sys
import os
import logging
from . import utils, iqmma

logger = logging.getLogger()


def run():
    parser = argparse.ArgumentParser(
        description = 'Proteomics quantitation workflow',
        epilog = '''
    Example of usage
    -------------
    (prefered way)
    Basic command for quantitation mode:

    $ iqmma -bio2 path_to_Biosaur2
            -dino path_to_Dinosaur
            -openms path_to_openMS
            -dif path_to_Diffacto
            -s1 paths_to_mzml_files_from_sample_1_*.mzML
            -s2 paths_to_mzml_files_from_sample_2_*.mzML
            -outdir output_directory

    Basic command for matching peptide intensities: (all mzml files goes into first sample without any differences, no quantitation applied)

    $ iqmma -bio2 path_to_Biosaur2
            -dino path_to_Dinosaur
            -openms path_to_openMS
            -dif path_to_Diffacto
            -s1 paths_to_all_mzml_files_*.mzML
            -outdir output_directory

    Or both mods could be used with config file for an advanced usage and configuration:

    $ iqmma -cfg path_to_config_file -cfg_category name_of_category_in_cfg

    -------------
    ''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('-logs', nargs='?', help='level of logging, (DEBUG, INFO, WARNING, ERROR, CRITICAL)', type=str, default='INFO', const='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NOTSET', 'debug', 'info', 'warning', 'error', 'critical', 'nonset'])
    parser.add_argument('-log_path', nargs='?', help='path to logging file', type=str, default='./iqmma.log', const='./iqmma.log')
    parser.add_argument('-cfg', nargs='?', help='path to config .ini file', type=str, default='', const='')
    parser.add_argument('-cfg_category', nargs='?', help='name of category to prioritize in the .ini file, default: DEFAULT', type=str, default='DEFAULT', const='DEFAULT')
    parser.add_argument('-example_cfg', nargs='?', help='Path to create example config .ini file or not if not stated', type=str, default='', const='')
    parser.add_argument('-dif', nargs='?', help='path to Diffacto', type=str, default='', const='')
#    parser.add_argument('-scav2dif', help='path to scav2diffacto')
    parser.add_argument('-s1', nargs='*', help='input mzML files for sample 1 (also file names are the keys for searching other needed files)', type=str, default='', )
    parser.add_argument('-s2', nargs='*', help='input mzML files for sample 2 (also file names are the keys for searching other needed files)', type=str, default='', )
#    parser.add_argument('-sampleNames', nargs='+', help='short names for samples for inner structure of results')
    parser.add_argument('-psm_folder', nargs='?', help='path to the folder with PSMs files', type=str, default='', const='')
    parser.add_argument('-psm_format', nargs='?', help='format or suffix to search PSMs files (may be PSMs_full.tsv or identipy.pep.xml for example)', type=str, default='PSMs_full.tsv', const='PSMs_full.tsv')
    parser.add_argument('-allowed_pepts', nargs='?', help='path to file with sequences of peptides (with "peptide" header) to use in quantitation', type=str, default='', const='')
    parser.add_argument('-allowed_prots', nargs='?', help='path to file with names of proteins (according fasta, with "dbname" header) to use in quantitation', type=str, default='', const='')
 
    parser.add_argument('-dino', nargs='?', help='path to Dinosaur', type=str, default='', const='')
#    parser.add_argument('-bio', nargs='?', help='path to Biosaur', type=str, default='', const='')
    parser.add_argument('-bio2', nargs='?', help='path to Biosaur2', type=str, default='', const='')
    parser.add_argument('-openMS', nargs='?', help='path to OpenMS FeatureFinderCentroided', type=str, default='', const='')
 
    parser.add_argument('-outdir', nargs='?', help='name of directory to store results', type=str, default='', const='')
    parser.add_argument('-feature_folder', nargs='?', help='directory to store features', type=str, default='', const='')
    parser.add_argument('-matching_folder', nargs='?', help='directory to store matched psm-feature pairs', type=str, default='', const='')
    parser.add_argument('-diffacto_folder', nargs='?', help='directory to store diffacto results', type=str, default='', const='')
 
    parser.add_argument('-overwrite_features', nargs='?', help='whether to overwrite existed features files (flag == 1) or use them (flag == 0)', type=int, default=0, const=0, choices=[0, 1])
    parser.add_argument('-overwrite_matching', nargs='?', help='whether to overwrite existed matched files (flag == 1) or use them (flag == 0)', type=int, default=0, const=0, choices=[0, 1])
    parser.add_argument('-overwrite_first_diffacto', nargs='?', help='whether to overwrite existed diffacto files (flag == 1) or use them (flag == 0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-mixed', nargs='?', help='whether to reanalyze mixed intensities (1) or not (0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-modified_seq', nargs='?', help='whether to use modified peptide sequences (1) or not (0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-venn', nargs='?', help='whether to plot venn diagrams (1) or not (0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-pept_intens', nargs='?', help='max_intens - as intensity for peptide would be taken maximal intens between charge states; summ_intens - as intensity for peptide would be taken sum of intens between charge states; z-attached - each charge state would be treated as independent peptide', type=str, default='z-attached', const='z-attached', choices=['z-attached', 'summ_intens', 'max_intens'])
    parser.add_argument('-choice', nargs='?', help='method how to choose right intensities for peptide. 0 - default order and min Nan values, 1 - min Nan and min of sum CV, 2 - min Nan and min of max CV, 3 - min Nan and min of squared sum CV, 4 - min Nan and min of squared sum of corrected CV', type=int, default=4, const=4, choices=[0, 1, 2, 3, 4,])
    parser.add_argument('-norm', nargs='?', help='normalization method for intensities to remove biases between feature detections. Applied for each file independently. Can be 3 - max, 2 - min, 1 - median or 0 - no normalization. 2 and 3 options are strongly not recommended to use.', type=int, default=1, const=1, choices=[0, 1, 2, 3])
    parser.add_argument('-isotopes', help='monoisotope error', nargs='+', type=int, default=[0,1,-1,2,-2])
    parser.add_argument('-outpept', nargs='?', help='name of output diffacto peptides file (important: .txt)', type=str, default='peptides.txt', const='peptides.txt')
    parser.add_argument('-outsampl', nargs='?', help='name of output diffacto samples file (important: .txt)', type=str, default='sample.txt', const='sample.txt')
    parser.add_argument('-outdiff', nargs='?', help='name of diffacto output file (important: .txt)', type=str, default='diffacto_out.txt', const='diffacto_out.txt')
    parser.add_argument('-min_samples', nargs='?', help='minimum number of samples for peptide usage, 0 means that the minimum of the number of files in s1 and s2 would be used, -1 means that half of the given number of files would be used', type=int, default=0, const=0)

    parser.add_argument('-mbr', nargs='?', help='match between runs (1 - on, 0 - off)', type=int, default=0, const=0, choices=[0, 1])
    parser.add_argument('-pval_threshold', nargs='?', help='P-value threshold for reliable differetially expressed proteins', type=float, default=0.05, const=0.05)
    parser.add_argument('-fc_threshold', nargs='?', help='Fold change threshold for reliable differetially expressed proteins', type=float, default=1., const=1.)
    parser.add_argument('-dynamic_fc_threshold', nargs='?', help='whether to apply dynamically calculated threshold (1) or not and use static -fc_threshold (0) ', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-pval_adj', nargs='?', help='P value adjustment method for multiple comparisons: Bonf - Bonferroni correction, BH - Benjamini–Hochberg procedure.', type=str, default='Bonf', const='Bonf', choices=['Bonf', 'BH'])

    parser.add_argument('-diffacto_args', nargs='?', help='String of additional arguments to submit into Diffacto (in command line the string should be in double quotes: \'\" \"\', in cfg file in single quotes) except: -i, -out, -samples, -min_samples; default: "-normalize median -impute_threshold 0.99" ', type=str, default='-normalize median -impute_threshold 0.99', const='')
    parser.add_argument('-dino_args', nargs='?', help='String of additional arguments to submit into Dinosaur (in command line the string should be in double quotes: \'\" \"\', in cfg file in single quotes) except: --outDir --outName; default: ""', type=str, default='', const='')
#    parser.add_argument('-bio_args', nargs='?', help='String of additional arguments to submit into Biosaur (hole string in single quotes in command line) except: -o; default: ""', type=str, default='', const='')
    parser.add_argument('-bio2_args', nargs='?', help='String of additional arguments to submit into Biosaur2 (in command line the string should be in double quotes: \'\" \"\', in cfg file in single quotes) except: -o; default: "-hvf 1000 -minlh 3"', type=str, default='-hvf 1000 -minlh 3', const='')
    parser.add_argument('-openms_args', nargs='?', help='String of additional arguments to submit into OpenMSFeatureFinder (in command line the string should be in double quotes: \'\" \"\', in cfg file in single quotes) except: -in, -out; default: "-algorithm:isotopic_pattern:charge_low 2 -algorithm:isotopic_pattern:charge_high 7"', type=str, default='-algorithm:isotopic_pattern:charge_low 2 -algorithm:isotopic_pattern:charge_high 7', const='')
    parser.add_argument('-decoy_prefix', nargs='?', help='String added to the protein name to showcase that it is a decoy (default: DECOY)', type=str, default='DECOY', const='DECOY')
    parser.add_argument('-threads', nargs='?', help='Number of threads for multiprocessing for Biosaur2 and FeatureFinderCentroided feature detections, recommended value is a number of the processor cores. Not overwrites individual values stated via -programm_args options.', type=int, default=4, const=4)

#    parser.add_argument('-version', action='version', version='%s' % (pkg_resources.require("scavager")[0], ))
    console_config = vars(parser.parse_args())
    console_keys = [x[1:] for x in sys.argv if x.startswith('-')]
    default_config = vars(parser.parse_args([]))
    users_config = {}
    if console_config['cfg'] :
        if os.path.exists(console_config['cfg']) :
            if console_config['cfg_category'] :
                s, users_keys = utils.read_cfg(console_config['cfg'] , console_config['cfg_category'] )
            else :
                s, users_keys = utils.read_cfg( console_config['cfg'], default_config['cfg_category'] )
            users_config = vars(parser.parse_args(s))
        else :
            logging.critical('path to config file does not exist')
            return -1

    args = default_config
    if users_config :
        for k in users_keys :
            args.update({k: users_config[k]})
    for k in console_keys :
        args.update({k: console_config[k]})

    loglevel = args['logs'].upper()
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logger.setLevel(numeric_level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args['log_path'] :
        args['log_path'] = os.path.abspath(os.path.normpath(args['log_path']))
        log_directory = os.path.dirname(args['log_path'])
        os.makedirs(log_directory, exist_ok=True)
        fh = logging.FileHandler(args['log_path'], mode='w', encoding='utf-8')
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    logger.info('Started')

    if args['example_cfg'] :
        p = os.path.abspath(os.path.normpath(args['example_cfg']))
        if os.path.exists(os.path.dirname(p)) or os.path.exists(p) :
            if os.path.exists(p) :
                logger.info('Example cfg would be overwrited')
            utils.write_example_cfg(args['example_cfg'], default_config)
            logger.info('Example cfg created')
            return 0
        else :
            logger.warning('Invalid path for example cfg creation. Directory does not exist')
            return 1

    iqmma.process_files(args)

if __name__ == '__main__':
    run()
