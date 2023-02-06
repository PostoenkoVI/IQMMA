import argparse
import subprocess
import sys
import configparser
import pandas as pd
import numpy as np
import copy
import os
import errno
import time
import ast
import re
from os import listdir
import logging

from .utils import read_cgf, write_example_cfg, call_Dinosaur, call_Biosaur2, call_OpenMS, gaus, noisygaus, opt_bin, generate_users_output, diffacto_call, mix_intensity, charge_states_intensity_processing, read_PSMs, calibrate_mass, total, found_mean_sigma, optimized_search_with_isotope_error_, mbr


class WrongInputError(NotImplementedError):
    pass


class EmptyFileError(ValueError):
    pass


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

    Basic command for matching peptide intensities: (all mzml files goes into first sample without any differences, no quantitation applied)

    $ iqmma -bio2 path_to_Biosaur2 
            -dino path_to_Dinosaur 
            -openms path_to_openMS 
            -dif path_to_Diffacto -s1 paths_to_all_mzml_files_*.mzML -outdir out_path
    
    Or both mods could be used with config file for an advanced usage and configuration:

    $ iqmma -cfg path_to_config_file -cfg_category name_of_category_in_cfg
                                          
    -------------
    ''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    
    parser.add_argument('-logs', nargs='?', help='level of logging, (DEBUG, INFO, WARNING, ERROR, CRITICAL)', type=str, default='INFO', const='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NOTSET'])
    parser.add_argument('-log_path', nargs='?', help='path to logging file', type=str, default='./iqmma.log', const='./iqmma.log')
    parser.add_argument('-cfg', nargs='?', help='path to config .ini file', type=str, default='', const='')
    parser.add_argument('-cfg_category', nargs='?', help='name of category to prioritize in the .ini file, default: DEFAULT', type=str, default='DEFAULT', const='DEFAULT')
    parser.add_argument('-example_cfg', nargs='?', help='Path to create example config .ini file or not if not stated', type=str, default='', const='')
    parser.add_argument('-dif', nargs='?', help='path to Diffacto', type=str, default='', const='')
#    parser.add_argument('-scav2dif', help='path to scav2diffacto')
    parser.add_argument('-s1', nargs='+', help='input mzML files for sample 1 (also file names are the keys for searching other needed files)', type=str, default='', )
    parser.add_argument('-s2', nargs='*', help='input mzML files for sample 2 (also file names are the keys for searching other needed files)', type=str, default='', )
#    parser.add_argument('-sampleNames', nargs='+', help='short names for samples for inner structure of results')
    parser.add_argument('-PSM_folder', nargs='?', help='path to the folder with PSMs files', type=str, default='', const='')
    parser.add_argument('-PSM_format', nargs='?', help='format or suffix to search PSMs files (may be PSMs_full.tsv or identipy.pep.xml for example)', type=str, default='PSMs_full.tsv', const='PSMs_full.tsv')
    parser.add_argument('-pept_folder', nargs='?', help='path to folder with files with peptides filtered on certain FDR (default: searching for them near PSMs)', type=str, default='', const='')
    parser.add_argument('-prot_folder', nargs='?', help='path to folder with files with proteins filtered on certain FDR (default: searching for them near PSMs)', type=str, default='', const='')
    
    parser.add_argument('-dino', nargs='?', help='path to Dinosaur', type=str, default='', const='')
#    parser.add_argument('-bio', nargs='?', help='path to Biosaur', type=str, default='', const='')
    parser.add_argument('-bio2', nargs='?', help='path to Biosaur2', type=str, default='', const='')
    parser.add_argument('-openMS', nargs='?', help='path to OpenMS feature', type=str, default='', const='')
    
    parser.add_argument('-outdir', nargs='?', help='name of directory to store results', type=str, default='', const='')
    parser.add_argument('-feature_folder', nargs='?', help='directory to store features', type=str, default='', const='')
    parser.add_argument('-matching_folder', nargs='?', help='directory to store matched psm-feature pairs', type=str, default='', const='')
    parser.add_argument('-diffacto_folder', nargs='?', help='directory to store diffacto results', type=str, default='', const='')
    
    parser.add_argument('-overwrite_features', nargs='?', help='whether to overwrite existed features files (flag == 1) or use them (flag == 0)', type=int, default=0, const=0, choices=[0, 1])
    parser.add_argument('-overwrite_matching', nargs='?', help='whether to overwrite existed matched files (flag == 1) or use them (flag == 0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-overwrite_first_diffacto', nargs='?', help='whether to overwrite existed diffacto files (flag == 1) or use them (flag == 0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-mixed', nargs='?', help='whether to reanalyze mixed intensities (1) or not (0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-venn', nargs='?', help='whether to plot venn diagrams (1) or not (0)', type=int, default=1, const=1, choices=[0, 1])
    parser.add_argument('-pept_intens', nargs='?', help='max_intens - as intensity for peptide would be taken maximal intens between charge states; summ_intens - as intensity for peptide would be taken sum of intens between charge states; z-attached - each charge state would be treated as independent peptide', type=str, default='z-attached', const='z-attached', choices=['z-attached', 'summ_intens', 'max_intens'])
    parser.add_argument('-choice', nargs='?', help='method how to choose right intensities for peptide. 0 - default order and min Nan values, 1 - min Nan and min of sum CV, 2 - min Nan and min of max CV, 3 - min Nan and min of squared sum CV, 4 - min Nan and min of squared sum of corrected CV', type=int, default=4, const=4, choices=[0, 1, 2, 3, 4,])
    parser.add_argument('-norm', nargs='?', help='normalization method for intensities. Can be 1 - median or 0 - no normalization', type=int, default=0, const=0, choices=[0, 1])
    parser.add_argument('-isotopes', help='monoisotope error', nargs='+', type=int, default=[0,1,-1,2,-2])
    parser.add_argument('-outPept', nargs='?', help='name of output diffacto peptides file (important: .txt)', type=str, default='peptides.txt', const='peptides.txt')
    parser.add_argument('-outSampl', nargs='?', help='name of output diffacto samples file (important: .txt)', type=str, default='sample.txt', const='sample.txt')
    parser.add_argument('-outDiff', nargs='?', help='name of diffacto output file (important: .txt)', type=str, default='diffacto_out.txt', const='diffacto_out.txt')
    parser.add_argument('-min_samples', nargs='?', help='minimum number of samples for peptide usage, -1 means that half of the given number of files would be used', type=int, default=-1, const=-1)
    
    parser.add_argument('-mbr', nargs='?', help='match between runs (1 - on, 0 - off)', type=int, default=0, const=0, choices=[0, 1])
    parser.add_argument('-pval_threshold', nargs='?', help='P-value threshold for reliable differetially expressed proteins', type=float, default=0.05, const=0.05)
    parser.add_argument('-fc_threshold', nargs='?', help='Fold change threshold for reliable differetially expressed proteins', type=float, default=2., const=2.)
    parser.add_argument('-dynamic_fc_threshold', nargs='?', help='whether to apply dynamically calculated threshold (1) or not and use static -fc_threshold (0) ', type=int, default=1, const=1, choices=[0, 1])
    
    parser.add_argument('-diffacto_args', nargs='?', help='String of additional arguments to submit into Diffacto (hole string in single quotes in command line) except: -i, -out, -samples, -min_samples; default: "-normalize median -impute_threshold 0.25" ', type=str, default='-normalize median -impute_threshold 0.25', const='')
    parser.add_argument('-dino_args', nargs='?', help='String of additional arguments to submit into Dinosaur (hole string in single quotes in command line) except: --outDir --outName; default: ""', type=str, default='', const='')
#    parser.add_argument('-bio_args', nargs='?', help='String of additional arguments to submit into Biosaur (hole string in single quotes in command line) except: -o; default: ""', type=str, default='', const='')
    parser.add_argument('-bio2_args', nargs='?', help='String of additional arguments to submit into Biosaur2 (hole string in single quotes in command line) except: -o; default: "-hvf 1000 -minlh 3"', type=str, default='-hvf 1000 -minlh 3', const='')
    parser.add_argument('-openms_args', nargs='?', help='String of additional arguments to submit into OpenMSFeatureFinder (hole string in single quotes in command line) except: -in, -out; default: "-algorithm:isotopic_pattern:charge_low 2 -algorithm:isotopic_pattern:charge_high 7"', type=str, default='-algorithm:isotopic_pattern:charge_low 2 -algorithm:isotopic_pattern:charge_high 7', const='')

#    parser.add_argument('-version', action='version', version='%s' % (pkg_resources.require("scavager")[0], ))
    args = vars(parser.parse_args())
    default_config = vars(parser.parse_args([]))
    users_config = {}
    if args['cfg'] :
        if os.path.exists(args['cfg']) :
            if args['cfg_category'] :
                s = read_cgf(args['cfg'] , args['cfg_category'] )
            else :
                s = read_cgf( args['cfg'], default_config['cfg_category'] )
            users_config = vars(parser.parse_args(s))
        else :
            logging.critical('path to config file does not exist')
            return -1
    
    if users_config :
        for k in users_config :
            if users_config[k] != default_config[k] and args[k] == default_config[k] :
                args.update({k: users_config[k]})
    
    loglevel = args['logs']
    if args['log_path'] :
        lst = args['log_path'].split('/')[:-1]
        log_directory = '/' + str(os.path.join(*lst))
        subprocess.call(['mkdir', '-p', log_directory])
        
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s]: %(message)s',
        level=numeric_level,
        handlers=[  logging.FileHandler(args['log_path'], mode='w', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout) ]
    )
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    
    logging.info('Started')
    
    if args['example_cfg'] :
        p = os.path.abspath(args['example_cfg'])
        if os.path.exists(p.rsplit('/', maxsplit=1)[0]) or os.path.exists(p) :
            write_example_cfg(args['example_cfg'], default_config)
            logging.info('Example cfg created')
    
    logging.debug(args)
    
    mode = None
    if not args['s2'] :
        sample_nums = ['s1']
        logging.info('mode = feature matching')
        mode = 'feature matching'
    else :
        sample_nums = ['s1', 's2']
        logging.info('mode = diffacto')
        mode = 'diffacto'
    
    for sample_num in sample_nums :
        if args[sample_num] :
            if type(args[sample_num]) is str :
                args[sample_num] = [x.strip()+'.mzML' for x in re.split(r'\.mzML|\.mzml|\.MZML' , args[sample_num], )][:-1]
            elif type(args[sample_num]) is list :
                pass
            else :
                logging.critical('invalid {} input'.format(sample_num))
                return -1
    
    
#    print(args['s1'].split())

    if not args['dif'] :
        logging.critical('path to diffacto file is required')
        return -1

#     if not args['scav2dif'] :
#         logging.warning('path to scav2diffacto.py file is required')
#         return -1
    
    arg_suff = ['dino', 'bio2', 'openMS']
    suffixes = []
    for suf in arg_suff :
        if args[suf] :
            suffixes.append(suf)
    
    k = len(suffixes)
    if k == 0 :
        logging.critical('At least one feature detector shoud be given!')
        return -1
    elif k == 1 and mode == 'diffacto' :
        logging.info('First diffacto run applied')
        args['venn'] = 0
    elif k >= 2 and mode == 'diffacto' :
        logging.info('Second diffacto run applied')
    else :
        logging.info('No diffacto run applied')
    
    samples = []
    samples_dict = {}
    mzML_paths = []
    for sample_num in sample_nums :
        samples_dict[sample_num] = []
        for z in args[sample_num]:
            mzML_paths.append(z)
            samples.append(z.split('/')[-1].replace('.mzML', ''))
            samples_dict[sample_num].append(z.split('/')[-1].replace('.mzML', ''))
            
    if args['min_samples'] == -1 :
        args['min_samples'] = int(len(samples)/2)
    
    logging.debug('samples_dict = ' + str(samples_dict))

    PSMs_full_paths = []
    PSMs_full_dict = {}
    PSMs_suf = args['PSM_format']
    if args['PSM_folder'] :
        dir_name = args['PSM_folder']
    else :
        logging.warning('trying to find *%s files in the same directory as .mzML', PSMs_suf)
        dir_name = os.path.abspath(mzML_paths[0]).split(samples[0])[0]
    for sample_num in sample_nums :
        PSMs_full_dict[sample_num] = {}
        for sample in samples_dict[sample_num] :
            i = 0
            for filename in os.listdir(dir_name) :
                if filename.startswith(sample) and filename.endswith(PSMs_suf) :
                    PSMs_full_paths.append(os.path.join(dir_name, filename))
                    PSMs_full_dict[sample_num][sample] = os.path.join(dir_name, filename)
                    i += 1
            if i == 0 :
                logging.critical('sample ' + sample + ' PSM file not found')
                return -1
    logging.debug(PSMs_full_dict)
    
    if not args['outdir'] :
        logging.info('Path to output directory is not specified. Using input files directory instead.')
        args['outdir'] = os.path.abspath(mzML_paths[0]).split(samples[0])[0]
    else :
        logging.info('Results are stored at {}'.format(args['outdir']))
    
    mzML_dict = {}
    for sample, mzML in zip(samples, mzML_paths) :
        mzML_dict[sample] = mzML
    
    if mode != 'feature matching' : 
        peptides_dict = {}
        peptides_suf = 'peptides.tsv'
        if args['pept_folder'] and args['pept_folder'] != args['PSM_folder'] :
            if os.path.exists(args['pept_folder']) :
                dir_name = args['pept_folder']
            else :
                logging.critical('path to peptides files folder does not exist')
                return -1
        else :
            logging.warning('trying to find *_%s files in the same directory as PSMs', peptides_suf)
            dir_name = PSMs_full_paths[0].split(sample[0])[0]
            logging.debug(dir_name)
        for sample in samples :
            i = 0
            for filename in os.listdir(dir_name) :
                if filename.startswith(sample) and filename.endswith(peptides_suf) :
                    peptides_dict[sample] = os.path.join(dir_name, filename)
                    logging.debug(os.path.join(dir_name, filename))
                    i += 1
            if i == 0 :
                logging.critical('sample '+ sample + ' peptides file not found in ' + dir_name)
                return -1

        proteins_dict = {}
        proteins_suf = 'proteins.tsv'
        if args['prot_folder'] and args['prot_folder'] != args['PSM_folder'] :
            if os.path.exists(args['prot_folder']) :
                dir_name = args['pept_folder']
            else :
                logging.critical('path to proteins files folder does not exist')
                return -1
        else :
            logging.warning('trying to find *_proteins.tsv files in the same directory as PSMs')
            dir_name = PSMs_full_paths[0].split(sample[0])[0]
        for sample in samples :
            i = 0
            for filename in os.listdir(dir_name) :
                if filename.startswith(sample) and filename.endswith(proteins_suf) :
                    proteins_dict[sample] = os.path.join(dir_name, filename)
                    i += 1
            if i == 0 :
                logging.critical('sample '+ sample + ' proteins file not found')
                return -1
    
    paths = {'mzML': mzML_dict, 
             'PSM_full' : PSMs_full_dict,
            }
    if mode != 'feature matching' :
        paths['peptides'] = peptides_dict
        paths['proteins'] = proteins_dict
#    print(paths)
    out_directory = args['outdir']
    sample_1 = args['s1']
    sample_2 = args['s2']

    # args['overwrite_features'] = int( args['overwrite_features'])
    # args['overwrite_matching'] = int( args['overwrite_matching'])
    # args['overwrite_first_diffacto'] = int( args['overwrite_first_diffacto'])
    # args['mixed'] = int( args['mixed'])
    # args['venn'] = int( args['venn'])
    # args['choice'] = int( args['choice'])
    # args['norm'] = int( args['norm'])
    # args['isotopes'] = [int(isoval.strip()) for isoval in args['isotopes'].split(',')]
    # args['pval_threshold'] = float(args['pval_threshold'])
    # args['fc_threshold'] = float(args['fc_threshold'])
    # args['mbr'] = int(args['mbr'])
    # if args['dynamic_fc_threshold'] == '1' :
    #     args['dynamic_fc_threshold'] = True
    # elif args['dynamic_fc_threshold'] == '0' :
    #     args['dynamic_fc_threshold'] = False
    # else :
    #     logging.critical('Invalid value for setting: -dynamic_fc_threshold %s', args['dynamic_fc_threshold'])
    #     raise ValueError('Invalid value for setting: -dynamic_fc_threshold %s', args['dynamic_fc_threshold'])

        
    
    logging.debug('PSMs_full_paths: %s', PSMs_full_paths)
    logging.debug('mzML_paths: %s', mzML_paths)
    logging.debug('out_directory: %s', out_directory)
    logging.debug('suffixes: %s', suffixes)
    logging.debug('sample_1: %s', sample_1)
    logging.debug('sample_2: %s', sample_2)
    logging.debug('mixed = %s', args['mixed'])
    logging.debug('venn = %s', args['venn'])
    logging.debug('choice = %s', args['choice'])
    logging.debug('overwrite_features = %s', args['overwrite_features'])
    logging.debug('overwrite_first_diffacto = %s', args['overwrite_first_diffacto'])
    logging.debug('overwrite_matching = %d', args['overwrite_matching'])

    subprocess.call(['mkdir', '-p', out_directory])

## Генерация фич
    if args['feature_folder'] :
        if os.path.exists(args['feature_folder']) :
            feature_path = args['feature_folder']
        else :
            logging.warning('Path to feature files folder does not exist. Creating it.')
            feature_path =  args['feature_folder']
    else :
        if args['PSM_folder'] :
            dir_name = args['PSM_folder']
        else :
            dir_name = PSMs_full_paths[0].split(sample[0])[0]
        feature_path = os.path.join(dir_name,  'features')
    subprocess.call(['mkdir', '-p', feature_path])


### Dinosaur

# На выходе добавляет в папку feature_path файлы *sample*_features_dino.tsv

    if args['dino'] :
        for path, sample in zip(mzML_paths, samples) :
            outName = sample + '_features_' + 'dino' + '.tsv'
            if args['overwrite_features'] == 1 or not os.path.exists(os.path.join(feature_path, outName)) :
                logging.info('\n' + 'Writing features' + ' dino ' + sample + '\n')
                exitscore = call_Dinosaur(args['dino'], path, feature_path, outName, args['dino_args'])
                logging.debug(exitscore)
                os.rename(os.path.join(feature_path, outName + '.features.tsv'),  os.path.join(feature_path, outName) )
            else :
                logging.info('\n' + 'Not overwriting features ' + ' dino ' + sample + '\n')

### Biosaur2

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio2.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio2'] :
        for path, sample in zip(mzML_paths, samples) :
            outPath = os.path.join(feature_path, sample + '_features_bio2.tsv')
            if args['overwrite_features'] == 1 or not os.path.exists(outPath) :
                logging.info('\n' + 'Writing features ' + ' bio2 ' + sample + '\n')
                exitscore = call_Biosaur2(args['bio2'], path, outPath, args['bio2_args'])
                logging.debug(exitscore)
            else :
                logging.info('\n' + 'Not overwriting features ' + ' bio2 ' + sample + '\n')

            
### OpenMS

# На выходе создает в feature_path папку OpenMS с файлами *.featureXML и добавляет в папку out_directory/features файлы *sample*_features_openMS.tsv
    
    if args['openMS'] :
        out_path = os.path.join(feature_path, 'openMS')
        subprocess.call(['mkdir', '-p', out_path])
            
        for path, sample in zip(mzML_paths, samples) :
            out_path = os.path.join(feature_path, 'openMS', sample + '.featureXML')
            if args['overwrite_features'] == 1 or not os.path.exists(out_path) :
                logging.info('\n' + 'Writing .featureXML ' + ' openMS ' + sample + '\n')
                exitscore = call_OpenMS(args['openMS'], path, out_path, args['openms_args'])
                logging.debug(exitscore)
            else :
                logging.info('\n' + 'Not ovetwriting .featureXML ' + ' openMS ' + sample + '\n')

        for path, sample in zip(mzML_paths, samples) :
            out_path = os.path.join(feature_path, 'openMS', sample + '.featureXML')
            o = os.path.join(feature_path, sample + '_features_' + 'openMS.tsv')
            if args['overwrite_features'] == 1 or not os.path.exists(o) : 
                logging.info('Writing features ' + ' openMS ' + sample)
                a = featurexml.read(out_path)

                features_list = []
                for z in a : 
                    mz = float(z['position'][1]['position'])
                    # rtApex = float(z['position'][0]['position']) / 60
                    rtStart = float(z['convexhull'][0]['pt'][0]['x'])/60
                    rtEnd = float(z['convexhull'][0]['pt'][1]['x'])/60
                    intensityApex = float(z['intensity'])
                    charge = int(z['charge'])
                    feature_index = z['id']
                    features_list.append([feature_index, mz, charge, rtStart,rtEnd, intensityApex])
                b = pd.DataFrame(features_list, columns = ['id', 'mz', 'charge', 'rtStart', 'rtEnd', 'intensityApex'])
                b.to_csv(o, sep='\t', encoding='utf-8')
            else :
                logging.info('Not overwriting features ' + ' openMS ' + sample + '\n')


### Сопоставление

    if args['matching_folder'] :
        if os.path.exists(args['matching_folder']) :
            matching_path = args['matching_folder']
        else :
            logging.critical('Path to matching_folder does not exists')
            return -1
    else :
        matching_path = os.path.join(out_directory, 'feats_matched')
    subprocess.call(['mkdir', '-p', matching_path])

    logging.info('Start matching features')
    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        PSM = read_PSMs(PSM_path)
        logging.info('sample %s', sample)
        for suf in suffixes :
            if args['overwrite_matching'] == 1 or not os.path.exists(os.path.join(matching_path, sample + '_' + suf + '.tsv')) :
                feats = pd.read_csv( os.path.join(feature_path, sample + '_features_' + suf + '.tsv'), sep = '\t')
                feats = feats.sort_values(by='mz')

                logging.info(suf + ' features ' + sample + '\n' + 'START')
                temp_df = optimized_search_with_isotope_error_(feats, PSM, isotopes_array=args['isotopes'])[0]
                # temp_df = optimized_search_with_isotope_error_(feats, PSM, mean_rt1=0,sigma_rt1=1e-6,mean_rt2=0,sigma_rt2=1e-6,mean_mz = False,sigma_mz = False,mean_im = False,sigma_im = False, isotopes_array=[0,1,-1,2,-2])[0]

                if args['mbr']:	
                    temp_df = mbr(feats, temp_df, PSMs_full_paths, PSM_path)
                

                median = temp_df['feature_intensityApex'].median()
                temp_df['med_norm_feature_intensityApex'] = temp_df['feature_intensityApex']/median
                cols = list(temp_df.columns)
                
                logging.info(suf + ' features ' + sample + ' DONE')
                temp_df.to_csv(os.path.join(matching_path, sample + '_' + suf + '.tsv'), sep='\t', columns=cols)
                logging.info(sample + ' PSMs matched ' + str(temp_df['feature_intensityApex'].notna().sum()) + '/' 
                             + str(len(temp_df)) + ' ' + str(round(temp_df['feature_intensityApex'].notna().sum()/len(temp_df)*100, 2)) + '%')
                logging.info(suf + ' MATCHED')
                
        temp_df = None   
    
    paths['feats_matched'] = {}
    for sample in samples :
        feats_matched_paths = {}
        for suf in suffixes :
            s = os.path.join(matching_path, sample + '_' + suf + '.tsv')
            if os.path.exists(s):
                feats_matched_paths[suf] = s
            else :
                logging.critical('File not found: %s', s)
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), s)
                return -1
        paths['feats_matched'][sample] = feats_matched_paths
    logging.info('Matching features for PSMs done')

    
    ## Подготовка и прогон Диффакто 1

    if mode == 'diffacto' :
        logging.info('Going for quantitative analysis with diffacto')
        if args['diffacto_folder'] :
            if os.path.exists(args['diffacto_folder']) :
                diffacto_folder = args['diffacto_folder']
            else :
                diffacto_folder = os.path.join(out_directory, 'diffacto')
                logging.warning('Path to diffacto results does not exists, using default one:', diffacto_folder)
        else :
            diffacto_folder = os.path.join(out_directory, 'diffacto')
        subprocess.call(['mkdir', '-p', diffacto_folder])

        intens_colomn_name = 'feature_intensityApex'
        if args['norm'] == 1 :
            intens_colomn_name = 'med_norm_feature_intensityApex'
        else :
            intens_colomn_name = 'feature_intensityApex'

        allowed_prots = set()
        allowed_peptides = set()
        for sample in samples :
            df0 = pd.read_table(paths['proteins'][sample])
            allowed_prots.update(df0['dbname'])
        for sample in samples :
            df0 = pd.read_table(paths['peptides'][sample])
            allowed_peptides.update(df0['peptide'])
        logging.debug('allowed proteins total: %d', len(allowed_prots))
        logging.debug('allowed peptides total: %d', len(allowed_peptides))

        paths['DiffPept'] = {}
        paths['DiffSampl'] = {}
        paths['DiffOut'] = {}
        for suf in suffixes:
            logging.info('Charge states processing %s', suf)
            psms_dict = {}
            for sample in samples:
                logging.debug('Starting %s', sample)
                if args['logs'] == 'DEBUG' :
                    charge_faims_intensity_path = os.path.join(diffacto_folder, 'charge_faims_intensity')
                    subprocess.call(['mkdir', '-p', charge_faims_intensity_path])
                    out_path = os.path.join(diffacto_folder, 'charge_faims_intensity', sample+'_'+suf+'.tsv')
                else :
                    out_path = None
                psms_dict[sample] = charge_states_intensity_processing(
                    paths['feats_matched'][sample][suf],
                    method=args['pept_intens'], 
                    intens_colomn_name='feature_intensityApex', 
                    allowed_peptides=allowed_peptides, # set()
                    allowed_prots=allowed_prots, # set()
                    out_path=out_path
                )
                logging.debug('Done %s', sample)

            paths['DiffPept'][suf] = os.path.join(diffacto_folder, args['outPept'].replace('.txt', '_' + suf + '.txt'))
            paths['DiffSampl'][suf] = os.path.join(diffacto_folder, args['outSampl'].replace('.txt', '_' + suf + '.txt'))
            paths['DiffOut'][suf] = os.path.join(diffacto_folder, args['outDiff'].replace('.txt', '_' + suf + '.txt'))
            if args['overwrite_first_diffacto'] == 1 or not os.path.exists( paths['DiffOut'][suf] ) :
                exitscore = diffacto_call(diffacto_path = args['dif'],
                                          out_path = paths['DiffOut'][suf],
                                          peptide_path = paths['DiffPept'][suf],
                                          sample_path = paths['DiffSampl'][suf],
                                          min_samples = args['min_samples'],
                                          psm_dfs_dict = psms_dict,
                                          samples_dict = samples_dict,
                                          write_peptides=True,
                                          str_of_other_args = args['diffacto_args']
                                         )  
            logging.info('Done Diffacto run with %s', suf)
            psms_dict = False
            
        if k <= 2 or args['mixed'] != 1 :
            save = True
            if k <= 2 or args['venn'] != 1 :
                plot_venn = False
            else :
                plot_venn = True
        else :
            save = False
            plot_venn = False

        num_changed_prots = generate_users_output(
            diffacto_out = paths['DiffOut'], 
            out_folder = out_directory, 
            plot_venn = plot_venn, 
            pval_threshold = args['pval_threshold'], 
            fc_threshold = args['fc_threshold'], 
            dynamic_fc_threshold = args['dynamic_fc_threshold'], 
            save = save
        )
        
        if args['choice'] == 0 :
            default_order = sorted(suffixes, key= lambda x: num_changed_prots[x], reverse=True)
        else : 
            default_order = False
        
        
        ## Второй прогон диффакто
        
        
        if k >= 2 and args['mixed'] == 1 :
            suf = 'mixed'
            logging.info('Mixing intensities STARTED')
            
            to_diffacto = os.path.join(diffacto_folder, args['outPept'].replace('.txt', '_' + suf + '.txt'))
            a = mix_intensity(paths['DiffPept'],
                              samples_dict,
                              choice=args['choice'], 
                              suf_dict={'dino':'D', 'bio2':'B2', 'openMS':'O', 'mixed':'M'}, 
                              out_dir= charge_faims_intensity_path,
                              default_order= default_order,
                              to_diffacto= to_diffacto, 
                              )
            
            logging.info('Mixing intensities DONE with exitscore {}'.format(a))
            
            paths['DiffPept'][suf] = to_diffacto
            paths['DiffSampl'][suf] = os.path.join(diffacto_folder, args['outSampl'].replace('.txt', '_' + suf + '.txt'))
            paths['DiffOut'][suf] = os.path.join(diffacto_folder, args['outDiff'].replace('.txt', '_' + suf + '.txt'))
            
            logging.info('Diffacto START')        
            exitscore = diffacto_call(diffacto_path = args['dif'],
                                      out_path = paths['DiffOut'][suf],
                                      peptide_path = paths['DiffPept'][suf],
                                      sample_path = paths['DiffSampl'][suf],
                                      min_samples = args['min_samples'],
                                      psm_dfs_dict = {},
                                      samples_dict = samples_dict,
                                      str_of_other_args = args['diffacto_args']
                                     )
            logging.info('Done Diffacto run with {} with exitscore {}'.format(suf, exitscore))
            
            save = True
            if args['venn'] != 1 :
                plot_venn = False
            else :
                plot_venn = True

            num_changed_prots = generate_users_output(
                diffacto_out = paths['DiffOut'], 
                out_folder = out_directory, 
                plot_venn = plot_venn, 
                pval_threshold = args['pval_threshold'], 
                fc_threshold = args['fc_threshold'], 
                dynamic_fc_threshold = args['dynamic_fc_threshold'], 
                save = save
            )
            logging.info('Numbers of differentially expressed proteins:')
            for suf in suffixes+['mixed'] :
                logging.info('{}: {}'.format(suf, num_changed_prots[suf]))
                
            logging.info('IQMMA finished')

if __name__ == '__main__':
    run()

