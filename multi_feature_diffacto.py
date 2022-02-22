import argparse
import subprocess
import configparser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pyteomics
import copy
import os
import time
from os import listdir
from matplotlib.ticker import PercentFormatter
from matplotlib_venn import venn3, venn3_circles
from pyteomics.openms import featurexml

import venn
from venn import venn
    

def run():
    parser = argparse.ArgumentParser(
        description = 'run multiple feature detection matching and diffacto for scavager results',
        epilog = '''
    Example usage
    -------------
    $ multi_features_diffacto_analisys.py -mzML sample1_1.mzML sample1_n.mzML sample2_1.mzML sample1_n.mzML 
                                          -S1 sample1_1_PSMs_full.tsv sample1_n_PSMs_full.tsv 
                                          -S2 sample2_1_PSMs_full.tsv sample2_n_PSMs_full.tsv
                                          -sampleNames sample1_1 sample1_n sample2_1 sample2_n
                                          -outDir ./script_out
                                          -dif path_to/diffacto
                                          -dino /home/bin/dinosaur
                                          -bio /home/bin/biosaur
                                          -bio2 /home/bin/biosaur2
                                          -openMS path_to_openMS
    or
    $ multi_features_diffacto_analisys.py @default.txt
    or 
    $ multi_features_diffacto_analisys.py -cfg /path_to/default.ini
    -------------
    ''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    
    parser.add_argument('-cfg', help='path to config .ini file')
    parser.add_argument('-dif', help='path to Diffacto')
    parser.add_argument('-scav2dif', help='path to scav2diffacto')
    parser.add_argument('-S1', nargs='+', help='input files for S1 sample')
    parser.add_argument('-S2', nargs='+', help='input files for S2 sample')
    parser.add_argument('-sampleNames', nargs='+', help='short names for samples for inner structure of results')
    parser.add_argument('-mzML', nargs='+', 
                        help='paths to mzML files for samples in the same order in one line S1 the S2...')
    
    parser.add_argument('-dino', help='path to Dinosaur')
    parser.add_argument('-bio', help='path to Biosaur')
    parser.add_argument('-bio2', help='path to Biosaur2')
    parser.add_argument('-openMS', help='path to OpenMS feature')
    
    parser.add_argument('-S3', nargs='+', help='input files for S3 sample')
    parser.add_argument('-S4', nargs='+', help='input files for S4 sample')
    
    parser.add_argument('-outDir', help='name of  directory, where results would be stored', default='./MultiFeat_diffacto')
    parser.add_argument('-overwrite_features', help='whether to overwrite existed files or use them (without this flag)', action='store_true')
    parser.add_argument('-overwrite_matching', help='whether to overwrite existed files or use them (without this flag)', action='store_true')
    parser.add_argument('-mixed', help='whether to reanalyze mixed intensities or not', action='store_true')
    parser.add_argument('-venn', help='whether to plot venn diagrams or not', action='store_true')
    
    parser.add_argument('-outPept', help='name of output diffacto peptides file (important: .txt)', default='peptides.txt')
    parser.add_argument('-outSampl', help='name of output diffacto samples file (important: .txt)', default='sample.txt')
    parser.add_argument('-outDiff', help='name of diffacto output file (important: .txt)', default='diffacto_out.txt')
    parser.add_argument('-norm', help='normalization method. Can be average, median, GMM or None', default='None')
    parser.add_argument('-impute_threshold', help='impute_threshold for missing values fraction', default='0.25')
    parser.add_argument('-min_samples', help='minimum number of samples for peptide usage', default='3')
#    parser.add_argument('-version', action='version', version='%s' % (pkg_resources.require("scavager")[0], ))
    args = vars(parser.parse_args())


    '''    if args['cfg'] :
            config = configparser.RawConfigParser()
            with open(os.path.abspath(args['cfg']), 'r') as f :
                config.read(f)
                print(config.sections())
            print('user' in config)
            print(set(args.keys())&set(config['user'].keys()))
            new_keys = [key for key in set(args.keys())&set(config['user'].keys()) if (args[key]==None)]  
            for key in new_keys :
                args[key] = config['user'][key]
    #        print(new_keys)
            new_keys = [key for key in set(args.keys())&set(config['DEFAULT'].keys()) if ((args[key]==None) and (config['DEFAULT'][key]!=None))]
            for key in new_keys :
                args[key] = config['DEFAULT'][key]
    #        print(new_keys)
        print(args)
    '''
    if not args['dif'] :
        print('path to diffacto file is required')
        return -1

    if not args['scav2dif'] :
        print('path to scav2diffacto.py file is required')
        return -1
    
    suffixes = []
    k = 0
    if args['dino'] :
        print(args['dino'])
        suffixes.append('dino')
    else :
        k += 1
    if args['bio'] :
        print(args['bio'])
        suffixes.append('bio')
    else :
        k += 1
    if args['bio2'] :
        print(args['bio2'])
        suffixes.append('bio2')
    else :
        k += 1
    if args['openMS'] :
        print(args['openMS'])
        suffixes.append('openMS')
    else :
        k += 1
    if k == 4 :
        print('At least one feature finder shoud be given!')
        return -1
    if k == 3 :
        print('One diffacto run')
    if k <= 2 :
        print('Second diffacto run is applied')
    
    PSMs_full_paths = []
    for sample_num in ['S1', 'S2']:
        if args[sample_num] :
            for z in args[sample_num]:
                PSMs_full_paths.append(z)
        else :
            print('sample '+ sample_num + ' *_PSMs_full.tsv files are required')
            return -1

    if args['mzML'] and (len(args['mzML']) == len(PSMs_full_paths)) :
        mzML_paths = args['mzML']
    else :
        print('paths to all .mzML files are required')
        return -1
    
    if args['sampleNames'] and (len(args['sampleNames']) == len(PSMs_full_paths)) :
        samples = args['sampleNames']
    else :
        print('short name for every PSMs file is required')
        return -1
    
    out_directory = args['outDir']
    sample_1 = args['S1']
    sample_2 = args['S2']
    
    print(PSMs_full_paths)
    print(mzML_paths)
    print(out_directory)
    print(suffixes)
    print(sample_1)
    print(sample_2)
       
#    subprocess.call(['python3', args['dif'], '-i', args['peptides'], '-samples', args['samples'], '-out',\
#     args['out'], '-normalize', args['norm'], '-impute_threshold', args['impute_threshold'], '-min_samples', args['min_samples']])

    
    subprocess.call(['mkdir', '-p', out_directory])

## Генерация фич

### Dinosaur

    outDir = out_directory + '/features'
    subprocess.call(['mkdir', '-p', outDir])

# На выходе добавляет в папку out_directory/features файлы *sample*_features_dino.tsv

    if args['dino'] :
        if args['overwrite_features'] :
            for path, sample in zip(mzML_paths, samples) :
                outName = sample + '_features_' + 'dino' + '.tsv'
                outName_false = outName + '.features.tsv'
                with open(os.path.join(outDir, outName_false), mode='w+', buffering=- 1, encoding='utf-8') :
                    subprocess.call([args['dino'], '--outDir', outDir, '--outName', outName, path])
                os.rename(outDir + '/' + outName + '.features.tsv', outDir + '/' + outName)

        else :
            for path, sample in zip(mzML_paths, samples) :
                outName = sample + '_features_' + 'dino' + '.tsv'
                outName_false = outName + '.features.tsv'
                if not os.path.exists(outDir + '/' + outName) :
                    with open(os.path.join(outDir, outName_false), mode='w+', buffering=- 1, encoding='utf-8') :
                        subprocess.call([args['dino'], '--outDir', outDir, '--outName', outName, path])
                    os.rename(outDir + '/' + outName + '.features.tsv', outDir + '/' + outName)

### Biosaur

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio'] :
        if args['overwrite_features'] :
            for path, sample in zip(mzML_paths, samples) :
                outPath = out_directory + '/features/' + sample + '_features_bio.tsv'
                subprocess.call([args['bio'], path, '-out', outPath, '-hvf', '1000'])
        else :
            for path, sample in zip(mzML_paths, samples) :
                outPath = out_directory + '/features/' + sample + '_features_bio.tsv'
                if not os.path.exists(outPath) :
                    subprocess.call([args['bio'], path, '-out', outPath, '-hvf', '1000'])

### Biosaur2

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio2.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio2'] :
        if args['overwrite_features'] :
            for path, sample in zip(mzML_paths, samples) :
                outPath = out_directory + '/features/' + sample + '_features_bio2.tsv'
                subprocess.call([args['bio2'], path, '-o', outPath, '-hvf', '1000', '-minlh', '3'])
        else :
            for path, sample in zip(mzML_paths, samples) :
                outPath = out_directory + '/features/' + sample + '_features_bio2.tsv'
                if not os.path.exists(outPath) : 
                    subprocess.call([args['bio2'], path, '-o', outPath, '-hvf', '1000', '-minlh', '3'])
            
### OpenMS

# На выходе создает в out_directory папку OpenMS с файлами *.featureXML и добавляет в папку out_directory/features файлы *sample*_features_openMS.tsv
    
    if args['openMS'] :
        out_path = out_directory + '/openMS/'
        subprocess.call(['mkdir', '-p', out_path])
        
        if args['overwrite_features'] :
            for path, sample in zip(mzML_paths, samples) :
                out_path = out_directory + '/openMS/' + sample + '.featureXML'
                subprocess.call([args['openMS'], '-in', path, '-out', out_path, '-algorithm:isotopic_pattern:charge_low', '2', '-algorithm:isotopic_pattern:charge_high', '7'])

            for path, sample in zip(mzML_paths, samples) :
                out_path = out_directory + '/openMS/' + sample + '.featureXML'
                a = featurexml.read(out_path)

                features_list = []
                for z in a : 
                    mz = float(z['position'][1]['position'])
                    rtApex = float(z['position'][0]['position']) / 60
                    intensityApex = float(z['intensity'])
                    charge = int(z['charge'])
                    feature_index = z['id']
                    features_list.append([feature_index, mz, charge, rtApex, intensityApex])
                b = pd.DataFrame(features_list, columns = ['id', 'mz', 'charge', 'rtApex', 'intensityApex'])
                b.to_csv(os.path.join(out_directory + '/features', sample + '_features_' + 'openMS.tsv'), sep='\t', encoding='utf-8')
                
        else :
            for path, sample in zip(mzML_paths, samples) :
                out_path = out_directory + '/openMS/' + sample + '.featureXML'
                if not os.path.exists(out_path) : 
                    subprocess.call([args['openMS'], '-in', path, '-out', out_path, '-algorithm:isotopic_pattern:charge_low', '2', '-algorithm:isotopic_pattern:charge_high', '7'])

            for path, sample in zip(mzML_paths, samples) :
                out_path = out_directory + '/openMS/' + sample + '.featureXML'
                if not os.path.exists(os.path.join(out_directory + '/features', sample + '_features_' + 'openMS.tsv')) : 
                    a = featurexml.read(out_path)
                    features_list = []
                    for z in a : 
                        mz = float(z['position'][1]['position'])
                        rtApex = float(z['position'][0]['position']) / 60
                        intensityApex = float(z['intensity'])
                        charge = int(z['charge'])
                        feature_index = z['id']
                        features_list.append([feature_index, mz, charge, rtApex, intensityApex])
                    b = pd.DataFrame(features_list, columns = ['id', 'mz', 'charge', 'rtApex', 'intensityApex'])
                    b.to_csv(os.path.join(out_directory + '/features', sample + '_features_' + 'openMS.tsv'), sep='\t', encoding='utf-8')


## Функции для сопоставления
    
                                
    def find_feature_for_psm_1(features_db, mz, z, rt, accuracy_ppm=10, accuracy_minut = 0.5, openMS = False) :
        if not openMS :
            features = features_db.copy()
            features.sort_values('mz', inplace = True)
            dm = accuracy_ppm*mz/1e6
            l = mz - dm
            r = mz + dm
            feature = features_db.query('`mz` >= @l and `mz` <= @r and (`charge` == @z) and (`rtStart` <= @rt) and (`rtEnd` >= @rt)')
            
            if not feature.empty :
                feature_index = feature.index[0]
                mz = feature['mz'].values[0]
                charge = feature['charge'].values[0]
                rtStart = feature['rtStart'].values[0]
                rtEnd = feature['rtEnd'].values[0]
                feature_intensityApex = feature['intensityApex'].values[0]
                num_matches = len(feature)
            else :
                feature_index, mz, charge, rtStart, rtEnd, feature_intensityApex, num_matches = None, None, None, None, None, None, None
            return (feature_index, mz, charge, rtStart, rtEnd, feature_intensityApex, num_matches)
        else :
            features = features_db.copy()
            features.sort_values('mz', inplace = True)
            
            dm = accuracy_ppm*mz/1e6
            l = mz - dm
            r = mz + dm
                                    
            dt = accuracy_minut
            l_rt = rt - dt
            r_rt = rt + dt
            
            feature = features_db.query('(`mz` >= @l_mz) and (`mz` <= @r_mz) and (`charge` == @z) and (`rtApex` <= @r_rt) and (`rtApex` >= @l_rt)')
            if not feature.empty :
                feature_index = feature['id'].values[0]
                mz = feature['mz'].values[0]
                charge = feature['charge'].values[0]
                rtApex = feature['rtApex'].values[0]
                feature_intensityApex = feature['intensityApex'].values[0]
                num_matches = len(feature)
            else :
                feature_index, mz, charge, rtApex, feature_intensityApex, num_matches = None, None, None, None, None, None
            return (feature_index, mz, charge, rtApex, feature_intensityApex, num_matches)
    
    
    def feature_for_psm_files_1(features_db, psms_db, accuracy_ppm=10, accuracy_minut = 0.5, openMS = False) :
        not_matched = 0
        i = 0
        length = len(psms_db)
        seconds = time.time()
        local_time = time.ctime(seconds)
        point = length // 10
        features_for_psm = [] # 9 []
    # pd.DataFrame(colomns = ['PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 'PSM_charge', 'feature_charge', 'feature_rtStart', 'PSM_rt_exp', 'feature_rtEnd'])
        for index, row in psms_db.iterrows() :

            psm_index = index
            peptide = row['peptide']
            psm_mass = row['calc_neutral_pep_mass']
            psm_charge = row['assumed_charge']
            psm_rt = row['RT exp']
            psm_mz = (psm_mass+psm_charge*1.00697)/psm_charge
            protein = row['protein']
            
            if not openMS :
                feature_index, feature_mz, feature_charge, feature_rtStart, feature_rtEnd, feature_intensityApex, num_features_matched = find_feature_for_psm_1(features_db, psm_mz, 
                                                                                                                                        psm_charge, psm_rt, accuracy_ppm=accuracy_ppm, 
                                                                                                                                        accuracy_minut = accuracy_minut, openMS = False)

                features_for_psm.append([peptide, protein, psm_index, feature_index, psm_mz, feature_mz, psm_charge, 
                                     feature_charge, feature_rtStart, psm_rt, feature_rtEnd, feature_intensityApex, num_features_matched])
            else :
                feature_index, feature_mz, feature_charge, feature_rtApex, feature_intensityApex, num_features_matched = find_feature_for_psm_openMS(features_db, psm_mz, 
                                                                                                                        psm_charge, psm_rt, accuracy_ppm=accuracy_ppm, 
                                                                                                                        accuracy_minut=accuracy_minut, openMS = True)
        
                features_for_psm.append([peptide, protein, psm_index, feature_index, psm_mz, feature_mz, psm_charge, 
                                 feature_charge, psm_rt, feature_rtApex, feature_intensityApex, num_features_matched])
            
            if num_features_matched == None :
                not_matched += 1

            i += 1
            if i % point == 0 :
                seconds = time.time()
                local_time = time.ctime(seconds)
                print(i,  'from', length, local_time)
        print('Number of PMS without features:', not_matched)
        if not openMS :
            cols = ['peptide', 'protein', 'PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 
                    'PSM_charge', 'feature_charge', 'feature_rtStart', 'PSM_rt_exp', 'feature_rtEnd',
                    'feature_intensityApex', 'num_features_matched']
        else :
            cols = ['peptide', 'protein', 'PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 
                    'PSM_charge', 'feature_charge', 'PSM_rt_exp', 'feature_rtApex',
                    'feature_intensityApex', 'num_features_matched']
        features_for_psm_db = pd.DataFrame(features_for_psm, columns = cols )
        return features_for_psm_db


### Сопоставление

                                    
    a = out_directory + '/feats_matched'
    subprocess.call(['mkdir', '-p', a])

#    suffixes = ['dino', 'bio', 'bio2', 'openMS'] - уже заданы

    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        PSM = pd.read_csv(PSM_path, sep = '\t')
        print('sample', sample)
        for suff in suffixes :
            if args['overwrite_matching'] :
                feats = pd.read_csv( out_directory + '/features/' + sample + '_features_' + suff + '.tsv', sep = '\t')    
                print(suff, 'features', sample, '\n', 'START')
                if suf == 'openMS' :
                    temp_df = feature_for_psm_files_1( feats, PSM , accuracy_ppm = 10, openMS = True )
                    cols = ['peptide', 'protein', 'PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 
                            'PSM_charge', 'feature_charge', 'PSM_rt_exp', 'feature_rtApex',
                            'feature_intensityApex', 'num_features_matched']
                else :
                    temp_df = feature_for_psm_files_1( feats, PSM , accuracy_ppm = 10, openMS = False )
                    cols = ['peptide', 'protein', 'PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 
                            'PSM_charge', 'feature_charge', 'feature_rtStart', 'PSM_rt_exp', 'feature_rtEnd',
                            'feature_intensityApex', 'num_features_matched']
                
                print(suff, 'features', sample, 'DONE')
                temp_df.to_csv(out_directory + '/feats_matched/' + sample + '_' + suff + '.tsv', sep='\t', columns=cols)
                print(sample, 'PSMs matched' , temp_df['feature_intensityApex'].notna().sum() )
            
            else : 
                if not os.path.exists(out_directory + '/feats_matched/' + sample + '_' + suff + '.tsv') :  
                    feats = pd.read_csv( out_directory + '/features/' + sample + '_features_' + suff + '.tsv', sep = '\t')    
                    print(suff, 'features', sample, '\n', 'START')
                    if suf == 'openMS' :
                        temp_df = feature_for_psm_files_1( feats, PSM , accuracy_ppm = 10, openMS = True )
                        cols = ['peptide', 'protein', 'PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 
                                'PSM_charge', 'feature_charge', 'PSM_rt_exp', 'feature_rtApex',
                                'feature_intensityApex', 'num_features_matched']
                    else :
                        temp_df = feature_for_psm_files_1( feats, PSM , accuracy_ppm = 10, openMS = False )
                        cols = ['peptide', 'protein', 'PSM_index', 'feature_index', 'PSM_mz', 'feature_mz', 
                                'PSM_charge', 'feature_charge', 'feature_rtStart', 'PSM_rt_exp', 'feature_rtEnd',
                                'feature_intensityApex', 'num_features_matched']

                    print(suff, 'features', sample, 'DONE')
                    temp_df.to_csv(out_directory + '/feats_matched/' + sample + '_' + suff + '.tsv', sep='\t', columns=cols)
                    print(sample, 'PSMs matched' , temp_df['feature_intensityApex'].notna().sum() )
            print('MATCHED')


## Создание новых PSMs_full_tool.tsv

                                    
#    suffixes = ['dino', 'bio', 'bio2', 'openMS'] - уже заданы в начале

    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        f = PSM_path.split('/')[-1]
        l = len(f)

        PSM = pd.read_csv(PSM_path, sep = '\t')
        for suf in suffixes :
            feats = out_directory + '/feats_matched/' + sample + '_' + suf + '.tsv'
            feat_df = pd.read_csv(feats , sep='\t')
            PSM['MS1Intensity'] = feat_df['feature_intensityApex'].fillna(0.0)

            new_f = f.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv' )
            old_path = PSM_path[:-l]
            PSM.to_csv(os.path.join(old_path, new_f), sep = '\t', encoding='utf-8')
                                    
                                    
## Потоковый анализ Diffacto


    # переименование начальных файлов из PSMs_full.tsv в PSMs_full_base.tsv чтобы не менять scav2diffacto.py

    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        f = PSM_path.split('/')[-1]
        l = len(f)
        old_path = PSM_path[:-l]
        new_f = f.replace('PSMs_full.tsv', 'PSMs_full_base.tsv')
        os.rename(PSM_path, os.path.join(old_path, new_f))



    out_path = out_directory + '/diffacto/'
    subprocess.call(['mkdir', '-p', out_path])

    full_suf = suffixes
    mixed_suf = ['mixed']

    # Построение файлов _proteins.tsv на подачу

    #S1 и S2 - это файлы _proteins.tsv !!!!!!!

    s1 = [x.replace('_PSMs_full.tsv', '_proteins.tsv') for x in sample_1]
    s2 = [x.replace('_PSMs_full.tsv', '_proteins.tsv') for x in sample_2]

    # Циклично для каждого суффикса (dino, bio2...) переименовывает нужный файл в PSMs_full.tsv, загоняет в Diffacto, 
    # затем переименовывает файл обратно в PSMs_full_suf.tsv
    # Для смешанных "лучших" интенсивностей суффикс mixed

    for suf in full_suf :
        diff_out_name = out_path + args['outDiff'].replace('.txt', '_' + suf + '.txt')
        diff_sample_name = out_path + args['outSampl'].replace('.txt', '_' + suf + '.txt')
        diff_peptides_name = out_path + args['outPept'].replace('.txt', '_' + suf + '.txt')

        for PSM_path, sample in zip(PSMs_full_paths, samples) :
            file_name = PSM_path.split('/')[-1]
            file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv')
            l = len(file_name)
            path_to_name = PSM_path[:-l]
            os.rename( path_to_name + file_name_with_tool,  path_to_name + file_name )
        
        subprocess.call([args['scav2dif'], '-dif', args['dif'], '-S1', *s1, '-S2', *s2,
                        '-out', diff_out_name, '-samples', diff_sample_name, '-peptides', diff_peptides_name,
                        '-norm', args['norm'], '-impute_threshold', args['impute_threshold'], '-min_samples', args['min_samples']])
        
        for PSM_path, sample in zip(PSMs_full_paths, samples) :
            file_name = PSM_path.split('/')[-1]
            file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv')
            l = len(file_name)
            path_to_name = PSM_path[:-l]
            os.rename( path_to_name + file_name, path_to_name + file_name_with_tool)
    
    # Переименование PSMs_full_base.tsv обратно в PSMs_full.tsv
    
    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        file_name = PSM_path.split('/')[-1]
        file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + 'base' + '.tsv')
        l = len(file_name)
        path_to_name = PSM_path[:-l]
        os.rename( path_to_name + file_name_with_tool, path_to_name + file_name)
            
                                    
### Анализ 
                                    
                                    
    full_suf = suffixes

    diff_out = {}
    for suf in full_suf :
        diff_out[suf] = pd.read_csv(out_directory + '/diffacto/' + 'diffacto_out_' + suf + '.txt', sep = '\t')

    for suf in full_suf :
        diff_out[suf]['log2_FC'] = np.log2(diff_out[suf]['S1']/diff_out[suf]['S2'])
        diff_out[suf]['FC'] = diff_out[suf]['S1']/diff_out[suf]['S2']
        diff_out[suf]['Human'] = diff_out[suf]['Protein'].apply(lambda x: True if x.find('HUMAN') > 0 else False)

    d = {}
    for suf in full_suf :
        Bonferroni = 0.05/len(diff_out[suf])
        d[suf] = diff_out[suf].query('(`log2_FC` > 0.5 or `log2_FC` < -0.5) and `P(PECA)` < @Bonferroni')


    comp_df = d[full_suf[0] ][ d[full_suf[0] ]['S/N'] > 0.01 ].loc[:, ['Protein', 'log2_FC']]
    for suf in full_suf[1:] :
        comp_df = comp_df.merge(d[suf][d[suf]['S/N'] > 0.01 ].loc[:, ['Protein', 'log2_FC']],
                                on='Protein', how='outer', suffixes = ('', '_'+suf) )
    comp_df.rename(columns={'log2_FC': 'log2_FC_' + full_suf[0]}, inplace=True)
    comp_df.dropna(how = 'all', subset=['log2_FC_' + suf for suf in full_suf], inplace=True)
    comp_df.to_csv(out_directory + '/proteins_compare.tsv', sep='\t', index = list(comp_df.index), columns = list(comp_df.columns), encoding = 'utf-8')  
    
    num_changed_prots = {}
    for suf in full_suf :
        col = 'log2_FC_'+suf
        num_changed_prots[suf] = len(comp_df[col].dropna())

    default_tool = max(num_changed_prots, key = lambda x: num_changed_prots[x])
           
### Первая диаграмма Вена
                                    
    if args['venn'] :
        dataset_dict = {}
        for suf in full_suf :
            col = 'log2_FC_'+suf
            dataset_dict[suf] = set(comp_df[comp_df[col].notna()]['Protein'])

        venn(dataset_dict, cmap=plt.get_cmap('Dark2'), fontsize=28, legend_loc="upper left", alpha = 0.5, figsize = (16, 16))
        plt.savefig(out_directory + '/venn_no_mix.png')
                                    
                                    
    # samples = ['10_15ng', '11_15ng', '12_15ng', '4_7-5ng', '5_7-5ng', '6_7-5ng']
    # suffixes = ['dino', 'bio', 'bio2', 'openMS']
    short_suffixes = {'dino': '_D', 'bio': '_B', 'bio2': '_B2', 'openMS': '_O'}

    for sample in samples :
        aggr_df = pd.read_csv( out_directory + '/feats_matched/' + sample + '_dino.tsv', sep='\t')
        aggr_df = aggr_df[['peptide', 'protein']]
        aggr_df.drop_duplicates(keep='first', inplace = True)
        for suf in suffixes :   
            feats = out_directory + '/feats_matched/' + sample + '_' + suf + '.tsv'
            temp_df = pd.read_csv(feats , sep='\t')
            temp_df = temp_df[['peptide', 'protein', 'feature_intensityApex']]
            temp_df['feature_intensityApex'] = temp_df['feature_intensityApex'].fillna(0.0)
            temp_df.sort_values('feature_intensityApex', ascending=False, inplace=True)
            temp_df.drop_duplicates(subset='peptide', keep='first', inplace=True)
            temp_df.rename(columns={'feature_intensityApex': sample + short_suffixes[suf]}, inplace = True)
            aggr_df = aggr_df.merge(temp_df,  how = 'outer', on = ['peptide', 'protein'], suffixes = (None, '^'))
            print('peptide aggregating', sample + ' ' + suf + ' done' )
    #    aggr_df.drop_duplicates(keep='first', inplace = True)
        aggr_df.to_csv( out_directory + '/diffacto/aggr_intens' + sample + '.tsv', sep='\t', 
                       index = list(aggr_df.index), columns = list(aggr_df.columns), encoding = 'utf-8')
        aggr_df = False

    merge_df = pd.read_csv(out_directory + '/diffacto/aggr_intens' + samples[0] + '.tsv', sep='\t')[['peptide', 'protein']]
    for sample in samples : 
        temp_df = pd.read_csv(out_directory + '/diffacto/aggr_intens' + sample + '.tsv', sep='\t')
        merge_df = merge_df.merge(temp_df[temp_df.columns[1:]], how = 'outer', on = ['peptide', 'protein'], suffixes = (None, '^'))

    cols = [sample + short_suffixes[suf] for suf in suffixes for sample in samples]

    merge_df[cols] = merge_df[cols].replace({'0':np.nan, 0.0:np.nan})
    merge_df.to_csv(out_directory + '/diffacto/aggr_intens_all.tsv', sep='\t', index = list(merge_df.index), columns = list(merge_df.columns), encoding = 'utf-8')  
    

### Выбор интенсивностей

                                
    for suf in suffixes :
        suf_cols = [sample + short_suffixes[suf] for sample in samples]
        col = 'num_NaN' + short_suffixes[suf]
        merge_df[col] = merge_df[suf_cols].isna().sum(axis = 1)            

    very_short_suf_dict = {'dino': 'D', 'bio': 'B', 'bio2': 'B2', 'openMS': 'O'}
    very_short_suf = [very_short_suf_dict[suf] for suf in suffixes]
    suffix_list = [ very_short_suf_dict[default_tool] ]
    suffix_list.extend(very_short_suf)
    suffix_list = list(dict.fromkeys(suffix_list))
    num_na_cols = ['num_NaN_' + suf for suf in suffix_list]
    #print(num_na_cols)

    merge_df['tool'] = merge_df[num_na_cols].idxmin(axis=1)
    merge_df['tool'] = merge_df['tool'].apply(lambda x: x[-1])


## Второй прогон диффакто


    if k <= 2 and args['mixed'] :
        
        a = out_directory + '/diffacto/mixed_intensity/' 
        subprocess.call(['mkdir', '-p', a])
                                    
        # Генерация таблиц смешанных интенсивностей на каждый образец

        for sample in samples :
            mixed_intens_df = merge_df[['peptide', 'protein']]

            cols = [x for x in merge_df.columns if x.startswith(sample)]
            cols.append('tool')
            tab = merge_df[cols].to_numpy()
            l = len(tab)
            Intensity = []

            for i in range(l) :
                if tab[i][-1] == 'D' :
                    Intensity.append(tab[i][0])
                elif tab[i][-1] == 'B' :
                    Intensity.append(tab[i][1])
                elif tab[i][-1] == '2' :
                    Intensity.append(tab[i][2])
                elif tab[i][-1] == 'O' :
                    Intensity.append(tab[i][3])
            mixed_intens_df['Intensity'] = pd.Series(data = Intensity, index = mixed_intens_df.index).fillna(0.0)
            mixed_intens_df.to_csv(out_directory + '/diffacto/mixed_intensity/' + sample + '.tsv', sep='\t', 
                                   index = list(mixed_intens_df.index), columns = list(mixed_intens_df.columns), encoding = 'utf-8')
                                    
# Создает новые PSMs_full_mixed.tsv
# Читает PSMs_full_base.tsv и mixed_intensity, заменяет колонку MS1Intensity на интенсивности из mixed

        suf = 'mixed'
        for PSM_path, sample in zip(PSMs_full_paths, samples) :
            f = PSM_path.split('/')[-1]
            l = len(f)
            old_path = PSM_path[:-l]
            psm_df = pd.read_csv( os.path.join(old_path, f), sep='\t' )

            intens = out_directory + '/diffacto/mixed_intensity/' + sample + '.tsv'
            intens_df = pd.read_csv(intens , sep='\t')[['peptide', 'protein', 'Intensity']]
            #print(intens_df[:1])
            psm_df = psm_df.merge(intens_df,  how = 'inner', on = ['peptide', 'protein'], suffixes = (None, '^'))
            #print(list(psm_df.columns))
            psm_df.rename(columns={'MS1Intensity' : 'Old_Intensity', 'Intensity' : 'MS1Intensity'}, inplace=True)

            new_f = f.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv' )
            psm_df.to_csv(old_path + new_f, sep = '\t', encoding='utf-8')

# Собственно прогон Diffacto на новых PSM файлах
# кусок скопипастен сверху, просто заменил суффикс

        out_path = out_directory + '/diffacto/'
        mixed_suf = ['mixed']

        # Построение файлов c белками!!! _proteins.tsv

        s1 = [x.replace('_PSMs_full.tsv', '_proteins.tsv') for x in sample_1]
        s2 = [x.replace('_PSMs_full.tsv', '_proteins.tsv') for x in sample_2]

        # экранирование начальных PSMs_full в PSMs_full_base.tsv
        for PSM_path, sample in zip(PSMs_full_paths, samples) :
            file_name = PSM_path.split('/')[-1]
            file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + 'base' + '.tsv')
            l = len(file_name)
            path_to_name = PSM_path[:-l]
            os.rename( path_to_name + file_name, path_to_name + file_name_with_tool)
        
        # Циклично для каждого суффикса (dino, bio2...) переименовывает нужный файл в PSMs_full.tsv, загоняет в Diffacto, 
        # затем переименовывает файл обратно в PSMs_full_suf.tsv

        # Для только OpenMS заменить full_suf на new_suf
        # Для смешанных интенсивностей суффикс mixed
        
        for suf in mixed_suf :
            diff_out_name = out_path + args['outDiff'].replace('.txt', '_' + suf + '.txt')
            diff_sample_name = out_path + args['outSampl'].replace('.txt', '_' + suf + '.txt')
            diff_peptides_name = out_path + args['outPept'].replace('.txt', '_' + suf + '.txt')

            for PSM_path, sample in zip(PSMs_full_paths, samples) :
                file_name = PSM_path.split('/')[-1]
                file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv')
                l = len(file_name)
                path_to_name = PSM_path[:-l]
                os.rename( path_to_name + file_name_with_tool,  path_to_name + file_name )
            
            subprocess.call([args['scav2dif'], '-dif', args['dif'], '-S1', *s1, '-S2', *s2,
                        '-out', diff_out_name, '-samples', diff_sample_name, '-peptides', diff_peptides_name,
                        '-norm', args['norm'], '-impute_threshold', args['impute_threshold'], '-min_samples', args['min_samples']])

            for PSM_path, sample in zip(PSMs_full_paths, samples) :
                file_name = PSM_path.split('/')[-1]
                file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv')
                l = len(file_name)
                path_to_name = PSM_path[:-l]
                os.rename( path_to_name + file_name, path_to_name + file_name_with_tool)
            
        # Переименование PSMs_full_base.tsv обратно в PSMs_full.tsv
    
        for PSM_path, sample in zip(PSMs_full_paths, samples) :
            file_name = PSM_path.split('/')[-1]
            file_name_with_tool = file_name.replace('PSMs_full.tsv', 'PSMs_full_' + 'base' + '.tsv')
            l = len(file_name)
            path_to_name = PSM_path[:-l]
            os.rename( path_to_name + file_name_with_tool, path_to_name + file_name)
                                    

        full_suf = suffixes + ['mixed']

        diff_out = {}
        for suf in full_suf :
            diff_out[suf] = pd.read_csv(out_directory + '/diffacto/' + 'diffacto_out_' + suf + '.txt', sep = '\t')

        for suf in full_suf :
            diff_out[suf]['log2_FC'] = np.log2(diff_out[suf]['S1']/diff_out[suf]['S2'])
            diff_out[suf]['FC'] = diff_out[suf]['S1']/diff_out[suf]['S2']
            diff_out[suf]['Human'] = diff_out[suf]['Protein'].apply(lambda x: True if x.find('HUMAN') > 0 else False)

        d = {}
        for suf in full_suf :
            Bonferroni = 0.05/len(diff_out[suf])
            d[suf] = diff_out[suf].query('(`log2_FC` > 0.5 or `log2_FC` < -0.5) and `P(PECA)` < @Bonferroni')


        comp_df = d[full_suf[0] ][ d[full_suf[0] ]['S/N'] > 0.01 ].loc[:, ['Protein', 'log2_FC']]
        for suf in full_suf[1:] :
            comp_df = comp_df.merge(d[suf][d[suf]['S/N'] > 0.01 ].loc[:, ['Protein', 'log2_FC']],
                                    on='Protein', how='outer', suffixes = ('', '_'+suf) )
        comp_df.rename(columns={'log2_FC': 'log2_FC_' + full_suf[0]}, inplace=True)
        comp_df.dropna(how = 'all', subset=['log2_FC_' + suf for suf in full_suf], inplace=True)
        comp_df.to_csv(out_directory + '/proteins_compare_full.tsv', sep='\t', index = list(comp_df.index), columns = list(comp_df.columns), encoding = 'utf-8')  

        for suf in full_suf :
            col = 'log2_FC_'+suf
            num_changed_prots[suf] = len(comp_df[col].dropna())
                                    
        l = len(comp_df)

        print('For ALL proteins mean squared error:')

        for suf in full_suf :
            print(suf + ' log2 fold change mean squared error =', round((1/np.sqrt(l))*np.sqrt(((comp_df['log2_FC_'+suf] - 1)**2).sum()), 3))
        print('\n')

        t = comp_df.dropna(axis=0, how='any')
        l = len(t)

        print('For COMMON proteins mean squared error:')

        for suf in full_suf :
            print(suf + ' log2 fold change mean squared error =', round((1/np.sqrt(l))*np.sqrt(((t['log2_FC_'+suf] - 1)**2).sum()), 3))
        print('\n')
        

### Вторая диаграмма Вена

        
        if args['venn'] :
            dataset_dict = {}
            for suf in full_suf :
                col = 'log2_FC_'+suf
                dataset_dict[suf] = set(comp_df[comp_df[col].notna()]['Protein'])

            venn(dataset_dict, cmap=plt.get_cmap('Dark2'), fontsize=28, legend_loc="upper left", alpha = 0.5, figsize = (16, 16))
            plt.savefig(out_directory + '/venn_mix.png')

if __name__ == '__main__':
    run()
