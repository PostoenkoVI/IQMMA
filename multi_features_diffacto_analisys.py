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
    (prefered way)
    $ multi_features_diffacto_analisys.py -cfg /path_to/default.ini
    
    or (possible problems with file path)
    $ multi_features_diffacto_analisys.py @default.txt
    
    or 
    $ multi_features_diffacto_analisys.py -mzML sample1_1.mzML sample1_n.mzML sample2_1.mzML sample1_n.mzML 
                                          -s1 sample1_1_PSMs_full.tsv sample1_n_PSMs_full.tsv 
                                          -s2 sample2_1_PSMs_full.tsv sample2_n_PSMs_full.tsv
                                          -sampleNames sample1_1 sample1_n sample2_1 sample2_n
                                          -outdir ./script_out
                                          -dif path_to/diffacto
                                          -dino /home/bin/dinosaur
                                          -bio /home/bin/biosaur
                                          -bio2 /home/bin/biosaur2
                                          -openMS path_to_openMS
                                          -venn 1
                                          -mixed 1
                                          -overwrite_features 1
                                          -overwrite_matching 1
                                          
    -------------
    ''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    
    parser.add_argument('-cfg', help='path to config .ini file')
    parser.add_argument('-cfg_category', help='name of category to prioritize in the .ini file')
    parser.add_argument('-dif', help='path to Diffacto')
    parser.add_argument('-scav2dif', help='path to scav2diffacto')
    parser.add_argument('-s1', nargs='+', help='input files PSMs_full.tsv (and _proteins.tsv should be in the same directory) for S1 sample')
    parser.add_argument('-s2', nargs='+', help='input files PSMs_full.tsv (and _proteins.tsv should be in the same directory) for S2 sample')
    parser.add_argument('-sampleNames', nargs='+', help='short names for samples for inner structure of results')
    parser.add_argument('-mzML', nargs='+', 
                        help='paths to mzML files for samples in the same order in one line S1 the S2...')
    
    parser.add_argument('-dino', help='path to Dinosaur')
    parser.add_argument('-bio', help='path to Biosaur')
    parser.add_argument('-bio2', help='path to Biosaur2')
    parser.add_argument('-openms', help='path to OpenMS feature')
    
    parser.add_argument('-s3', nargs='+', help='input files for S3 sample')
    parser.add_argument('-s4', nargs='+', help='input files for S4 sample')
    
    parser.add_argument('-outdir', help='name of  directory, where results would be stored')
    parser.add_argument('-overwrite_features', help='whether to overwrite existed features files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-overwrite_new_PSMs', help='whether to overwrite existed new PSMs_full files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-overwrite_matching', help='whether to overwrite existed matched files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-overwrite_first_diffacto', help='whether to overwrite existed diffacto files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-mixed', help='whether to reanalyze mixed intensities (1) or not (0)')
    parser.add_argument('-venn', help='whether to plot venn diagrams (1) or not (0)')
    parser.add_argument('-choice', help='method how to choose final intensities for peptide. 0 - default order and min Nan values, 1 - min Nan and min of summ CV, 2 - min Nan and min of max CV, 3 - default order with filling Nan values between programs (if using this variant -norm MUST be applied), 4 - min Nan and min of root of squared summ CV')
    parser.add_argument('-norm', help='normalization method for intensities. Can be 1 - median or 0 - no normalization')
    
    parser.add_argument('-outPept', help='name of output diffacto peptides file (important: .txt)', default='peptides.txt')
    parser.add_argument('-outSampl', help='name of output diffacto samples file (important: .txt)', default='sample.txt')
    parser.add_argument('-outDiff', help='name of diffacto output file (important: .txt)', default='diffacto_out.txt')
    parser.add_argument('-normDiff', help='normalization method for Diffacto. Can be average, median, GMM or None', default='None')
    parser.add_argument('-impute_threshold', help='impute_threshold for missing values fraction', default='0.25')
    parser.add_argument('-min_samples', help='minimum number of samples for peptide usage', default='3')
#    parser.add_argument('-version', action='version', version='%s' % (pkg_resources.require("scavager")[0], ))
    args = vars(parser.parse_args())


    if args['cfg'] :
        if args['cfg_category'] :
            cat = args['cfg_category']
        else :
            cat = 'settings_one'
        config = configparser.RawConfigParser(allow_no_value=True, empty_lines_in_values=False, )
        config.read(os.path.join(os.path.abspath(__file__), args['cfg']))
        for key in args.keys() :
            if args[key] == None :
                try :
                    args[key] = config[cat][key]
                except :   
                    if args[key] == None :
                        try :
                            args[key] = config['DEFAULT'][key]
                        except :
                            pass
                        finally :
                            pass
                finally :
                    pass
#    print(args)
#    print(args['s1'].split())

    if not args['dif'] :
        print('path to diffacto file is required')
        return -1

    if not args['scav2dif'] :
        print('path to scav2diffacto.py file is required')
        return -1
    
    if not args['outdir'] :
        print('path to output directory is required')
        return -1
    
    arg_suff = {'dino': 'dino', 'bio' : 'bio', 'bio2' : 'bio2', 'openms' : 'openMS'}
    suffixes = []
    for suf in arg_suff :
        if args[suf] :
            suffixes.append(arg_suff[suf])
            
    k = len(suffixes)
    if k == 0 :
        print('At least one feature finder shoud be given!')
        return -1
    elif k == 1 :
        print('One diffacto run')
        args['venn'] = 0
    elif k >= 2 :
        print('Second diffacto run is applied')
    
    PSMs_full_paths = []
    for sample_num in ['s1', 's2']:
        if args[sample_num] :
            for z in args[sample_num].split():
                PSMs_full_paths.append(z)
        else :
            print('sample '+ sample_num + ' *_PSMs_full.tsv files are required')
            return -1

    if args['mzML'] and (len(args['mzML'].split()) == len(PSMs_full_paths)) :
        mzML_paths = args['mzML'].split()
    else :
        print('paths to all .mzML files are required')
        return -1
    
    if args['sampleNames'] and (len(args['sampleNames'].split()) == len(PSMs_full_paths)) :
        samples = args['sampleNames'].split()
    else :
        print('short name for every PSMs file is required')
        return -1
    
    out_directory = args['outdir']
    sample_1 = args['s1'].split()
    sample_2 = args['s2'].split()
    
    args['overwrite_features'] = int( args['overwrite_features'])
    args['overwrite_matching'] = int( args['overwrite_matching'])
    args['overwrite_first_diffacto'] = int( args['overwrite_first_diffacto'])
    args['overwrite_new_PSMs'] = int( args['overwrite_new_PSMs'])
    args['mixed'] = int( args['mixed'])
    args['venn'] = int( args['venn'])
    args['choice'] = int( args['choice'])
    args['norm'] = int( args['norm'])
    
#    print(PSMs_full_paths)
    print(mzML_paths)
    print(out_directory)
    print(suffixes)
    print(sample_1)
    print(sample_2)
    print('mixed = ', args['mixed'])
    print('venn = ', args['venn'])
    print('choice = ', args['choice'])
    print('overwrite_features = ', args['overwrite_features'])
    print('overwrite_first_diffacto = ', args['overwrite_first_diffacto'])
    print('overwrite_new_PSMs = ', args['overwrite_new_PSMs'])
    print('overwrite_matching = ', args['overwrite_matching'], '\n\n')
    
#    subprocess.call(['python3', args['dif'], '-i', args['peptides'], '-samples', args['samples'], '-out',\
#     args['out'], '-normalize', args['norm'], '-impute_threshold', args['impute_threshold'], '-min_samples', args['min_samples']])

    
    subprocess.call(['mkdir', '-p', out_directory])

## Генерация фич

### Dinosaur

    outDir = out_directory + '/features'
    subprocess.call(['mkdir', '-p', outDir])

# На выходе добавляет в папку out_directory/features файлы *sample*_features_dino.tsv

    if args['dino'] :
        print('\n')
        for path, sample in zip(mzML_paths, samples) :
            outName = sample + '_features_' + 'dino' + '.tsv'
            outName_false = outName + '.features.tsv'
            if args['overwrite_features'] == 1 or not os.path.exists(outDir + '/' + outName) :
                with open(os.path.join(outDir, outName_false), mode='w+', buffering= -1, encoding='utf-8') :
                    print('Writing features' + ' dino ' + sample)
                    subprocess.call([args['dino'], '--outDir='+ outDir, '--outName='+ outName, path])
                os.rename(outDir + '/' + outName + '.features.tsv', outDir + '/' + outName)
            else :
                print('Not ovetwriting features ' + ' dino ' + sample)
        print('\n')

### Biosaur

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio'] :
        print('\n')
        for path, sample in zip(mzML_paths, samples) :
            outPath = out_directory + '/features/' + sample + '_features_bio.tsv'
            if args['overwrite_features'] == 1 or not os.path.exists(outPath) :
                print('Writing features ' + ' bio ' + sample)
                subprocess.call([args['bio'], path, '-out', outPath, '-hvf', '1000'])
            else :
                print('Not ovetwriting features ' + ' bio ' + sample)
        print('\n')

### Biosaur2

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio2.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio2'] :
        print('\n')
        for path, sample in zip(mzML_paths, samples) :
            outPath = out_directory + '/features/' + sample + '_features_bio2.tsv'
            if args['overwrite_features'] == 1 or not os.path.exists(outPath) :
                print('Writing features ' + ' bio2 ' + sample)
                subprocess.call([args['bio2'], path, '-o', outPath, '-hvf', '1000', '-minlh', '3'])
            else :
                print('Not ovetwriting features ' + ' bio2 ' + sample)
        print('\n')
            
### OpenMS

# На выходе создает в out_directory папку OpenMS с файлами *.featureXML и добавляет в папку out_directory/features файлы *sample*_features_openMS.tsv
    
    if args['openms'] :
        print('\n')
        out_path = out_directory + '/openMS/'
        subprocess.call(['mkdir', '-p', out_path])
        
        
        for path, sample in zip(mzML_paths, samples) :
            out_path = out_directory + '/openMS/' + sample + '.featureXML'
            if args['overwrite_features'] == 1 or not os.path.exists(out_path) :
                print('Writing .featureXML ' + ' openMS ' + sample)
                subprocess.call([args['openms'], '-in', path, '-out', out_path, '-algorithm:isotopic_pattern:charge_low', '2', '-algorithm:isotopic_pattern:charge_high', '7'])
            else :
                print('Not ovetwriting .featureXML ' + ' openMS ' + sample)
        print('\n')

        for path, sample in zip(mzML_paths, samples) :
            out_path = out_directory + '/openMS/' + sample + '.featureXML'
            o = os.path.join(out_directory + '/features', sample + '_features_' + 'openMS.tsv')
            if args['overwrite_features'] == 1 or not os.path.exists(o) : 
                print('Writing features ' + ' openMS ' + sample)
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
                b.to_csv(o, sep='\t', encoding='utf-8')
            else :
                print('Not overwriting features ' + ' openMS ' + sample)
        print('\n')


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
            l_mz = mz - dm
            r_mz = mz + dm
                                    
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
                feature_index, feature_mz, feature_charge, feature_rtStart, feature_rtEnd, feature_intensityApex, num_features_matched = find_feature_for_psm_1(features_db, psm_mz, psm_charge, psm_rt, accuracy_ppm=accuracy_ppm, accuracy_minut = accuracy_minut, openMS = False)

                features_for_psm.append([peptide, protein, psm_index, feature_index, psm_mz, feature_mz, psm_charge, 
                                     feature_charge, feature_rtStart, psm_rt, feature_rtEnd, feature_intensityApex, num_features_matched])
            else :
                feature_index, feature_mz, feature_charge, feature_rtApex, feature_intensityApex, num_features_matched = find_feature_for_psm_1(features_db, psm_mz, 
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
    print('Start matching features')
    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        PSM = pd.read_csv(PSM_path, sep = '\t')
        print('sample', sample)
        for suf in suffixes :
            if args['overwrite_matching'] == 1 or not os.path.exists(out_directory + '/feats_matched/' + sample + '_' + suf + '.tsv') :
                feats = pd.read_csv( out_directory + '/features/' + sample + '_features_' + suf + '.tsv', sep = '\t')    
                print(suf, 'features', sample, '\n', 'START')
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
                
                median = temp_df['feature_intensityApex'].median()
                temp_df['med_norm_feature_intensityApex'] = temp_df['feature_intensityApex']/median
                cols.append('med_norm_feature_intensityApex')
                
                print(suf, 'features', sample, 'DONE')
                temp_df.to_csv(out_directory + '/feats_matched/' + sample + '_' + suf + '.tsv', sep='\t', columns=cols)
                print(sample, 'PSMs matched' , temp_df['feature_intensityApex'].notna().sum() )
                print(suf + ' MATCHED')


## Создание новых PSMs_full_tool.tsv

                                    
#    suffixes = ['dino', 'bio', 'bio2', 'openMS'] - уже заданы в начале
    
    print('Creating new PSMs_full')
    feature_colomn_name = 'feature_intensityApex'
    if args['norm'] == 0 :
        feature_colomn_name = 'feature_intensityApex'
    elif args['norm'] == 1 :
        feature_colomn_name = 'med_norm_feature_intensityApex'
    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        f = PSM_path.split('/')[-1]
        l = len(f)
        old_path = PSM_path[:-l]
        for suf in suffixes :
            PSM = pd.read_csv(PSM_path, sep = '\t')
            feats = out_directory + '/feats_matched/' + sample + '_' + suf + '.tsv'
            new_f = f.replace('PSMs_full.tsv', 'PSMs_full_' + suf + '.tsv')
            if args['overwrite_new_PSMs'] == 1 or not os.path.exists(os.path.join(old_path, new_f)) : 
                feat_df = pd.read_csv(feats , sep='\t')[['peptide', 'protein', feature_colomn_name]]
                PSM = PSM.merge(feat_df,  how = 'inner', on = ['peptide', 'protein'], suffixes = (None, '^'))
                PSM.rename(columns={'MS1Intensity' : 'Old_Intensity', feature_colomn_name : 'MS1Intensity'}, inplace=True)
                PSM['MS1Intensity'] = PSM['MS1Intensity'].fillna(0.0)
                PSM.to_csv(os.path.join(old_path, new_f), sep = '\t', encoding='utf-8')
    print('New PSMs_full created')


## Первый прогон Diffacto


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
        
        print('Diffacto START')
        
        if args['overwrite_first_diffacto'] == 1 or k == 1 or not os.path.exists(diff_out_name) :
            subprocess.call([args['scav2dif'], '-dif', args['dif'], '-S1', *s1, '-S2', *s2,
                            '-out', diff_out_name, '-samples', diff_sample_name, '-peptides', diff_peptides_name,
                            '-norm', args['normDiff'], '-impute_threshold', args['impute_threshold'], '-min_samples', args['min_samples']])

        print('Diffacto END')

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
    d = False
    num_changed_prots = {}
    for suf in full_suf :
        col = 'log2_FC_'+suf
        num_changed_prots[suf] = len(comp_df[col].dropna())

    default_order = sorted(suffixes, key= lambda x: num_changed_prots[x], reverse=True)
    
           
### Первая диаграмма Вена
    
    
    if args['venn'] == 1 :
        dataset_dict = {}
        for suf in full_suf :
            col = 'log2_FC_'+suf
            dataset_dict[suf] = set(comp_df[comp_df[col].notna()]['Protein'])

        venn(dataset_dict, cmap=plt.get_cmap('Dark2'), fontsize=28, legend_loc="upper left", alpha = 0.5, figsize = (16, 16))
        plt.savefig(out_directory + '/venn_no_mix.png')
        print('First Venn created')
                                    
    # samples = ['10_15ng', '11_15ng', '12_15ng', '4_7-5ng', '5_7-5ng', '6_7-5ng']
    # suffixes = ['dino', 'bio', 'bio2', 'openMS']
    short_suffixes = {'dino': '_D', 'bio': '_B', 'bio2': '_B2', 'openMS': '_O'}
    

### Peptide aggregating
    
    
    if args['norm'] == 0 :
        feature_colomn_name = 'feature_intensityApex'
    elif args['norm'] == 1 :
        feature_colomn_name = 'med_norm_feature_intensityApex'
        
    for sample in samples :
        aggr_df = pd.read_csv( out_directory + '/feats_matched/' + sample + '_dino.tsv', sep='\t', usecols=['peptide', 'protein'])
        aggr_df.drop_duplicates(keep='first', inplace = True)
        for suf in suffixes :   
            feats = out_directory + '/feats_matched/' + sample + '_' + suf + '.tsv'
            temp_df = pd.read_csv(feats , sep='\t')
            temp_df = temp_df[['peptide', 'protein', feature_colomn_name]]
            temp_df[feature_colomn_name] = temp_df[feature_colomn_name].fillna(0.0)
            temp_df.sort_values(feature_colomn_name, ascending=False, inplace=True)
            temp_df.drop_duplicates(subset='peptide', keep='first', inplace=True)
            temp_df.rename(columns={feature_colomn_name: sample + short_suffixes[suf]}, inplace = True)
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
    
    
### Выбор интенсивностей


    print('Choosing intensities STARTED')
    for suf in suffixes :
        suf_cols = [sample + short_suffixes[suf] for sample in samples]
        col = 'num_NaN' + short_suffixes[suf]
        merge_df[col] = merge_df[suf_cols].isna().sum(axis = 1)
    
    for suf in suffixes :
        l = len(samples)//2
        s1_cols = [x for x in merge_df.columns if x.startswith(tuple(samples[:l])) and x.endswith(short_suffixes[suf])]
        s2_cols = [x for x in merge_df.columns if x.startswith(tuple(samples[l:])) and x.endswith(short_suffixes[suf])]
        merge_df[ 's1_'+'mean'+short_suffixes[suf] ] = merge_df.loc[:, s1_cols].mean(axis=1)
        merge_df[ 's2_'+'mean'+short_suffixes[suf] ] = merge_df.loc[:, s2_cols].mean(axis=1)
        merge_df[ 's1_'+'std'+short_suffixes[suf] ] = merge_df.loc[:, s1_cols].std(axis=1)
        merge_df[ 's2_'+'std'+short_suffixes[suf] ] = merge_df.loc[:, s2_cols].std(axis=1)
        merge_df[ 's1_'+'cv'+short_suffixes[suf] ] = merge_df['s1_std'+short_suffixes[suf]] / merge_df['s1_mean'+short_suffixes[suf]]
        merge_df[ 's2_'+'cv'+short_suffixes[suf] ] = merge_df['s2_std'+short_suffixes[suf]] / merge_df['s2_mean'+short_suffixes[suf]]
        merge_df[ 'summ_cv'+short_suffixes[suf] ] = merge_df['s1_cv'+short_suffixes[suf]] + merge_df['s2_cv'+short_suffixes[suf]]
        merge_df[ 'sq_summ_cv'+short_suffixes[suf] ] = np.sqrt(merge_df['s1_cv'+short_suffixes[suf]]**2 + merge_df['s2_cv'+short_suffixes[suf]]**2)
        merge_df[ 'max_cv'+short_suffixes[suf] ] = merge_df.loc[:, ['s1_cv'+short_suffixes[suf], 's2_cv'+short_suffixes[suf]] ].max(axis=1)
        
        merge_df.drop(columns=[ 's1_mean'+short_suffixes[suf], 's2_mean'+short_suffixes[suf], 
                                's1_std'+short_suffixes[suf], 's2_std'+short_suffixes[suf] ], inplace=True)

    very_short_suf_dict = {'dino': 'D', 'bio': 'B', 'bio2': 'B2', 'openMS': 'O'}
    very_short_suf = [very_short_suf_dict[suf] for suf in suffixes]
    
    # min number of Nan values + default order (in order with maximum differentially expressed proteins in single diffacto runs)
    if args['choice'] == 0 :
        suffix_list = [ very_short_suf_dict[suf] for suf in default_order ]
        num_na_cols = ['num_NaN_' + suf for suf in suffix_list]
        #print(num_na_cols)

        merge_df['tool'] = merge_df[num_na_cols].idxmin(axis=1)
        merge_df['tool'] = merge_df['tool'].apply(lambda x: x.split('_')[-1])
    
    # min number of Nan values and min summ or max CV
    elif args['choice'] == 1 or args['choice'] == 2 or args['choice'] == 4 :
        num_na_cols = ['num_NaN' + short_suffixes[suf] for suf in suffixes]
        merge_df['NaN_border'] = merge_df[num_na_cols].min(axis=1)
        if args['choice'] == 1 :
            cv = 'summ_cv'
        if args['choice'] == 2 :
            cv = 'max_cv'
        if args['choice'] == 4 :
            cv = 'sq_summ_cv'
        for suf in suffixes :
            merge_df['masked_CV' + short_suffixes[suf] ] = merge_df[ cv + short_suffixes[suf] ].mask(merge_df[ 'num_NaN' + short_suffixes[suf] ] > merge_df['NaN_border'])
        masked_CV_cols = ['masked_CV' + short_suffixes[suf] for suf in suffixes]
        merge_df['tool'] = merge_df[masked_CV_cols].idxmin(axis=1)
        merge_df['tool'].mask(merge_df['tool'].isna(), other=merge_df[num_na_cols].idxmin(axis=1), inplace=True)
        merge_df['tool'] = merge_df['tool'].apply(lambda x: x.split('_')[-1])
    
    # min number of Nan values + default order to fill NaNs (in order with maximum differentially expressed proteins in single diffacto runs)
    elif args['choice'] == 3 :
        suffix_list = [ very_short_suf_dict[suf] for suf in default_order ]
        suffix_tuple = tuple(suffix_list)
        num_na_cols = ['num_NaN_' + suf for suf in suffix_list]
        #print(num_na_cols)

        merge_df['tool'] = pd.Series(data = [suffix_tuple for i in range(len(merge_df)) ], index = merge_df.index)
        
#        masked_CV_cols.append('NaN_border')
#        merge_df.drop(columns=masked_CV_cols, inplace=True)
        
    merge_df.to_csv(out_directory + '/diffacto/aggr_intens_all.tsv', sep='\t', index = list(merge_df.index), columns = list(merge_df.columns), encoding = 'utf-8')  
    print('Choosing intensities DONE')


## Второй прогон диффакто


    if k >= 2 and args['mixed'] == 1 :
        a = out_directory + '/diffacto/mixed_intensity/' 
        subprocess.call(['mkdir', '-p', a])
                                    
        # Генерация таблиц смешанных интенсивностей на каждый образец
        
        temp_dict = {}
        for i, suf in zip(range(len(very_short_suf)), very_short_suf) :
            temp_dict[suf] = i
        
        mixed_intens_df = merge_df[['peptide', 'protein']].copy()
        for sample in samples :

            cols = [x for x in merge_df.columns if x.startswith(sample)]
            cols.append('tool')
            tab = merge_df[cols].to_numpy()
            l = len(tab)
            Intensity = []

            if args['choice'] == 3 :
                for i in range(l) :
                    for suf in tab[i][-1] :
                        if not np.isnan(tab[i][ temp_dict[suf] ]) :
                            intens = tab[i][ temp_dict[suf] ]
                            break
                        else :
                            intens = 0.0
                    Intensity.append(intens)
            else :
                for i in range(l) :
                    Intensity.append(tab[i][ temp_dict[ tab[i][-1] ] ])


            mixed_intens_df['Intensity'] = pd.Series(data = Intensity, index = mixed_intens_df.index).fillna(0.0)
            mixed_intens_df.to_csv(out_directory + '/diffacto/mixed_intensity/' + sample + '.tsv', sep='\t', 
                                   index = list(mixed_intens_df.index), columns = list(mixed_intens_df.columns), encoding = 'utf-8')
        print('Creating new mixed_intensity files DONE')
                                    
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
        print('Creating new PSMs_full DONE')

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
                        '-norm', args['normDiff'], '-impute_threshold', args['impute_threshold'], '-min_samples', args['min_samples']])

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
        print('Second Diffacto run DONE')
        
        
        # Analysis
        
        
        full_suf = suffixes + ['mixed']

        diff_out = {}
        for suf in full_suf :
            print(out_directory + '/diffacto/' + 'diffacto_out_' + suf + '.txt')
            diff_out[suf] = pd.read_csv(out_directory + '/diffacto/' + 'diffacto_out_' + suf + '.txt', sep = '\t')

        for suf in full_suf :
            diff_out[suf]['log2_FC'] = np.log2(diff_out[suf]['S1']/diff_out[suf]['S2'])
            diff_out[suf]['FC'] = diff_out[suf]['S1']/diff_out[suf]['S2']

        d = {}
        for suf in full_suf :
            Bonferroni = 0.05/len(diff_out[suf])
            d[suf] = diff_out[suf].query('(`log2_FC` > 0.5 or `log2_FC` < -0.5) and `P(PECA)` < @Bonferroni')


        comp_df = d[full_suf[0] ][ d[full_suf[0]]['S/N'] > 0.01 ].loc[:, ['Protein', 'log2_FC']]
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
        
        print('Comparing proteins DONE')
        
        '''
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
        '''      

### Вторая диаграмма Вена

        
        if args['venn'] == 1 :
            dataset_dict = {}
            for suf in full_suf :
                col = 'log2_FC_'+suf
                dataset_dict[suf] = set(comp_df[comp_df[col].notna()]['Protein'])

            venn(dataset_dict, cmap=plt.get_cmap('Dark2'), fontsize=28, legend_loc="upper left", alpha = 0.5, figsize = (16, 16))
            plt.savefig(out_directory + '/venn_mix.png')
            print('Second Venn DONE')

if __name__ == '__main__':
    run()
