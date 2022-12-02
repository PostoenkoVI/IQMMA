import argparse
import subprocess
import sys
import configparser
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pyteomics
import copy
import os
import errno
import time
import ast
import re
from os import listdir
from matplotlib.ticker import PercentFormatter
from pyteomics.openms import featurexml
from pyteomics import pepxml, mzid
import venn
from venn import venn
from scipy.stats import pearsonr, scoreatpercentile, percentileofscore
from scipy.optimize import curve_fit
import logging
import pickle


class WrongInputError(NotImplementedError):
    pass


class EmptyFileError(ValueError):
    pass

def call_Dinosaur(path_to_fd, mzml_path, outdir, outname, str_of_other_args ) :
    other_args = ['--' + x.strip().replace(' ', '=') for x in str_of_other_args.split('--')]
    final_args = [path_to_fd, mzml_path, '--outDir='+outdir, '--outName='+outname, ] + other_args
    final_args = list(filter(lambda x: False if x=='--' else True, final_args))
    process = subprocess.Popen(final_args, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout)
    exitscore = process.wait()
    os.rename(os.path.join(outdir, outname + '.features.tsv'),  os.path.join(outdir, outname) )
    return exitscore

def call_Biosaur2(path_to_fd, mzml_path, outpath, str_of_other_args) :
    other_args = [x.strip() for x in str_of_other_args.split(' ')]
    final_args = [path_to_fd, mzml_path, '-o', outpath, ] + other_args
    final_args = list(filter(lambda x: False if x=='' else True, final_args))
    process = subprocess.Popen(final_args, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout)
    exitscore = process.wait()
    return exitscore

def call_OpenMS(path_to_fd, mzml_path, outpath, str_of_other_args) :
    other_args = [x.strip() for x in str_of_other_args.split(' ')]
    final_args = [path_to_fd, 
                  '-in', mzml_path, 
                  '-out', outpath, 
                  ] + other_args
    final_args = list(filter(lambda x: False if x=='' else True, final_args))
    process = subprocess.Popen(final_args, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout)
    exitscore = process.wait()
    return exitscore

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def generate_users_output(diffacto_out={},
                          out_folder='',
                          plot_venn=True,
                          pval_treshold=0.05,
                          fc_treshold=2,
                          dynamic_fc_treshold=True,
                          save = True,
                          ) :
    # diffacto_out = {
    #     'suf1': path_to_diffacto_results,
    #     'suf2': path_to_diffacto_results,
    # }
    suffixes = [suf for suf in diffacto_out]
    
    flag = True
    comp_df = False
    for suf in suffixes :
        table = pd.read_csv(diffacto_out[suf], sep = '\t')
        table['log2_FC'] = np.log2(table['s1']/table['s2'])
        table['FC'] = table['s1']/table['s2']
        table.sort_values('P(PECA)', ascending=True, inplace=True)
        if flag :
            comp_df = table[['Protein']]
            flag = False
        bonferroni = pval_treshold/len(table)
        table = table[ table['S/N'] > 0.01 ]
        table = table[table['P(PECA)'] < bonferroni][['Protein', 'log2_FC']]
        
        if not dynamic_fc_treshold :
            logging.info('Static fold change treshold is applied')
            border_fc = abs(np.log2(fc_treshold))
            table = table[abs(table['log2_FC']) > border_fc]
        else :
            logging.info('Dynamic fold change treshold is applied')
            shift, sigma = table['log2_FC'].median(), table['log2_FC'].std()
            right_fc_treshold = shift + 3*sigma
            left_fc_treshold = shift - 3*sigma
            table = table.query('`log2_FC` >= @right_fc_treshold or `log2_FC` <= @right_fc_treshold')
        comp_df = comp_df.merge(table, how='outer', on='Protein', suffixes = (None, '_'+suf))
    comp_df.rename(columns={'log2_FC': 'log2_FC_'+suffixes[0] }, inplace=True )
    comp_df.dropna(how = 'all', subset=['log2_FC_' + suf for suf in suffixes], inplace=True)
    # comp_df.loc['Total number', :] = df0.notna().sum(axis=0)
    
    total_de_prots = {}
    for suf in suffixes :
        total_de_prots[suf] = comp_df['log2_FC_'+suf].notna().sum()
    
    if save :
        comp_df.to_csv(os.path.join(out_folder, 'proteins_compare.tsv'), 
                       sep='\t',
                       index=False,
                       columns = list(comp_df.columns), 
                       encoding = 'utf-8')
    
    if plot_venn :
        dataset_dict = {}
        for suf in suffixes :
            col = 'log2_FC_'+suf
            dataset_dict[suf] = set(comp_df[comp_df[col].notna()]['Protein'])
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        venn(dataset_dict, cmap=plt.get_cmap('Dark2'), fontsize=28, alpha = 0.5, ax=ax)
        cmap=plt.get_cmap('Dark2')
        colors = {}
        c = [cmap(x) for x in np.arange(0, 1.5, 0.33)]
        for suf, color in zip(suffixes, c) :
            colors[suf] = color
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], color = 'w', markerfacecolor = colors[suf], marker = 's', markersize = 10, markeredgecolor = 'k',  
                   markeredgewidth = .1, label = suf) for suf in suffixes
        ]
        ax.legend(handles = legend_elements, fontsize = 24, markerscale = 2.5)
        plt.savefig(os.path.join(out_folder, 'venn.png'))
        # logging.info('Venn diagram created')
        
    return total_de_prots


def diffacto_call(diffacto_path='',
                  out_path='',
                  peptide_path='',
                  sample_path='',
                  min_samples=3,
                  psm_dfs_dict={},
                  samples_dict={},
                  write_peptides=False,
                  str_of_other_args=''
                  ) :
    # samples_dict = {
    #     's1': ['rep_name1', 'rep_name2', ...]
    #     's2': ['rep_name3', 'rep_name4', ...]
    # }
    # psm_dfs_dict = {
    #     'rep_name1' : dataframe,
    #     'rep_name2': dataframe, ...
    # }
    # dataframe columns: ['peptide', 'protein', 'intensity']
    sample_nums = list(samples_dict.keys())
    samples = []
    for s in sample_nums :
        samples += samples_dict[s]
    
    if not os.path.exists(diffacto_path) :
        logging.critical('path to diffacto file is required')
        return -1
    
    if write_peptides :
        logging.info('Diffacto writing peptide file START')
        df0 = psm_dfs_dict[samples[0]][['peptide', 'protein']]
        for sample in samples :
            psm_dfs_dict[sample].rename(columns={'intensity':sample}, inplace=True)
            df0 = df0.merge(
                psm_dfs_dict[sample],
                how = 'outer',
                on = ['peptide', 'protein', ],
                )
        df0.fillna(value='', inplace=True)
        df0.to_csv(peptide_path, sep=',', index=False)
        logging.info('DONE')
    else :
        logging.info('Diffacto is using existing peptide file')
    
    logging.info('Diffacto writing sample file START')
    out = open( sample_path , 'w')
    for sample_num in sample_nums :
        for sample in samples_dict[sample_num] :
            label = sample
            out.write(label + '\t' + sample_num + '\n')
            logging.info(label + '\t' + sample_num)
    out.close()
    logging.info('DONE')
    
    other_args = [x.strip() for x in str_of_other_args.split(' ')]
    final_args = ['python3', diffacto_path, 
                  '-i', peptide_path, 
                  '-out', out_path, 
                  '-samples', sample_path, ] + ['-min_samples', min_samples] + other_args
    final_args = list(filter(lambda x: False if x=='' else True, final_args))
    logging.info('Diffacto START')
    process = subprocess.Popen(final_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout)
    exitscore = process.wait()
    logging.debug(exitscore)
    
    if exitscore == 0 :
        logging.info('DONE')
    else :
        logging.critical('Diffacto error: ', exitscore)
    
    return exitscore


# gets dict with PATHS to (peptide-intensity) files, merge it to one big table, mix intensities 
# and returns dict with peptide-intensity tables
# optionally write it as separate and/or diffacto-peptides-like files
def mix_intensity(input_dict,
                  samples_dict,
                  choice=4,
                  suf_dict={'dino':'D', 'bio': 'B', 'bio2':'B2', 'openMS':'O', 'mixed':'M'}, 
                  out_dir=None,
                  default_order=None,
                  to_diffacto=None, 
                  ) :
    # input_dict =   {'suf1' : path_to_diffacto_peptide_file
    #                 ...
    # }
    # samples_dict = {
    #     's1': ['rep_name1', 'rep_name2', ...]
    #     's2': ['rep_name3', 'rep_name4', ...]
    # }
    # default_order = ['dino', 'bio2', 'openMS'] in order of decreasing number of detected DE proteins in standalone diffacto runs
    # suf_dict = {'dino': 'D', 'bio2': 'B2', ...}
    sample_nums = list(samples_dict.keys())
    suffixes = [suf for suf in input_dict]    
    samples = []
    for s in sample_nums :
        samples += samples_dict[s]
    
    logging.info('Merging START')
    i = 0
    merge_df = pd.read_csv(input_dict[suffixes[0]], sep=',', usecols=['peptide', 'protein', ])
    for suf in suffixes :
        temp_df = pd.read_csv(input_dict[suf], sep=',')
        sufs = (None, '_'+suf_dict[suf] )
        merge_df = merge_df.merge(temp_df, 
                                  how = 'outer', 
                                  on = ['peptide', 'protein', ], 
                                  suffixes = sufs,
                                  copy=False,
                                 )
        temp_df = False
    rename_dct = {}
    for sample in samples :
        rename_dct[sample] = sample + '_' + suf_dict[suffixes[0]]
    merge_df.rename(columns = rename_dct, inplace=True)
    
    cols = [ col for col in merge_df.columns if col not in ['peptide', 'protein', ] ]
    merge_df.loc[:, cols].fillna(value=0, inplace=True)

    logging.info('Merging DONE')
    
    med_cv_to_corr = {}
    for suf in suffixes : 
        short_suf = suf_dict[suf]
        med_cv_to_corr[suf] = {}
        for sample_num in sample_nums : # s = 's1' or 's2'
            suf_cols = [sample +'_'+ short_suf for sample in samples_dict[sample_num]]
            col = sample_num+'_'+'not_NaN'+'_'+short_suf
            merge_df[col] = merge_df[suf_cols].notna().sum(axis = 1)
            
            cols = [x for x in merge_df.columns if x.startswith(tuple(samples_dict[sample_num])) and x.endswith(short_suf)]

            mean_col = sample_num+'_'+'mean'+'_'+short_suf
            merge_df[ mean_col ] = merge_df.loc[:, cols].mean(axis=1)

            std_col = sample_num+'_'+'std'+'_'+short_suf
            merge_df[ std_col ] = merge_df.loc[:, cols].std(axis=1)

            cv_col = sample_num+'_'+'cv'+'_'+short_suf
            merge_df[ cv_col ] = merge_df[ std_col ] / merge_df[ mean_col ]

            not_na_col = sample_num+'_'+'not_NaN'+'_'+ short_suf
            av = merge_df[ cv_col ].mean()
            merge_df[ cv_col ].mask( merge_df[ not_na_col ] == 1 , other = av, inplace = True)
            
            med_cv_to_corr[suf][sample_num] = merge_df[cv_col].median()
            
        suf_cols = [sample +'_'+ short_suf for sample in samples]
        col = 'num_NaN'+'_'+short_suf
        merge_df[col] = merge_df[suf_cols].isna().sum(axis = 1)

    for suf in suffixes :
        short_suf = suf_dict[suf]
        for sample_num in samples_dict :
            cols = [x for x in merge_df.columns if x.startswith(tuple(samples_dict[sample_num])) and x.endswith(short_suf)]

            cv_col = sample_num+'_'+'cv'+'_'+short_suf
            corr_cv_col = 'corr_'+sample_num+'_'+'cv'+'_'+short_suf
            ref_med = max([med_cv_to_corr[suf][sample_num] for suf in suffixes])
            merge_df[ corr_cv_col ] = merge_df[ cv_col ] / (med_cv_to_corr[suf][sample_num] / ref_med)

        merge_df[ 'summ_cv'+'_'+short_suf ] = merge_df['s1_cv'+'_'+short_suf ] + merge_df['s2_cv'+'_'+short_suf ]
        merge_df[ 'max_cv'+'_'+short_suf] = merge_df.loc[:, ['s1_cv'+'_'+short_suf , 's2_cv'+'_'+short_suf ] ].max(axis=1)
        merge_df[ 'sq_summ_cv'+'_'+short_suf ] = np.sqrt(merge_df['s1_cv'+'_'+short_suf ]**2 + merge_df['s2_cv'+'_'+short_suf ]**2)
        merge_df[ 'corr_sq_cv'+'_'+short_suf ] = np.sqrt(merge_df['corr_s1_cv'+'_'+short_suf ]**2 + merge_df['corr_s2_cv'+'_'+short_suf ]**2)
        merge_df.drop(columns=[ 's1_mean'+'_'+short_suf, 's2_mean'+'_'+short_suf, 
                                's1_std'+'_'+short_suf, 's2_std'+'_'+short_suf,
                                's1_'+'not_NaN' + '_' + short_suf, 's2_'+'not_NaN' + '_' + short_suf 
                              ], inplace=True)
        
    # min number of Nan values + default order (in order with maximum differentially expressed proteins in single diffacto runs)
    if choice == 0 :
        num_na_cols = ['num_NaN_' + suf for suf in default_order]
        merge_df['tool'] = merge_df[num_na_cols].idxmin(axis=1)
        merge_df['tool'] = merge_df['tool'].apply(lambda x: x.split('_')[-1])

    # min number of Nan values and min summ or or max or sq summ or sq summ of corrected CV
    elif choice == 1 or choice == 2 or choice == 3 or choice == 4 :
        num_na_cols = ['num_NaN' +'_'+ suf_dict[suf] for suf in suffixes]
        merge_df['NaN_border'] = merge_df[num_na_cols].min(axis=1)
        if choice== 1 :
            cv = 'summ_cv'
        elif choice == 2 :
            cv = 'max_cv'
        elif choice == 3 :
            cv = 'sq_summ_cv'
        elif choice == 4 :
            cv = 'corr_sq_cv'
        for suf in suffixes :
            short_suf = suf_dict[suf]
            merge_df['masked_CV' +'_'+short_suf ] = merge_df[ cv + '_'+short_suf ].mask(merge_df[ 'num_NaN' + '_'+short_suf ] > merge_df['NaN_border'])
        masked_CV_cols = ['masked_CV' + '_'+suf_dict[suf] for suf in suffixes]
        merge_df['tool'] = merge_df[masked_CV_cols].idxmin(axis=1)
        merge_df['tool'].mask(merge_df['tool'].isna(), other=merge_df[num_na_cols].idxmin(axis=1), inplace=True)
        merge_df['tool'] = merge_df['tool'].apply(lambda x: x.split('_')[-1])

    else :
        logging.critical('Invalid value for choice parameter: %s', choice)
        raise ValueError('Invalid value for choice parameter: %s', choice)
        return -2
    
    logging.info('Choosing intensities START')
    out_dict = {}
    for sample in samples :
        cols = [x for x in merge_df.columns if x.startswith(sample)] + ['tool']
        merge_df[sample] = merge_df[cols].apply(lambda x: x[ sample+'_'+x['tool'] ], axis = 1)
        out_dict[sample] = merge_df[['peptide', 'protein', sample, 'tool']]
        if out_dir :
            name = sample + '_mixed.tsv'
            t = merge_df[['peptide', 'protein', sample, 'tool']]
            t.rename(columns={sample: 'intensity'})
            t.to_csv( os.path.join(out_dir, name), 
                      sep='\t', 
                      index = list(merge_df.index), 
                      columns = ['peptide', 'protein', sample, 'tool'], 
                      encoding = 'utf-8'
                    )
    
    logging.info('Choosing intensities DONE')
    if to_diffacto :
        cols = ['peptide', 'protein'] + [sample for sample in samples]
        merge_df.to_csv(to_diffacto, 
                        sep=',', 
                        index = False,
                        columns = cols, 
                        encoding = 'utf-8')
    
    logging.info('Writing peptides files for Diffacto Mixed DONE')
    if out_dir :
        merge_df.to_csv(os.path.join(out_dir, 'mixing_all.tsv'), 
                        sep='\t', 
                        index = list(merge_df.index), 
                        columns = list(merge_df.columns), 
                        encoding = 'utf-8')
        
    logging.info('Mixing intensities DONE')
    if to_diffacto :
        return 0
    else :
        return out_dict


# make one intensity for all charge and FAIMS states of the peptide
# input: PSMs_full-like table (path to it)
# output: table 'peptide', 'protein', 'intensity'
# with only one occurrence of each peptide
# for Diffacto input
def charge_states_intensity_processing(path, 
                                       method='z-attached', 
                                       intens_colomn_name='feature_intensityApex',
                                       allowed_peptides=None, # set()
                                       allowed_prots=None, # set()
                                       out_path=None
                                      ) :
    psm_df = pd.read_table(path, sep='\t')
    
    cols = list(psm_df.columns)
    needed_cols = ['peptide', 'protein', 'assumed_charge', intens_colomn_name]
    for col in needed_cols :
        if col not in cols :
            logging.critical('Not all needed columns are in file: '+col+' not exists')
            raise ValueError('Not all needed columns are in file: '+col+' not exists')
    
    if allowed_peptides :
        psm_df = psm_df[psm_df['peptide'].apply(lambda z: z in allowed_peptides)]
    if allowed_prots :
        psm_df['protein'] = psm_df['protein'].apply(lambda z: ';'.join([u for u in ast.literal_eval(z) if u in allowed_prots]))
        psm_df = psm_df[psm_df['protein'].apply(lambda z: z != '')]
    
    if 'compensation_voltage' in cols :
        unique_comb_cols = ['peptide', 'assumed_charge', 'compensation_voltage']
    else :
        unique_comb_cols = ['peptide', 'assumed_charge',]
    
    psm_df = psm_df.sort_values(intens_colomn_name, ascending=False).drop_duplicates(subset=unique_comb_cols)
    if 'compensation_voltage' in cols :
        psm_df['peptide'] = psm_df.apply(lambda z: z['peptide'] + '_' + str(z['compensation_voltage']), axis=1)
    
    if method == 'z-attached' :
# use peptide sequence + charge + FAIMS in 'peptide' column as name of the peptide
# to use different charge states of one peptide as different peptides in quantitation
        psm_df['peptide'] = psm_df.apply(lambda z: z['peptide'] + '_' + str(z['assumed_charge']), axis=1)
        psm_df = psm_df.drop_duplicates(['peptide'], keep='first') # keeps max intensity according previous sort
    
    elif method == 'max_intens' :
# use max intensity between charge states as intensity for the peptide 
        psm_df[intens_colomn_name] = psm_df.groupby('peptide')[intens_colomn_name].transform(max)
        psm_df = psm_df.drop_duplicates(['peptide'], keep='first')
    
    elif method == 'summ_intens' :
# use sum of the intensities for all charge states as an intesity for the peptide
        psm_df[intens_colomn_name] = psm_df.groupby('peptide')[intens_colomn_name].transform(sum)
        psm_df = psm_df.drop_duplicates(['peptide'], keep='first')
    
    else :
        logging.critical('Invalid value for method: %s', method)
        raise ValueError('Invalid value for method: %s', method)

    psm_df.rename(columns={intens_colomn_name: 'intensity'}, inplace = True)
    psm_df = psm_df[['peptide', 'protein', 'intensity']]
    
    if out_path :
        psm_df.to_csv(out_path, 
                     sep='\t', 
                     index = False,
                     columns = list(psm_df.columns), 
                     encoding = 'utf-8')

    return psm_df


def read_PSMs(infile_path, usecols=None) :
    if infile_path.endswith('.tsv') :
        df1 = pd.read_csv(infile_path, sep = '\t', usecols=usecols)            
    elif infile_path.lower().endswith('.pep.xml') or infile_path.lower().endswith('.pepxml') :
        df1 = pepxml.DataFrame(infile_path)
        ftype = 'pepxml'
    elif infile_path.lower().endswith('.mzid') :
        df1 = mzid.DataFrame(infile_path)
    else:
        raise WrongInputError()
    if not df1.shape[0]:
        raise EmptyFileError()
    
    if 'MS-GF:EValue' in df1.columns:
        # MSGF search engine
        ftype = 'msgf'
        df1.rename(columns={'PeptideSequence':'peptide', 
                            'chargeState':'assumed_charge',
                            'spectrumID':'spectrum',
                            'accession':'protein',
                            'protein description':'protein_descr',
                            'MS-GF:EValue':'expect',
                           }, 
                   inplace=True)
        df1['precursor_neutral_mass'] = df1['calculatedMassToCharge'] * df1['assumed_charge'] - df1['assumed_charge'] * 1.00727649

    if set(df1['protein_descr'].str[0]) == {None}:
        # MSFragger
#        logger.debug('Adapting MSFragger DataFrame.')
#        logger.debug('Proteins before: %s', df1.loc[1, 'protein'])
        protein = df1['protein'].apply(lambda row: [x.split(None, 1) for x in row])
        df1['protein'] = protein.apply(lambda row: [x[0] for x in row])
#        logger.debug('Proteins after: %s', df1.loc[1, 'protein'])
    
    df1.loc[pd.isna(df1['protein_descr']), 'protein_descr'] = df1.loc[pd.isna(df1['protein_descr']), 'protein']
    df1 = df1[~pd.isna(df1['peptide'])]

    df1['spectrum'] = df1['spectrum'].apply(lambda x: x.split(' RTINS')[0])
    
    if 'RT exp' not in df1.columns :
        if 'retention_time_sec' not in df1.columns:
            if 'scan start time' in df1.columns:
                df1.rename(columns={'scan start time':'RT exp'}, inplace=True) 
            else:
                df1['RT exp'] = 0
        else:
            df1['RT exp'] = df1['retention_time_sec'] / 60
            df1 = df1.drop(['retention_time_sec', ], axis=1)
    
    cols = ['spectrum', 'peptide', 'protein', 'assumed_charge', 'precursor_neutral_mass', 'RT exp']
    if 'q' in df1.columns :
        cols.append('q')
    if 'ionmobility' in df1.columns and 'im' in df1.columns :
        cols.append('ionmobility')
    if 'compensation_voltage' in df1.columns :
        cols.append('compensation_voltage')
        
    df1 = df1[cols]
    
    return df1


## Функции для сопоставления

    
def noisygaus(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def opt_bin(ar, border=16) :
    num_bins = 4
    bwidth = (max(ar) - min(ar))/num_bins
    bbins = np.arange(min(ar), max(ar), bwidth)
    H1, b1 = np.histogram(ar, bins=bbins)
    max_percent = 100*max(H1)/sum(H1)

    bestbins1 = num_bins
    bestbins2 = num_bins
    mxp2 = max_percent
    mxp1 = max_percent
    while max_percent > border :
        num_bins = num_bins*2

        bwidth = (max(ar) - min(ar))/num_bins
        bbins = np.arange(min(ar), max(ar), bwidth)
        H1, b1 = np.histogram(ar, bins=bbins)
        max_percent = 100*max(H1)/sum(H1)
        if max_percent < border :
            bestbins1 = num_bins
            mxp1 = max_percent
    while max_percent < border :
        if num_bins < 16 :
            num_bins -= 1
        else :
            num_bins = round(num_bins/1.1, 0)

        bwidth = (max(ar) - min(ar))/num_bins
        bbins = np.arange(min(ar), max(ar), bwidth)
        H1, b1 = np.histogram(ar, bins=bbins)
        max_percent = 100*max(H1)/sum(H1)
        if max_percent < border :
            bestbins2 = num_bins
            mxp2 = max_percent
    if abs(mxp1 - border) < abs(mxp2 - border) :
        bestbins = bestbins1
    else :
        bestbins = bestbins2
    bwidth = (max(ar) - min(ar))/bestbins    
    bbins = np.arange(min(ar), max(ar), bwidth)
    H1, b1 = np.histogram(ar, bins=bbins)
    max_percent = 100*max(H1)/sum(H1)
    logging.debug('final num_bins: ' + str(int(num_bins)) + '\t' + 'final max percent per bin: ' + str(round(max_percent, 2)) + '%')

    return bwidth


def calibrate_mass(mass_left, mass_right, true_md, check_gauss=False, sort = 'mz') :

    bwidth = opt_bin(true_md)
    bbins = np.arange(mass_left, mass_right, bwidth)
    H1, b1 = np.histogram(true_md, bins=bbins)
    b1 = b1 + bwidth
    b1 = b1[:-1]

    H_marg = 2*np.median(H1)
    i = np.argmax(H1)
    max_k = len(H1) - 1
    j = i
    k = i
    while j >= 0 and H1[j] > H_marg:
        j -= 1
    while k <= max_k and H1[k] > H_marg:
        k += 1            
    w = (k-j)
    t = []
#        logging.debug('Интервал значений ' + str(b1[ll]-bwidth) + ' ' + str(b1[rr]))
    for el in true_md :
        if el >= b1[i]-bwidth*2*(i-j) and el <= b1[i]+bwidth*2*(k-i) :
            t.append(el)

    bwidth = opt_bin(t)
    bbins = np.arange(min(t), max(t) , bwidth)
    H2, b2 = np.histogram(t, bins=bbins)
    # pickle.dump(t, open('/home/leyla/project1/mbr/t.pickle', 'wb'))
    m = max(H2)
    mi = b2[np.argmax(H2)]
    s = (max(t) - min(t))/6
    noise = min(H2)
    if (sort == 'rt1') or (sort == 'rt2'):
        popt, pcov = curve_fit(noisygaus, b2[1:][b2[1:]>=mi], H2[b2[1:]>=mi], p0=[m, mi, s, noise])
    else:
        popt, pcov = curve_fit(noisygaus, b2[1:], H2, p0=[m, mi, s, noise])
    # popt, pcov = curve_fit(noisygaus, b2[1:], H2, p0=[m, mi, s, noise])
    logging.debug(popt)
    mass_shift, mass_sigma = popt[1], abs(popt[2])

    if check_gauss:
        logging.debug('GAUSS FIT, %f, %f' % (percentileofscore(t, mass_shift - 3 * mass_sigma), percentileofscore(t, mass_shift + 3 * mass_sigma)))

        if percentileofscore(t, mass_shift - 3 * mass_sigma) + 100 - percentileofscore(t, mass_shift + 3 * mass_sigma) > 10:
            mass_sigma = scoreatpercentile(np.abs(t-mass_shift), 95) / 2
            
    logging.debug('shift: ' + str(mass_shift) + '\t' + 'sigma: ' + str(mass_sigma))
    return mass_shift, mass_sigma, pcov[0][0]


def total(df_features, psms, mean1=0, sigma1=False, mean2 = 0, sigma2=False, mean_mz=0, mass_accuracy_ppm=10, mean_im = 0, sigma_im = False, isotopes_array=[0, ]):

    # df_features = df_features[['mz', 'charge', 'rtStart', 'rtEnd', 'id', 'intensityApex'б]]
    # print(mean1, sigma1, mean2, sigma2, mean_mz, mass_accuracy_ppm, mean_im, sigma_im , isotopes_array)
    df_features = df_features.sort_values(by='mz')
    mz_array_ms1 = df_features['mz'].values
    # ch_array_ms1 = df_features['charge'].values
    rtStart_array_ms1 = df_features['rtStart'].values
    rtEnd_array_ms1 = df_features['rtEnd'].values
    feature_intensityApex = df_features['intensityApex'].values
    
    feature_id = df_features.index #new

    check_charge = False
    if 'charge' in df_features.columns:
        ch_array_ms1 = df_features['charge'].values
        check_charge = True

    check_FAIMS = False
    if 'FAIMS' in df_features.columns:
        FAIMS_array_ms1 = df_features['FAIMS'].values
        if len(set(FAIMS_array_ms1)) != 1:
            check_FAIMS = True

    check_im = False
    if 'im' in df_features.columns:
        im_array_ms1 = df_features['im'].values
        if len(set(im_array_ms1)) != 1:
            check_im = True
            if sigma_im is False:
                max_im = max(im_array_ms1)
                min_im = min(im_array_ms1)
                sigm_im = (max_im - min_im)/2
            else:
                sigm_im = sigma_im

    results = defaultdict(list)

    if sigma1 is False:
        max_rtstart_err = max(rtStart_array_ms1)/25
        interval1 = max_rtstart_err
    else:
        interval1 = 3*sigma1

    if sigma2 is False:
        max_rtend_err = max(rtEnd_array_ms1)/25
        interval2 = max_rtend_err
    else:
        interval2 = 3*sigma2

    for i in isotopes_array: 
        for index, row in psms.iterrows(): 
            psms_index = row['spectrum']  
            peptide = row['peptide']
            psm_mass = row['precursor_neutral_mass']
            psm_charge = row['assumed_charge']
            psm_rt = row['RT exp']
            psm_mz = (psm_mass+psm_charge*1.007276)/psm_charge
            protein = row['protein']
            if check_im:
                if 'im' in row:
                    psm_im = row['ionmobility']
                else:
                    check_im = False
                    logging.info('there is no column "IM" in the PSMs')
            if check_FAIMS:
                if 'compensation_voltage' in row:
                    psm_FAIMS = row['compensation_voltage']
                else:
                    check_FAIMS = False
                    logging.info('there is no column "FAIMS" in the PSMs')
            if psms_index not in results:
                a = psm_mz/(1 - mean_mz*1e-6) -  i*1.003354/psm_charge 
                mass_accuracy = mass_accuracy_ppm*1e-6*a
                idx_l_psms1_ime = mz_array_ms1.searchsorted(a - mass_accuracy)
                idx_r_psms1_ime = mz_array_ms1.searchsorted(a + mass_accuracy, side='right')

                for idx_current_ime in range(idx_l_psms1_ime, idx_r_psms1_ime, 1):
                    # if np.nonzero(FAIMS_array_ms1[idx_current_ime])[0].size != 0:
                    if check_FAIMS:

                        if FAIMS_array_ms1[idx_current_ime] == psm_FAIMS:
                            pass
                        else:
                            continue 

                    if not check_charge or ch_array_ms1[idx_current_ime] == psm_charge:
                        rtS = rtStart_array_ms1[idx_current_ime]
                        rtE = rtEnd_array_ms1[idx_current_ime]
                        rt1 = psm_rt - rtS
                        rt2 = rtE - psm_rt
                        # if  psm_rt  - mean1> rtS- interval1    and  psm_rt + mean2 < rtE +interval2:
                        if rt1 >= min(0, mean1 - interval1) and rt2 >= min(0, mean2 - interval2):
                            ms1_mz = mz_array_ms1[idx_current_ime]
                            mz_diff_ppm = (ms1_mz - a) / a * 1e6
                            rt_diff = (rtE - rtS)/2+rtS - psm_rt
                            rt_diff1 = psm_rt - rtS
                            rt_diff2 = rtE - psm_rt
                            intensity = feature_intensityApex[idx_current_ime]
                            index = feature_id[idx_current_ime] #new
                            
                            cur_result = {'idx_current_ime': idx_current_ime,
                                         'id_feature': index, #new
                                         'mz_diff_ppm':mz_diff_ppm,
                                         'rt_diff':rt_diff,
                                         'im_diff': 0,
                                         'i':i,
                                         'rt1':rt_diff1,
                                         'rt2':rt_diff2,
                                         'intensity':intensity}
                            if check_im:
                                im_ms1 = im_array_ms1[idx_current_ime]
                                if im_ms1 - sigm_im < psm_im + mean_im < im_ms1 + sigm_im:
                                    im_diff = im_ms1 - psm_im
                                    cur_result['im_diff'] = im_diff
                                    results[psms_index].append(cur_result)
                                    # results[psms_index].append((idx_current_ime,mz_diff_ppm, rt_diff,im_diff, i,rt_diff1, rt_diff2, intensity))
                            else:
                                results[psms_index].append(cur_result)
                                # results[psms_index].append((idx_current_ime,mz_diff_ppm, rt_diff,0, i,rt_diff1, rt_diff2,intensity))

    return results


def found_mean_sigma(df_features,psms,parameters, sort ='mz_diff_ppm' , mean1=0,sigma1=False,mean2=0,sigma2=False, mean_mz = 0, sigma_mz = False):
# sort ='mz_diff_ppm'
    check_gauss = False
    rtStart_array_ms1 = df_features['rtStart'].values
    rtEnd_array_ms1 = df_features['rtEnd'].values

    if parameters == 'rt1' or parameters == 'rt2':
        results_psms = total(df_features = df_features,psms = psms,mass_accuracy_ppm = 100)

    if parameters == 'mz_diff_ppm' :
        check_gauss = True
        results_psms = total(df_features =df_features,psms =psms,mean1 = mean1,sigma1 = sigma1, mean2 = mean2,sigma2 = sigma2,mass_accuracy_ppm = 100)

    if parameters == 'im_diff':
        results_psms = total(df_features =df_features,psms =psms,mean1 = mean1,sigma1 = sigma1, mean2 = mean2,sigma2 = sigma2, mean_mz = mean_mz, mass_accuracy_ppm = 3*sigma_mz)

        if 'im' in df_features.columns:
            im_array_ms1 = df_features['im'].values
            if len(set(im_array_ms1)) != 1:
                pass
            else:
                return 0,0
        else:
            return 0,0
    
    ar = []
    for value in results_psms.values():
        if sort == 'intensity':
            ar.append(sorted(value, key=lambda x: -abs(x[sort]))[0][parameters])
        elif sort == 'dist':
            # lambda x: np.sqrt(x['mz_diff_ppm']**2 + x['rt1']**2 + x['rt2']**2 
            ar.append(sorted(value, key=dist)[0][parameters])
        else:
            ar.append(sorted(value, key=lambda x: abs(x[sort]))[0][parameters])

    if parameters == 'rt1' or parameters == 'rt2':
        mean, sigma,_ = calibrate_mass(min(ar),max(ar),ar, check_gauss, 'rt1')
    else:
        mean, sigma,_ = calibrate_mass(min(ar),max(ar),ar, check_gauss, 'mz')
    return mean, sigma


def optimized_search_with_isotope_error_(df_features,psms,mean_rt1=False,sigma_rt1=False,mean_rt2=False,sigma_rt2=False,mean_mz = False,sigma_mz = False,mean_im = False,sigma_im = False, isotopes_array=[0,1,-1,2,-2]):
    
    idx = {}
    for j, i in enumerate(isotopes_array):
        idx[i] = j
    
    if mean_rt1 is False and sigma_rt1 is False:
        logging.debug('rt1')
        mean_rt1, sigma_rt1 = found_mean_sigma(df_features,psms, 'rt1')

    if mean_rt2 is False and sigma_rt2 is False:
        logging.debug('rt2')
        mean_rt2, sigma_rt2 = found_mean_sigma(df_features,psms, 'rt2')

    if mean_mz is False and sigma_mz is False:
        logging.debug('mz')
        mean_mz, sigma_mz = found_mean_sigma(df_features,psms,'mz_diff_ppm', mean1 = mean_rt1, sigma1 = sigma_rt1, mean2 = mean_rt2,sigma2 = sigma_rt2)   

    if mean_im is False and sigma_im is False:
        logging.debug('im')
        mean_im, sigma_im = found_mean_sigma(df_features,psms,'im_diff', mean1 = mean_rt1, sigma1 = sigma_rt1, mean2 = mean_rt2,sigma2 = sigma_rt2, mean_mz = mean_mz, sigma_mz= sigma_mz)  

    # print(mean_rt1, sigma_rt1,mean_rt2, sigma_rt2,mean_mz, sigma_mz )    

    results_isotope = total(df_features = df_features,psms =psms,mean1 = mean_rt1, sigma1 = sigma_rt1,mean2 = mean_rt2, sigma2 = sigma_rt2, mean_mz = mean_mz, mass_accuracy_ppm = 3*sigma_mz, isotopes_array=isotopes_array)
    
    results_isotope_end = []
    cnt = Counter([z[0]['i'] for z in results_isotope.values()])
    for i in cnt.values():
        results_isotope_end.append(i/len(psms))
    end_isotope_ = list(np.add.accumulate(np.array(results_isotope_end))*100)
    logging.info(end_isotope_)
    df_features_dict = {}
    # intensity_dict = {}
    intensity_dict = defaultdict(float)
    for kk,v in results_isotope.items():
        tmp = sorted(v, key=lambda x: 1e6*idx[x['i']] + np.sqrt( (x['mz_diff_ppm']/sigma_mz)**2 + min([0, (x['rt1']/sigma_rt1)**2, (x['rt2']/sigma_rt2)**2])))[0]
        # for i in tmp:
        #     df_features_dict[kk] = i['id_feature']
        #     intensity_dict[kk] += i['intensity']#*(1 if i['intensity'] >=tmp[0]['intensity'] else 0)
        # tmp = sorted(v, key=lambda x: -x['intensity'])[0]
        df_features_dict[kk] = tmp['id_feature'] #new
        intensity_dict[kk] = tmp['intensity']
    ser1 = pd.DataFrame(df_features_dict.values(), index = df_features_dict.keys(), columns = ['df_features'])
    ser2 = pd.DataFrame(df_features_dict.keys(), index = df_features_dict.keys(), columns = ['spectrum'])
    ser3 = pd.DataFrame(intensity_dict.values(), index = intensity_dict.keys(), columns = ['feature_intensityApex'])
    s = pd.concat([ser1,ser2],sort = False,axis = 1 )
    ss = pd.concat([s,ser3],sort = False,axis = 1 )
    features_for_psm_db = pd.merge(psms,ss,on = 'spectrum',how='outer')
    return features_for_psm_db,end_isotope_, cnt.keys()
# end_isotope_, cnt.keys(),

## match between run

def mbr(feat,II,PSMs_full_paths, PSM_path):
    II = II.sort_values(by = 'feature_intensityApex', ascending = False)
    II['pep_charge'] = II['peptide'] + II['assumed_charge'].map(str)
    II = II.drop_duplicates(subset = 'pep_charge')
    match_between_runs_copy01 = II.copy()
    match_between_runs_copy01['pep_charge'] = np.where(~np.isnan(match_between_runs_copy01['feature_intensityApex']), match_between_runs_copy01['pep_charge'], np.nan)
    match_between_runs_copy01 = match_between_runs_copy01[match_between_runs_copy01['pep_charge'].notna()]
    found_set = set(match_between_runs_copy01['pep_charge'])
    for j in PSMs_full_paths:
        if PSM_path != j:
            psm_j_fdr = read_PSMs(j)
            psm_j_fdr['pep_charge'] = psm_j_fdr['peptide'] + psm_j_fdr['assumed_charge'].map(str)
            psm_j_fdr = psm_j_fdr[psm_j_fdr['pep_charge'].apply(lambda x: x not in found_set)]
            III = optimized_search_with_isotope_error_(feat, psm_j_fdr, isotopes_array=[0,1,-1,2,-2])[0]
            III = III.sort_values(by = 'feature_intensityApex', ascending = False)
            III = III.drop_duplicates(subset = 'pep_charge')
            II = pd.concat([II, III])
    return II


def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.info('From subprocess: %r', line)


def run():
    parser = argparse.ArgumentParser(
        description = 'run multiple feature detection matching and diffacto for scavager results',
        epilog = '''
    Example usage
    -------------
    (prefered way)
    $ multi_features_diffacto_analisys.py -cfg /path_to/default.ini
        
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
    
    parser.add_argument('-logs', help='level of logging, (DEBUG, INFO, WARNING, ERROR, CRITICAL)', )
    parser.add_argument('-log_path', help='path to logging file', )
    parser.add_argument('-cfg', help='path to config .ini file')
    parser.add_argument('-cfg_category', help='name of category to prioritize in the .ini file, default: DEFAULT')
    parser.add_argument('-dif', help='path to Diffacto')
#    parser.add_argument('-scav2dif', help='path to scav2diffacto')
    parser.add_argument('-s1', nargs='+', help='input mzML files for sample 1 (also file names are the keys for searching other needed files)')
    parser.add_argument('-s2', nargs='+', help='input mzML files for sample 2 (also file names are the keys for searching other needed files)')
#    parser.add_argument('-sampleNames', nargs='+', help='short names for samples for inner structure of results')
    parser.add_argument('-PSM_folder', help='path to the folder with PSMs files')
    parser.add_argument('-PSM_format', help='format or suffix to search PSMs files (may be PSMs_full.tsv or identipy.pep.xml for example)')
    parser.add_argument('-pept_folder', help='path to folder with files with peptides filtered on certain FDR (default: searching for them near PSMs)')
    parser.add_argument('-prot_folder', help='path to folder with files with proteins filtered on certain FDR (default: searching for them near PSMs)')
    
    parser.add_argument('-dino', help='path to Dinosaur')
    parser.add_argument('-bio', help='path to Biosaur')
    parser.add_argument('-bio2', help='path to Biosaur2')
    parser.add_argument('-openMS', help='path to OpenMS feature')
    
    parser.add_argument('-outdir', help='name of directory to store results')
    parser.add_argument('-feature_folder', help='directory to store features')
    parser.add_argument('-matching_folder', help='directory to store matched psm-feature pairs')
    parser.add_argument('-diffacto_folder', help='directory to store diffacto results')
    
    parser.add_argument('-overwrite_features', help='whether to overwrite existed features files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-overwrite_matching', help='whether to overwrite existed matched files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-overwrite_first_diffacto', help='whether to overwrite existed diffacto files (flag == 1) or use them (flag == 0)')
    parser.add_argument('-mixed', help='whether to reanalyze mixed intensities (1) or not (0)')
    parser.add_argument('-venn', help='whether to plot venn diagrams (1) or not (0)')
    parser.add_argument('-pept_intens', help='max_intens - as intensity for peptide would be taken maximal intens between charge states; summ_intens - as intensity for peptide would be taken sum of intens between charge states; z-attached - each charge state would be treated as independent peptide')
    parser.add_argument('-choice', help='method how to choose right intensities for peptide. 0 - default order and min Nan values, 1 - min Nan and min of sum CV, 2 - min Nan and min of max CV, 3 - min Nan and min of squared sum CV, 4 - default order with filling Nan values between programs (if using this variant -norm MUST be applied), 5 - min Nan and min of squared sum of corrected CV')
    parser.add_argument('-norm', help='normalization method for intensities. Can be 1 - median or 0 - no normalization')
    parser.add_argument('-isotopes', help='monoisotope error')
    
    parser.add_argument('-outPept', help='name of output diffacto peptides file (important: .txt)')
    parser.add_argument('-outSampl', help='name of output diffacto samples file (important: .txt)', )
    parser.add_argument('-outDiff', help='name of diffacto output file (important: .txt)', )
    parser.add_argument('-min_samples', help='minimum number of samples for peptide usage', )
    
    parser.add_argument('-mbr', help='match between runs', )
    parser.add_argument('-pval_treshold', help='P-value treshold for reliable differetially expressed proteins', )
    parser.add_argument('-fc_treshold', help='Fold change treshold for reliable differetially expressed proteins', )
    parser.add_argument('-dynamic_fc_treshold', help='whether to apply dynamically calculated treshold (1) or not and use static -fc_treshold (0) ',)
    
    parser.add_argument('-diffacto_args', help='String of additional arguments to submit into Diffacto (hole string in single quotes in command line) except: -i, -out, -samples, -min_samples; default: "-normalize median -impute_threshold 0.25" ')
    parser.add_argument('-dino_args', help='String of additional arguments to submit into Dinosaur (hole string in single quotes in command line) except: --outDir --outName; default: ""')
    parser.add_argument('-bio_args', help='String of additional arguments to submit into Biosaur (hole string in single quotes in command line) except: -o; default: ""')
    parser.add_argument('-bio2_args', help='String of additional arguments to submit into Biosaur2 (hole string in single quotes in command line) except: -o; default: "-hvf 1000 -minlh 3"')
    parser.add_argument('-openms_args', help='String of additional arguments to submit into OpenMSFeatureFinder (hole string in single quotes in command line) except: -in, -out; default: "-algorithm:isotopic_pattern:charge_low 2 -algorithm:isotopic_pattern:charge_high 7"')
#    parser.add_argument('-version', action='version', version='%s' % (pkg_resources.require("scavager")[0], ))
    args = vars(parser.parse_args())

    default_config = configparser.ConfigParser(allow_no_value=True, empty_lines_in_values=False, )
    default_config.optionxform = lambda option: option
    d_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default.ini')
    if os.path.exists(d_cfg_path) :
        default_config.read(d_cfg_path)
    else :
        logging.critical('default config file does not exist: ' + d_cfg_path)
        return -1
    options = dict(default_config['DEFAULT'])
    
    if args['cfg'] :
        config = configparser.ConfigParser(allow_no_value=True, empty_lines_in_values=False, )
        config.optionxform = lambda option: option
        if os.path.exists(args['cfg']) :
            config.read(args['cfg'])
        else :
            logging.critical('path to config file does not exist')
            return -1
        if args['cfg_category'] :
            cat = args['cfg_category']
        else :
            cat = options['cfg_category']
                
        tmp = {k: v for k, v in dict(config[cat]).items() if v != ''}
        options.update( tmp )
    
    tmp = {k: v for k, v in args.items() if v is not None}
    args.clear()
    args.update(tmp)
    tmp.clear()
    
    options.update(args)
    
    for k, v in options.items() :
        if v == '' :
            options.update( {k: None} )
    
    args = options.copy()
    options.clear()
    
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
    
    if not args['outdir'] :
        logging.critical('path to output directory is required')
        return -1
    
    arg_suff = ['dino', 'bio', 'bio2', 'openMS']
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

    args['overwrite_features'] = int( args['overwrite_features'])
    args['overwrite_matching'] = int( args['overwrite_matching'])
    args['overwrite_first_diffacto'] = int( args['overwrite_first_diffacto'])
    args['mixed'] = int( args['mixed'])
    args['venn'] = int( args['venn'])
    args['choice'] = int( args['choice'])
    args['norm'] = int( args['norm'])
    args['isotopes'] = [int(isoval.strip()) for isoval in args['isotopes'].split(',')]
    args['pval_treshold'] = float(args['pval_treshold'])
    args['fc_treshold'] = float(args['fc_treshold'])
    if args['dynamic_fc_treshold'] == '1' :
        args['dynamic_fc_treshold'] = True
    elif args['dynamic_fc_treshold'] == '0' :
        args['dynamic_fc_treshold'] = False
    else :
        logging.critical('Invalid value for setting: -dynamic_fc_treshold %s', args['dynamic_fc_treshold'])
        raise ValueError('Invalid value for setting: -dynamic_fc_treshold %s', args['dynamic_fc_treshold'])
        
    
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

### Biosaur

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio'] :
        for path, sample in zip(mzML_paths, samples) :
            outPath = os.path.join(feature_path, sample + '_features_bio.tsv')
            if args['overwrite_features'] == 1 or not os.path.exists(outPath) :
                logging.info('\n' + 'Writing features ' + ' bio ' + sample + '\n')
                exitscore = call_Biosaur2(args['bio'], path, outPath, args['bio_args'])
                logging.debug(exitscore)
            else :
                logging.info('\n' + 'Not overwriting features ' + ' bio ' + sample + '\n')

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

#    suffixes = ['dino', 'bio', 'bio2', 'openMS'] - уже заданы
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
            logging.info('Starting Diffacto run with %s', suf)
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
            pval_treshold = args['pval_treshold'], 
            fc_treshold = args['fc_treshold'], 
            dynamic_fc_treshold = args['dynamic_fc_treshold'], 
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
                              suf_dict={'dino':'D', 'bio': 'B', 'bio2':'B2', 'openMS':'O', 'mixed':'M'}, 
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
                pval_treshold = args['pval_treshold'], 
                fc_treshold = args['fc_treshold'], 
                dynamic_fc_treshold = args['dynamic_fc_treshold'], 
                save = save
            )
            logging.info('Numbers of differentially expressed proteins:')
            for suf in suffixes+['mixed'] :
                logging.info('{}: {}'.format(suf, num_changed_prots[suf]))
                
            logging.info('IQMMA finished')

if __name__ == '__main__':
    run()

