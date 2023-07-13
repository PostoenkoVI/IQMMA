import pandas as pd
import numpy as np
import copy
import ast
import os
import subprocess
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pyteomics
from matplotlib.ticker import PercentFormatter
from pyteomics import pepxml, mzid
import venn
from venn import venn
from scipy.stats import pearsonr, scoreatpercentile, percentileofscore
from scipy.optimize import curve_fit
import logging


def read_cfg(file, category) :
    with open(file, 'r') as ff :
        f = ff.read()
    
    start = int(f.find('['+category+']')) + 2 + len(category)
    end = min(f.find('\n\n', start), len(f))
    
    cfg_string = f[start:end]
    while '#' in cfg_string :
        l = cfg_string.find('#')
        r = cfg_string.find('\n', l) + 1
        cfg_string = cfg_string[:l] + cfg_string[r:]

    lst_of_strings = cfg_string.lstrip().split('\n')
    final = []
    keys = []
    for el in lst_of_strings :
        if el :
            key_value = el.split(' = ')
            if len(key_value) > 1 :
                key = key_value[0]
                value = key_value[1]
            else :
                key = key_value[0]
                value = None
            keys.append(key)
            key = '-' + key
            final.append(key)

            if value :
                if value.startswith('\"') or value.startswith("\'") :
                    final.append(value)
                else :
                    vals = value.split()
                    for v in vals :
                        final.append(v)
    return final, keys


def write_example_cfg(path, dct_args):
    with open(path, 'w') as f :
        f.write('#encoding=\'utf-8\'\n')
        f.write('[DEFAULT]\n')
        for k, v in dct_args.items() :
            if type(v) != list :
                if type(v) == str and v.startswith('-') :
                    f.write(k + ' = ' + '\"' + str(v) + '\"' + '\n')
                else :
                    f.write(k + ' = ' + str(v) + '\n')
            else :
                f.write(k + ' = ' + ' '.join([str(el) for el in v]) + '\n')
        f.write('\n[users_category]\n')
        f.write('# here you can set your parameters\n')


def call_Dinosaur(path_to_fd, mzml_path, outdir, outname, str_of_other_args, logger = logging.getLogger('function') ) :
    if str_of_other_args :
        other_args = ['--' + x.strip().replace(' ', '=') for x in str_of_other_args.strip('"').strip("'").split('--')]
    else :
        other_args = []
    if path_to_fd.lower().endswith('jar') :
        final_args = ['java', '-jar', path_to_fd, mzml_path, '--outDir='+outdir, '--outName='+outname, ] + other_args
    else :
        final_args = [path_to_fd, mzml_path, '--outDir='+outdir, '--outName='+outname, ] + other_args
    final_args = list(filter(lambda x: False if x=='--' else True, final_args))
    process = subprocess.Popen(final_args, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger=logger)
    exitscore = process.wait()
    os.rename(os.path.join(outdir, outname + '.features.tsv'),  os.path.join(outdir, outname) )
    return exitscore

def call_Biosaur2(path_to_fd, mzml_path, outpath, str_of_other_args, logger = logging.getLogger('function')) :
    if str_of_other_args :
        other_args = [x.strip() for x in str_of_other_args.strip('"').strip("'").split(' ')]
    else :
        other_args = []
    final_args = [path_to_fd, mzml_path, '-o', outpath, ] + other_args
    final_args = list(filter(lambda x: False if x=='' else True, final_args))
    process = subprocess.Popen(final_args, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger=logger)
    exitscore = process.wait()
    return exitscore

def call_OpenMS(path_to_fd, mzml_path, outpath, str_of_other_args, logger = logging.getLogger('function')) :
    if str_of_other_args :
        other_args = [x.strip() for x in str_of_other_args.strip('"').strip("'").split(' ')]
    else :
        other_args = []
    final_args = [path_to_fd, 
                  '-in', mzml_path, 
                  '-out', outpath, 
                  ] + other_args
    final_args = list(filter(lambda x: False if x=='' else True, final_args))
    process = subprocess.Popen(final_args, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger=logger)
    exitscore = process.wait()
    return exitscore

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def generate_users_output(diffacto_out={},
                          out_folder='',
                          plot_venn=True,
                          pval_threshold=0.05,
                          pval_adj='Bonf',
                          fc_threshold=2,
                          dynamic_fc_threshold=True,
                          save = True, 
                          logger = logging.getLogger('function')
                          ) :
    # diffacto_out = {
    #     'suf1': path_to_diffacto_results,
    #     'suf2': path_to_diffacto_results,
    # }
    suffixes = [suf for suf in diffacto_out]
    
    flag = True
    comp_df = False
    i = 0
    for suf in suffixes :
        try :
            table = pd.read_csv(diffacto_out[suf], sep = '\t')
        except :
            i += 1
            logger.critical('Something goes wrong with diffacto output file {}'.format(diffacto_out[suf]))
            if i == len(suffixes) :
                logger.critical('All diffacto output is unreadable')
                return dict([(suf, 0) for suf in suffixes])
            continue
        table['log2_FC'] = np.log2(table['s1']/table['s2'])
        table['FC'] = table['s1']/table['s2']
        table.sort_values('P(PECA)', ascending=True, inplace=True)
        if flag :
            comp_df = table[['Protein']]
            flag = False
        if pval_adj == 'Bonf' :
            bonferroni = pval_threshold/len(table)
            table['pval_adj_pass'] = table['P(PECA)'] <= bonferroni
            logger.info('Bonferroni p value correction is applied')
        elif pval_adj == 'BH' :
            l = len(table)
            table['BH'] = pd.Series([(i+1)*pval_threshold/l for i in range(l)], index=table.index)
            table['pval_adj_pass'] = table['P(PECA)'] <= table['BH']
            logger.info('Benjamini–Hochberg p value correction procedure is applied')
        else :
            logger.critical('Wrong value for the `pval_adj` argument')
            return dict([(suf, 0) for suf in suffixes])
        
        table = table[ table['S/N'] > 0.01 ]
        
        if len(table[~table['pval_adj_pass']]['log2_FC']) < 100:
            logger.info('Low number of proteins for FC dynamic threshold')
            dynamic_fc_threshold = 0
        if not dynamic_fc_threshold :
            table = table[table['pval_adj_pass']][['Protein', 'P(PECA)', 'log2_FC']]
            logger.info('Static fold change threshold is applied')
            border_fc = fc_threshold
            table = table[abs(table['log2_FC']) > border_fc]
        else:
            t = table[~table['pval_adj_pass']]['log2_FC'].to_numpy()
            t = t[~np.isnan(t)]
            w = opt_bin(t, logger=logger)
            bbins = np.arange(min(t), max(t), w)
            H2, b2 = np.histogram(t, bins=bbins)
            m, mi, s = max(H2), b2[np.argmax(H2)], (max(t) - min(t))/6
            noise = min(H2)
            popt, pcov = curve_fit(noisygaus, b2[1:], H2, p0=[m, mi, s, noise])
            shift, sigma = popt[1], abs(popt[2])
            right_fc_threshold = shift + 3*sigma
            left_fc_threshold = shift - 3*sigma
            logger.info('Dynamic fold change threshold is applied for {}: {} {}'.format(suf, left_fc_threshold, right_fc_threshold, ))
            table = table[table['pval_adj_pass']][['Protein', 'P(PECA)', 'log2_FC']]
            table = table.query('`log2_FC` >= @right_fc_threshold or `log2_FC` <= @left_fc_threshold')
        comp_df = comp_df.merge(table, how='outer', on='Protein', suffixes = (None, '_'+suf))
    comp_df.rename(columns={'log2_FC': 'log2_FC_'+suffixes[0], 'P(PECA)': 'P(PECA)_'+suffixes[0] }, inplace=True )
    comp_df.dropna(how = 'all', subset=['log2_FC_' + suf for suf in suffixes], inplace=True)
    # comp_df.loc['Total number', :] = df0.notna().sum(axis=0)
    
    total_de_prots = {}
    for suf in suffixes :
        total_de_prots[suf] = comp_df['log2_FC_'+suf].notna().sum()
    
    if save :
        comp_df.to_csv(os.path.join(out_folder, 'iqmma_results.tsv'), 
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
        v = venn(dataset_dict, cmap=plt.get_cmap('Dark2'), fontsize=28, alpha = 0.5, ax=ax)
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
        logger.info('Venn diagram created')
        
    return total_de_prots


def diffacto_call(diffacto_path='',
                  out_path='',
                  peptide_path='',
                  sample_path='',
                  min_samples=3,
                  psm_dfs_dict={},
                  samples_dict={},
                  write_peptides=False,
                  str_of_other_args='', 
                  logger = logging.getLogger('function')
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
        logger.critical('Existing path to diffacto file is required')
        return -1
    
    if write_peptides :
        logger.info('Diffacto writing peptide file START')
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
        logger.info('DONE')
    else :
        logger.info('Diffacto is using existing peptide file')
    
    logger.info('Diffacto writing sample file START')
    out = open( sample_path , 'w')
    for sample_num in sample_nums :
        for sample in samples_dict[sample_num] :
            label = sample
            out.write(label + '\t' + sample_num + '\n')
            logger.info(label + '\t' + sample_num)
    out.close()
    logger.info('DONE')
    
    other_args = [x.strip() for x in str_of_other_args.strip("'").strip('"').split(' ')]
    final_args = [diffacto_path, 
                  '-i', peptide_path, 
                  '-out', out_path, 
                  '-samples', sample_path, ] + ['-min_samples', min_samples] + other_args
    final_args = list(filter(lambda x: False if x=='' else True, final_args))
    final_args = [str(x) for x in final_args]
    logger.info('Diffacto START')
    logger.debug(final_args)
    process = subprocess.Popen(final_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout:
        log_subprocess_output(process.stdout, logger=logger)
    exitscore = process.wait()
    logger.debug(exitscore)
    
    if exitscore == 0 :
        logger.info('DONE')
    else :
        logger.critical('Diffacto error: ', exitscore)
    
    return exitscore


# gets dict with PATHS to (peptide-intensity) files, merge it to one big table, mix intensities 
# and returns dict with peptide-intensity tables
# optionally write it as separate and/or diffacto-peptides-like files
def mix_intensity(input_dict,
                  samples_dict,
                  choice=4,
                  suf_dict={'dino':'D', 'bio2':'B2', 'openMS':'O', 'mixed':'M'}, 
                  out_dir=None,
                  default_order=None,
                  to_diffacto=None, 
                  logger = logging.getLogger('function')
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
    
    logger.info('Merging START')
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

    logger.info('Merging DONE')
    
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
        logger.critical('Invalid value for choice parameter: %s', choice)
        raise ValueError('Invalid value for choice parameter: %s', choice)
        return -2
    
    logger.info('Choosing intensities START')
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
    
    logger.info('Choosing intensities DONE')
    if to_diffacto :
        cols = ['peptide', 'protein'] + [sample for sample in samples]
        merge_df.to_csv(to_diffacto, 
                        sep=',', 
                        index = False,
                        columns = cols, 
                        encoding = 'utf-8')
    
    logger.info('Writing peptides files for Diffacto Mixed DONE')
    if out_dir :
        merge_df.to_csv(os.path.join(out_dir, 'mixing_all.tsv'), 
                        sep='\t', 
                        index = list(merge_df.index), 
                        columns = list(merge_df.columns), 
                        encoding = 'utf-8')
        
    logger.info('Mixing intensities DONE')
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
                                       out_path=None, 
                                       logger = logging.getLogger('function')
                                      ) :
    psm_df = pd.read_table(path, sep='\t')
    
    cols = list(psm_df.columns)
    needed_cols = ['peptide', 'protein', 'assumed_charge', intens_colomn_name]
    for col in needed_cols :
        if col not in cols :
            logger.critical('Not all needed columns are in file: '+col+' not exists')
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
        logger.critical('Invalid value for method: %s', method)
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


def read_PSMs(infile_path, usecols=None, logger=logging.getLogger('function')) :
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

    if 'protein_descr' in df1.columns:

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

    if len(df1['spectrum']) != len(set(df1['spectrum'])):
        logger.warning('\nWARNING! Spectrum column values are not unique. Spectrum column values were modified to fix it!\n')
        df1['spectrum'] = df1['spectrum'] + '_' + np.arange(len(df1)).astype(str)
    
    return df1


## Функции для сопоставления

    
def noisygaus(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def opt_bin(ar, border=16, logger = logging.getLogger('function')) :
    num_bins = 4
    bwidth = (max(ar) - min(ar))/num_bins
    bbins = np.arange(min(ar), max(ar), bwidth)
    H1, b1 = np.histogram(ar, bins=bbins)
    max_percent = 100*max(H1)/sum(H1)

    bestbins1 = num_bins
    bestbins2 = num_bins
    mxp2 = max_percent
    mxp1 = max_percent
    i = 0
    while max_percent > border and i < 10000 :
        num_bins = num_bins*2

        bwidth = (max(ar) - min(ar))/num_bins
        bbins = np.arange(min(ar), max(ar), bwidth)
        H1, b1 = np.histogram(ar, bins=bbins)
        max_percent = 100*max(H1)/sum(H1)
        if max_percent < border :
            bestbins1 = num_bins
            mxp1 = max_percent
        i += 1
    i = 0
    while max_percent < border and i < 10000 :
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
        i += 1
    if abs(mxp1 - border) < abs(mxp2 - border) :
        bestbins = bestbins1
    else :
        bestbins = bestbins2
    bwidth = (max(ar) - min(ar))/bestbins    
    bbins = np.arange(min(ar), max(ar), bwidth)
    H1, b1 = np.histogram(ar, bins=bbins)
    max_percent = 100*max(H1)/sum(H1)
    logger.debug('final num_bins: ' + str(int(num_bins)) + '\t' + 'final max percent per bin: ' + str(round(max_percent, 2)) + '%')

    return bwidth



def calibrate_mass(mass_left, mass_right, true_md, check_gauss=False, logger = logging.getLogger('function')) :
    
    bwidth = opt_bin(true_md, logger=logger)
    bbins = np.arange(mass_left, mass_right, bwidth)
    H1, b1 = np.histogram(true_md, bins=bbins)
    noise_fraction = max(1, np.median(H1)) * len(H1) / H1.sum()

    H_marg = np.median(H1)
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
#        logger.debug('Интервал значений ' + str(b1[ll]-bwidth) + ' ' + str(b1[rr]))
    for el in true_md :
        if el >= b1[i]-bwidth*(i-j) and el <= b1[i]+bwidth*(k-i) :
            t.append(el)

    bwidth = opt_bin(t, border=min(8,8*0.5/noise_fraction), logger=logger)
    bbins = np.arange(min(t), max(t) , bwidth)
    H2, b2 = np.histogram(t, bins=bbins)
    # pickle.dump(t, open('/home/leyla/project1/mbr/t.pickle', 'wb'))
    m = max(H2)
    mi = b2[np.argmax(H2)]
    s = (max(t) - min(t))/6
    noise = np.median(H2)

    popt, pcov = curve_fit(noisygaus, b2[1:], H2, p0=[m, mi, s, noise])
    # popt, pcov = curve_fit(noisygaus, b2[1:], H2, p0=[m, mi, s, noise])
    logger.debug(popt)
    mass_shift, mass_sigma = popt[1], abs(popt[2])

    if check_gauss:
        logger.debug('GAUSS FIT, %f, %f' % (percentileofscore(t, mass_shift - 3 * mass_sigma), percentileofscore(t, mass_shift + 3 * mass_sigma)))

        if percentileofscore(t, mass_shift - 3 * mass_sigma) + 100 - percentileofscore(t, mass_shift + 3 * mass_sigma) > 10:
            mass_sigma = scoreatpercentile(np.abs(t-mass_shift), 95) / 2
            
    logger.debug('shift: ' + str(mass_shift) + '\t' + 'sigma: ' + str(mass_sigma))
    return mass_shift, mass_sigma, pcov[0][0]


def total(df_features, psms, mean1=0, sigma1=False, mean2 = 0, sigma2=False, mean_mz=0, mass_accuracy_ppm=10, mean_im = 0, sigma_im = False, isotopes_array=[0, ], logger = logging.getLogger('function')):

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
                    logger.info('there is no column "IM" in the PSMs')
            if check_FAIMS:
                if 'compensation_voltage' in row:
                    psm_FAIMS = row['compensation_voltage']
                else:
                    check_FAIMS = False
                    logger.info('there is no column "FAIMS" in the PSMs')
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


def found_mean_sigma(df_features,psms,parameters, sort ='mz_diff_ppm' , mean1=0,sigma1=False,mean2=0,sigma2=False, mean_mz = 0, sigma_mz = False, logger = logging.getLogger('function')):
# sort ='mz_diff_ppm'
    check_gauss = False
    rtStart_array_ms1 = df_features['rtStart'].values
    rtEnd_array_ms1 = df_features['rtEnd'].values

    if parameters == 'rt1' or parameters == 'rt2':
        results_psms = total(df_features = df_features,psms = psms,mass_accuracy_ppm = 100, logger=logger)

    if parameters == 'mz_diff_ppm' :
        check_gauss = True
        results_psms = total(df_features =df_features,psms =psms,mean1 = mean1,sigma1 = sigma1, mean2 = mean2,sigma2 = sigma2,mass_accuracy_ppm = 100, logger=logger)

    if parameters == 'im_diff':
        results_psms = total(df_features =df_features,psms =psms,mean1 = mean1,sigma1 = sigma1, mean2 = mean2,sigma2 = sigma2, mean_mz = mean_mz, mass_accuracy_ppm = 3*sigma_mz, logger=logger)

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
        mean, sigma,_ = calibrate_mass(min(ar),max(ar),ar, check_gauss, logger=logger)
    else:
        mean, sigma,_ = calibrate_mass(min(ar),max(ar),ar, check_gauss, logger=logger)
    return mean, sigma


def optimized_search_with_isotope_error_(df_features,psms,mean_rt1=False,sigma_rt1=False,mean_rt2=False,sigma_rt2=False,mean_mz = False,sigma_mz = False,mean_im = False,sigma_im = False, isotopes_array=[0,1,-1,2,-2], logger = logging.getLogger('function')):
    
    idx = {}
    for j, i in enumerate(isotopes_array):
        idx[i] = j
    
    if mean_rt1 is False and sigma_rt1 is False:
        logger.debug('rt1')
        mean_rt1, sigma_rt1 = found_mean_sigma(df_features,psms, 'rt1', logger=logger)

    if mean_rt2 is False and sigma_rt2 is False:
        logger.debug('rt2')
        mean_rt2, sigma_rt2 = found_mean_sigma(df_features,psms, 'rt2', logger=logger)

    if mean_mz is False and sigma_mz is False:
        logger.debug('mz')
        mean_mz, sigma_mz = found_mean_sigma(df_features,psms,'mz_diff_ppm', mean1 = mean_rt1, sigma1 = sigma_rt1, mean2 = mean_rt2,sigma2 = sigma_rt2, logger=logger)   

    if mean_im is False and sigma_im is False:
        logger.debug('im')
        mean_im, sigma_im = found_mean_sigma(df_features,psms,'im_diff', mean1 = mean_rt1, sigma1 = sigma_rt1, mean2 = mean_rt2,sigma2 = sigma_rt2, mean_mz = mean_mz, sigma_mz= sigma_mz, logger=logger)  

    # print(mean_rt1, sigma_rt1,mean_rt2, sigma_rt2,mean_mz, sigma_mz )    

    results_isotope = total(df_features = df_features,psms =psms,mean1 = mean_rt1, sigma1 = sigma_rt1,mean2 = mean_rt2, sigma2 = sigma_rt2, mean_mz = mean_mz, mass_accuracy_ppm = 3*sigma_mz, isotopes_array=isotopes_array, logger=logger)
    
    results_isotope_end = []
    cnt = Counter([z[0]['i'] for z in results_isotope.values()])
    for i in cnt.values():
        results_isotope_end.append(i/len(psms))
    end_isotope_ = list(np.add.accumulate(np.array(results_isotope_end))*100)
    logger.info(end_isotope_)
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

def mbr(feat,II,PSMs_full_paths, PSM_path, logger=logging.getLogger('function')):
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
            III = optimized_search_with_isotope_error_(feat, psm_j_fdr, isotopes_array=[0,1,-1,2,-2], logger=logger)[0]
            III = III.sort_values(by = 'feature_intensityApex', ascending = False)
            III = III.drop_duplicates(subset = 'pep_charge')
            II = pd.concat([II, III])
    return II


def log_subprocess_output(pipe, logger = logging.getLogger('function') ):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logger.info('From subprocess: %r', line)
        