import pandas as pd
from pyteomics.openms import featurexml
import os
import errno
import re
import logging

from .utils import call_Dinosaur, call_Biosaur2, call_OpenMS, generate_users_output, diffacto_call, mix_intensity, charge_states_intensity_processing, read_PSMs, optimized_search_with_isotope_error_, mbr


class WrongInputError(NotImplementedError):
    pass


class EmptyFileError(ValueError):
    pass


logger = logging.getLogger(__name__)


def process_files(args):

    if args['threads'] :
        if args['bio2'] :
            if not 'nprocs' in args['bio2_args'] :
                args['bio2_args'] = args['bio2_args'].strip('"'"'") + ' -nprocs ' + str(args['threads'])
                logger.debug('-bio2_args extended with -nprocs %s', args['threads'])
            else :
                logger.debug('-bio2_args already contains -nprocs option, iqmma -threads %s is ignored', args['threads'])
        if args['openMS'] :
            if not 'threads' in args['openms_args'] :
                args['openms_args'] = args['openms_args'].strip('"'"'") + ' -threads ' + str(args['threads'])
                logger.debug('-openms_args extended with -threads %s', args['threads'])
            else :
                logger.debug('-openms_args already contains -threads option, iqmma -threads %s is ignored', args['threads'])
    logger.debug(args)

    if not args['s1'] :
        logger.critical('At least one file in the argument is needed: -s1 %s', args['s1'])
        raise ValueError('At least one file in the argument is needed: -s1 {}'.format(args['s1']))

    mode = None
    if not args['s2'] :
        sample_nums = ['s1']
        logger.info('mode = feature matching')
        mode = 'feature matching'
    else :
        sample_nums = ['s1', 's2']
        logger.info('mode = diffacto')
        mode = 'diffacto'

    for sample_num in sample_nums :
        if args[sample_num] :
            if type(args[sample_num]) is str :
                args[sample_num] = [os.path.abspath(os.path.normpath(x.strip()+'.mzML')) for x in re.split(r'\.mzML|\.mzml|\.MZML' , args[sample_num], )][:-1]
            elif type(args[sample_num]) is list :
                args[sample_num] = [os.path.abspath(os.path.normpath(x)) for x in args[sample_num]]
            else :
                logger.critical('invalid %s input', sample_num)
                return -1

#    print(args['s1'].split())

    if mode == 'diffacto' :
        if not args['dif'] :
            logger.critical('Path to diffacto executable file is required')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args['dif'])
        elif not os.path.exists(os.path.normpath(args['dif'])) :
            logger.critical('Path to diffacto executable file does not exist: %s', os.path.normpath(args['dif']))
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args['dif'])
        else :
            args['dif'] = os.path.abspath(os.path.normpath(args['dif']))

        if set(args[sample_nums[0]]) & set(args[sample_nums[1]]) :
            logger.critical('Identical files added for both samples (s1 and s2). Statistical test cannot be performed.')
            raise ValueError(errno.ENOENT, os.strerror(errno.ENOENT), set(args[sample_nums[0]]) & set(args[sample_nums[1]]))

#     if not args['scav2dif'] :
#         logger.warning('path to scav2diffacto.py file is required')
#         return -1

    arg_suff = ['dino', 'bio2', 'openMS']
    suffixes = []
    for suf in arg_suff :
        if args[suf] :
            suffixes.append(suf)

    k = len(suffixes)
    if k == 0 :
        logger.critical('At least one feature detector shoud be given!')
        return -1
    elif k == 1 and mode == 'diffacto' :
        logger.info('First diffacto run applied')
        args['venn'] = 0
    elif k >= 2 and mode == 'diffacto' :
        logger.info('Second diffacto run applied')
    else :
        logger.info('No diffacto run applied')

    samples = []
    samples_dict = {}
    mzML_paths = []
    for sample_num in sample_nums :
        samples_dict[sample_num] = []
        for z in args[sample_num]:
            mzML_paths.append(z)
            samples.append(os.path.basename(z).replace('.mzML', ''))
            samples_dict[sample_num].append(os.path.basename(z).replace('.mzML', ''))

    if args['min_samples'] == -1 :
        args['min_samples'] = int(len(samples)/2)
    if args['min_samples'] == 0 :
        args['min_samples'] = min([len(samples_dict[x]) for x in sample_nums])

    logger.debug('samples_dict = ' + str(samples_dict))

    PSMs_full_paths = []
    PSMs_full_dict = {}
    PSMs_suf = args['psm_format']
    if args['psm_folder'] :
        dir_name = os.path.abspath(os.path.normpath(args['psm_folder']))
        if os.path.exists(dir_name) :
            logger.info('Searching *%s files in %s', PSMs_suf, dir_name)
    else :
        logger.warning('Searching *%s files in the same directory as .mzML', PSMs_suf)
        dir_name = os.path.dirname(mzML_paths[0])
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
                logger.critical('sample ' + sample + ' PSM file not found')
                return -1
    logger.debug(PSMs_full_dict)

    if not args['outdir'] :
        logger.info('Path to output directory is not specified. Using input files directory instead.')
        args['outdir'] = os.path.dirname(mzML_paths[0])
    else :
        args['outdir'] = os.path.abspath(os.path.normpath(args['outdir']))
        logger.info('Results are stored at %s', args['outdir'])

    mzML_dict = {}
    for sample, mzML in zip(samples, mzML_paths) :
        mzML_dict[sample] = mzML

    if mode != 'feature matching' :
        if args['allowed_pepts'] :
            if os.path.exists(os.path.normpath(args['allowed_pepts'])) :
                logger.info('Peptides from file %s are allowed for quantitative analysis', os.path.normpath(args['allowed_pepts']))
                peptides_path = os.path.normpath(args['allowed_pepts'])
            else :
                logger.warning('Path to peptides files folder does not exist: %s', os.path.normpath(args['allowed_pepts']))
                peptides_path = ''
        else :
            peptides_path = ''

        if args['allowed_prots'] :
            if os.path.exists(os.path.normpath(args['allowed_prots'])) :
                logger.info('Proteins from file %s are allowed for quantitative analysis', os.path.normpath(args['allowed_prots']))
                proteins_path = os.path.normpath(args['allowed_prots'])
            else :
                logger.warning('path to proteins files folder does not exist: %s', os.path.normpath(args['allowed_prots']))
                proteins_path = ''
        else :
            proteins_path = ''

    paths = {'mzML': mzML_dict,
             'PSM_full' : PSMs_full_dict,
            }
    if mode != 'feature matching' :
        paths['peptides'] = peptides_path
        paths['proteins'] = proteins_path
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
    #     logger.critical('Invalid value for setting: -dynamic_fc_threshold %s', args['dynamic_fc_threshold'])
    #     raise ValueError('Invalid value for setting: -dynamic_fc_threshold %s', args['dynamic_fc_threshold'])



    logger.debug('PSMs_full_paths: %s', PSMs_full_paths)
    logger.debug('mzML_paths: %s', mzML_paths)
    logger.debug('out_directory: %s', out_directory)
    logger.debug('suffixes: %s', suffixes)
    logger.debug('sample_1: %s', sample_1)
    logger.debug('sample_2: %s', sample_2)
    logger.debug('mixed = %s', args['mixed'])
    logger.debug('venn = %s', args['venn'])
    logger.debug('choice = %s', args['choice'])
    logger.debug('overwrite_features = %s', args['overwrite_features'])
    logger.debug('overwrite_first_diffacto = %s', args['overwrite_first_diffacto'])
    logger.debug('overwrite_matching = %d', args['overwrite_matching'])

    os.makedirs(out_directory, exist_ok=True)

## Генерация фич
    if args['feature_folder'] :
        if not os.path.exists(os.path.normpath(args['feature_folder'])) :
            logger.warning('Path to feature files folder does not exist. Creating it.')
        feature_path = os.path.abspath(os.path.normpath(args['feature_folder']))
    else :
        if args['psm_folder'] :
            dir_name = os.path.abspath(os.path.normpath(args['psm_folder']))
        else :
            dir_name = os.path.dirname(PSMs_full_paths[0])
        feature_path = os.path.join(dir_name,  'features')
    os.makedirs(feature_path, exist_ok=True)


### Dinosaur

# На выходе добавляет в папку feature_path файлы *sample*_features_dino.tsv

    if args['dino'] :
        if os.path.exists(os.path.normpath(args['dino'])) :
            args['dino'] = os.path.abspath(os.path.normpath(args['dino']))
            for path, sample in zip(mzML_paths, samples) :
                outName = sample + '_features_' + 'dino' + '.tsv'
                if args['overwrite_features'] == 1 or not os.path.exists(os.path.join(feature_path, outName)) :
                    logger.info('Writing features' + ' dino ' + sample)
                    exitscore = call_Dinosaur(args['dino'], path, feature_path, outName, args['dino_args'])
                    logger.debug(exitscore)
                else :
                    logger.info('Not overwriting features ' + ' dino ' + sample)
        else :
            logger.error('Skipping Dinosaur. Path to Dinosaur does not exists: %s', args['dino'])

### Biosaur2

# На выходе добавляет в папку out_directory/features файлы *sample*_features_bio2.tsv
# Важно: опция -hvf 1000 (без нее результаты хуже)

    if args['bio2'] :
        if os.path.exists(os.path.normpath(args['bio2'])) :
            args['bio2'] = os.path.abspath(os.path.normpath(args['bio2']))
            for path, sample in zip(mzML_paths, samples) :
                outPath = os.path.join(feature_path, sample + '_features_bio2.tsv')
                if args['overwrite_features'] == 1 or not os.path.exists(outPath) :
                    logger.info('Writing features ' + ' bio2 ' + sample)
                    exitscore = call_Biosaur2(args['bio2'], path, outPath, args['bio2_args'])
                    logger.debug(exitscore)
                else :
                    logger.info('Not overwriting features ' + ' bio2 ' + sample)
        else :
            logger.error('Skipping Biosaur2. Path to Biosaur2 does not exists: %s', args['bio2'])

### OpenMS

# На выходе создает в feature_path папку OpenMS с файлами *.featureXML и добавляет в папку out_directory/features файлы *sample*_features_openMS.tsv

    if args['openMS'] :
        if os.path.exists(os.path.normpath(args['openMS'])) :
            out_path_dir = os.path.join(feature_path, 'openMS')
            try :
                os.makedirs(out_path_dir, exist_ok=True)
            except :
                out_path_dir = feature_path
            for path, sample in zip(mzML_paths, samples) :
                o = os.path.join(feature_path, sample + '_features_' + 'openMS.tsv')
                out_path = os.path.join(out_path_dir, sample + '.featureXML')
                if args['overwrite_features'] == 1 or (not os.path.exists(out_path) and not os.path.exists(o)) :
                    logger.info('Writing .featureXML ' + ' openMS ' + sample)
                    exitscore = call_OpenMS(args['openMS'], path, out_path, args['openms_args'])
                    logger.debug(exitscore)
                else :
                    logger.info('Not ovetwriting .featureXML ' + ' openMS ' + sample)

            for path, sample in zip(mzML_paths, samples) :
                out_path = os.path.join(out_path_dir, sample + '.featureXML')
                o = os.path.join(feature_path, sample + '_features_' + 'openMS.tsv')
                if args['overwrite_features'] == 1 or not os.path.exists(o) :
                    logger.info('Writing features ' + ' openMS ' + sample)
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
                    logger.info('Not overwriting features ' + ' openMS ' + sample)
        else :
            logger.error('Skipping OpenMS. Path to OpenMSFeatureFinderCentroided does not exists: %s', args['openMS'])


### Сопоставление

    if args['matching_folder'] :
        if os.path.exists(args['matching_folder']) :
            matching_path = os.path.abspath(os.path.normpath(args['matching_folder']))
        else :
            matching_path = os.path.abspath(os.path.normpath(args['matching_folder']))
            logger.warning('Path to matching folder does not exists, creating it: %s', matching_path)
    else :
        matching_path = os.path.join(out_directory, 'feats_matched')
        logger.warning('Path to matching folder does not exists, using default one: %s', matching_path)
    os.makedirs(matching_path, exist_ok=True)

    logger.info('Start matching features')
    for PSM_path, sample in zip(PSMs_full_paths, samples) :
        PSM = read_PSMs(PSM_path, modified_seq=args['modified_seq'])
        logger.info('sample %s', sample)
        for suf in suffixes :
            if args['overwrite_matching'] == 1 or not os.path.exists(os.path.join(matching_path, sample + '_' + suf + '.tsv')) :
                feats = pd.read_csv( os.path.join(feature_path, sample + '_features_' + suf + '.tsv'), sep = '\t')
                feats = feats.sort_values(by='mz')

                logger.info(suf + ' features ' + sample + 'START')
                temp_df = optimized_search_with_isotope_error_(feats, PSM, isotopes_array=args['isotopes'])[0]
                # temp_df = optimized_search_with_isotope_error_(feats, PSM, mean_rt1=0,sigma_rt1=1e-6,mean_rt2=0,sigma_rt2=1e-6,mean_mz = False,sigma_mz = False,mean_im = False,sigma_im = False, isotopes_array=[0,1,-1,2,-2])[0]

                if args['mbr']:
                    logger.info('Start match-between-runs for features %s %s', sample, suf)
                    temp_df = mbr(feats, temp_df, PSMs_full_paths, PSM_path)

                median = temp_df['feature_intensityApex'].median()
                temp_df['med_norm_feature_intensityApex'] = temp_df['feature_intensityApex']/median
                temp_df['min_norm_feature_intensityApex'] = temp_df['feature_intensityApex']/temp_df['feature_intensityApex'].min()
                temp_df['max_norm_feature_intensityApex'] = temp_df['feature_intensityApex']/temp_df['feature_intensityApex'].max()
                cols = list(temp_df.columns)

                logger.info(suf + ' features ' + sample + ' DONE')
                temp_df.to_csv(os.path.join(matching_path, sample + '_' + suf + '.tsv'), sep='\t', columns=cols)
                logger.info(sample + ' PSMs matched ' + str(temp_df['feature_intensityApex'].notna().sum()) + '/'
                             + str(len(temp_df)) + ' ' + str(round(temp_df['feature_intensityApex'].notna().sum()/len(temp_df)*100, 2)) + '%')
                logger.info(suf + ' MATCHED')

        temp_df = None

    paths['feats_matched'] = {}
    for sample in samples :
        feats_matched_paths = {}
        for suf in suffixes :
            s = os.path.join(matching_path, sample + '_' + suf + '.tsv')
            if os.path.exists(s):
                feats_matched_paths[suf] = s
            else :
                logger.critical('File not found: %s', s)
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), s)
                return -1
        paths['feats_matched'][sample] = feats_matched_paths
    logger.info('Matching features for PSMs done')


    ## Подготовка и прогон Диффакто 1

    if mode == 'diffacto' :
        logger.info('Going for quantitative analysis with diffacto')
        if args['diffacto_folder'] :
            diffacto_folder = os.path.abspath(os.path.normpath(args['diffacto_folder']))
            if not os.path.exists(diffacto_folder):
                logger.warning('Path to diffacto results does not exist, creating: %s', diffacto_folder)
        else :
            diffacto_folder = os.path.join(out_directory, 'diffacto')
        os.makedirs(diffacto_folder, exist_ok=True)

        intens_colomn_name = 'feature_intensityApex'
        if args['norm'] == 3 :
            intens_colomn_name = 'max_norm_feature_intensityApex'
        elif args['norm'] == 2 :
            intens_colomn_name = 'min_norm_feature_intensityApex'
        elif args['norm'] == 1 :
            intens_colomn_name = 'med_norm_feature_intensityApex'
        elif args['norm'] == 0 :
            intens_colomn_name = 'feature_intensityApex'

        allowed_prots = set()
        if paths['proteins'] :
            df0 = pd.read_table(paths['proteins'], usecols=['dbname'])
            allowed_prots.update(df0['dbname'])
            logger.info('allowed proteins total: %d', len(allowed_prots))
        else :
            logger.info('allowed proteins total: all')

        allowed_peptides = set()
        if paths['peptides'] :
            logger.info('Allowed peptides for quantitation from file: %s', paths['peptides'][key])
            df0 = pd.read_table(paths['peptides'], usecols=['peptide'])
            allowed_peptides.update(df0['peptide'])
            allowed_pept_modif = False
            logger.info('allowed peptides total: %d', len(allowed_peptides))
        else :
            q_counter = 0
            temp_s = set()
            for psm_path in PSMs_full_paths :
                logger.debug('Reading PSM file %s', psm_path)
                t = read_PSMs(psm_path, modified_seq=args['modified_seq'])
                if 'q' in list(t.columns) :
                    logger.info('Using q-value < 0.01 filtering on input peptides for file: %s', psm_path)
                    temp_s.update(t[(t['q'] < 0.01) & (t['protein'].str.find(args['decoy_prefix']) < 0)]['peptide'])
                    q_counter += 1
                else :
                    temp_s.update(t['peptide'])
            allowed_peptides.update(temp_s)
            if q_counter > 0 :
                logger.info('Using Scaveger input files, peptide modifications allowed')
                allowed_pept_modif = True
            else :
                logger.info('No filtering on the input peptides applied, all accepted for quantitation')
                allowed_pept_modif = False
            logger.info('Allowed peptides total: %d', len(allowed_peptides))

        paths['DiffPept'] = {}
        paths['DiffSampl'] = {}
        paths['DiffOut'] = {}
        for suf in suffixes:
            logger.info('Charge states processing %s', suf)
            psms_dict = {}
            for sample in samples:
                logger.debug('Starting %s', sample)
                if args['logs'].upper() == 'DEBUG' :
                    charge_faims_intensity_path = os.path.join(diffacto_folder, 'charge_faims_intensity')
                    os.makedirs(charge_faims_intensity_path, exist_ok=True)
                    out_path = os.path.join(diffacto_folder, 'charge_faims_intensity', sample+'_'+suf+'.tsv')
                else :
                    out_path = None
                    charge_faims_intensity_path = None
                psms_dict[sample] = charge_states_intensity_processing(paths['feats_matched'][sample][suf],
                                                                        method=args['pept_intens'],
                                                                        intens_colomn_name=intens_colomn_name,
                                                                        allowed_peptides=allowed_peptides, # set()
                                                                        allowed_pept_modif=allowed_pept_modif,
                                                                        allowed_prots=allowed_prots, # set()
                                                                        out_path=out_path,
                )
                logger.debug('Done %s', sample)

            paths['DiffPept'][suf] = os.path.join(diffacto_folder, args['outpept'].replace('.txt', '_' + suf + '.txt'))
            paths['DiffSampl'][suf] = os.path.join(diffacto_folder, args['outsampl'].replace('.txt', '_' + suf + '.txt'))
            paths['DiffOut'][suf] = os.path.join(diffacto_folder, args['outdiff'].replace('.txt', '_' + suf + '.txt'))
            if args['overwrite_first_diffacto'] == 1 or not os.path.exists( paths['DiffOut'][suf] ) :
                exitscore = diffacto_call(diffacto_path = args['dif'],
                                          out_path = paths['DiffOut'][suf],
                                          peptide_path = paths['DiffPept'][suf],
                                          sample_path = paths['DiffSampl'][suf],
                                          min_samples = args['min_samples'],
                                          psm_dfs_dict = psms_dict,
                                          samples_dict = samples_dict,
                                          write_peptides=True,
                                          str_of_other_args = args['diffacto_args'],
                                         )
            logger.info('Done Diffacto run with %s', suf)
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
            pval_adj=args['pval_adj'],
            fc_threshold = args['fc_threshold'],
            dynamic_fc_threshold = args['dynamic_fc_threshold'],
            save = save,
        )

        if sum(list(num_changed_prots.values())) == 0 :
            logger.warning('No differentially expressed proteins detected in separate diffacto runs')

        if args['choice'] == 0 :
            default_order = sorted(suffixes, key= lambda x: num_changed_prots[x], reverse=True)
        else :
            default_order = False


        ## Второй прогон диффакто


        if k >= 2 and args['mixed'] == 1 :
            suf = 'mixed'
            logger.info('Mixing intensities STARTED')

            to_diffacto = os.path.join(diffacto_folder, args['outpept'].replace('.txt', '_' + suf + '.txt'))
            a = mix_intensity(paths['DiffPept'],
                              samples_dict,
                              choice=args['choice'],
                              suf_dict={'dino':'D', 'bio2':'B2', 'openMS':'O', 'mixed':'M'},
                              out_dir= charge_faims_intensity_path,
                              default_order= default_order,
                              to_diffacto= to_diffacto,
                              )

            logger.info('Mixing intensities DONE with exitscore %s', a)

            paths['DiffPept'][suf] = to_diffacto
            paths['DiffSampl'][suf] = os.path.join(diffacto_folder, args['outsampl'].replace('.txt', '_' + suf + '.txt'))
            paths['DiffOut'][suf] = os.path.join(diffacto_folder, args['outdiff'].replace('.txt', '_' + suf + '.txt'))

            logger.info('Diffacto START')
            exitscore = diffacto_call(diffacto_path = args['dif'],
                                      out_path = paths['DiffOut'][suf],
                                      peptide_path = paths['DiffPept'][suf],
                                      sample_path = paths['DiffSampl'][suf],
                                      min_samples = args['min_samples'],
                                      psm_dfs_dict = {},
                                      samples_dict = samples_dict,
                                      str_of_other_args = args['diffacto_args'],
                                     )
            logger.info('Done Diffacto run with %s with exitscore %s', suf, exitscore)

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
                pval_adj=args['pval_adj'],
                fc_threshold = args['fc_threshold'],
                dynamic_fc_threshold = args['dynamic_fc_threshold'],
                save = save,
            )
            logger.info('Numbers of differentially expressed proteins:')
            for suf in suffixes+['mixed'] :
                logger.info('%s: %s', suf, num_changed_prots[suf])

            logger.info('IQMMA finished')
