
`iqmma` consumes LC-MS data (**mzML**) and results of post-search analysis (or just peptide identifications) (**tsv**) and performs multiple tool (Dinosaur, Biosaur and OpenMS FeatureFinderCentroided) feature detection, peptide-intensity matching and quantitation.

## Installation

Using pip:

    pip install iqmma

It will install, if necessary, the minimum set (Biosaur2 and Diffacto) to `iqmma` to function. However, additional feature detections will need to be installed manually according to their manuals (see links below).

## Usage

### Some explanation and requirements

`iqmma` has two working mods. First of all, it can be quantitation workflow with generating peptide features using multiple tools, matching them on peptides, and two Diffacto quantitation stages (separated and mixed, where the algorithm is choosing the best intensities for each peptide between different feature detections). The second one is stopping after peptide-feature matching to allow user to apply any other quantitation approach on matched intensities.

For `iqmma` to work properly, each mzML file must have a related PSM file which name starts with the name of mzML.

For basic usage all PSMs and mzML files should be stored in the same directory, otherwise -PSM_folder parameter must be applied. All PSMs files must be `*PSM_full.tsv` tables obtained from the Scavager output (https://github.com/markmipt/scavager).

### Quantitation mode

Basic command for quantitation mode: 

    iqmma -bio2 path_to_Biosaur2 -dino path_to_Dinosaur -openms path_to_openMS -dif path_to_Diffacto -s1 paths_to_mzml_files_from_sample_1_*.mzML -s2 paths_to_mzml_files_from_sample_2_*.mzML -outdir out_dir

Note 1: -s2 argument is necessary for quantitation mode to activate.

Note 2: at least two feature detections should be given for Mix algorithm to work.

### Feature matching mode

Basic command for matching peptide intensities: 

    iqmma -bio2 path_to_Biosaur2 -dino path_to_Dinosaur -openms path_to_openMS -dif path_to_Diffacto -s1 paths_to_all_mzml_files_*.mzML -outdir out_path

Note: all mzml files go into `-s1` - the first sample option - without any differences between them, no quantitation applied.

### Advanced example

Full quantitation mode (Linux-based example):

    iqmma -bio2 /usr/bin/biosaur2 -dino /home/user/downloads/Dinosaur-1.2.0.free.jar -dif /usr/bin/diffacto -s1 /home/user/downloads/sample1_rep1.mzml /home/user/downloads/sample1_rep2.mzml -s2 /home/user/downloads/sample2_rep1.mzml /home/user/downloads/sample2_rep2.mzml -logs INFO -log_path /home/user/iqmma_logs/logs_N.log -mbr 1 -overwrite_matching 1 -mixed 1 -fc_threshold 2.5 -pval_threshold 0.01
    
Here, two samples with two replicas per sample compared against each other in quantitation mode. `-psm_folder` and `-psm_format` are not specified so `iqmma` will search peptide identifications in folder `/home/user/downloads` (near .mzml default for `-psm_folder`) by searching files which names start for example with `sample1_rep1` (and other names of .mzml files) and end with `PSMs_full.tsv` (default value of `-psm_format`, for Scavager results). More than one feature detection is available so by default mixed algorithm would also be turned on (`-mixed 1` is default). `-mbr 1` turns on match between runs, so matching needs to be overwritten not to use old files without matching between runs. `-logs` and `-log_path` specify level of logging messages and where to store them. `-fc_threshold 2.5` and `-pval_threshold 0.01` specifies thresholds on Fold Change and p-value to apply to differentially expressed proteins in final filtering.

Matching mode (Anaconda, Windows paths):
    
    iqmma -dino c:\user\downloads\dinosaur-1.2.0.free.jar -bio2 c:\user\anaconda3\scripts\biosaur2.exe -dif c:\user\Anaconda3\Scripts\diffacto.exe -s1 c:\user\downloads\sample1_rep1.mzml c:\user\downloads\sample2_rep1.mzml -outdir c:\user\iqmma_analysis\out_1 -logs info -log_path  c:\user\iqmma_analysis\out_1\logs.log -psm_folder c:\user\downloads\mzid_peptides -psm_format .mzid 

Here there are two samples in one replica each to match on peptides identifications that are stored in files `-psm_folder` + `\` + (.mzml filename) + `-psm_format` which results in `c:\user\downloads\mzid_peptides\` + `sample1_rep1` + `.mzid`. Two feature detections are given (paths to executable files are given), so there would be two rows of matched files in the `-outdir` in the end: Dinosaur-generated features matched (ends with `_dino.tsv`) on peptides and Biosaur2-generated (ends with `_bio2.tsv`) features matched on peptides.

Note 1: Paths to feature detections or Diffacto should be paths to its executable files. In Linux-based systems, executable files are usually stored in `/usr/bin/`; on Windows with Anaconda - in `C:\User\Anaconda3\Scripts` or `C:\User\Anaconda3\envs\current_environment\Scripts`.

Note 2: To use Dinosaur, java should be installed in the environment.

Note 3: Since Windows has a case-insensitive file system, despite `iqmma`'s overall compatibility some options related to other used programs (`-diffacto_args`, `-dino_args` to be precise) could not work properly according to Diffacto and Dinosaur case-sensitive option's names. With that fact in mind, it is recommended to use `iqmma` on Linux-based system.

### Config file

Both mods could be used with config file for an advanced settings configuration:

    iqmma -cfg path_to_config_file -cfg_category name_of_category_in_cfg

Example config file could be downloaded from here (example.ini) or could be generated by the command:

    iqmma -example_cfg path_to_file_to_be_created

Full option's description could be obtained with:

    iqmma -h

### Input files

Multiple formats could be used for input PSMs files. In simple matching mode it could be .tsv Identipy output or .pep.xml (.pepxml) from Identipy or MSFragger output or .mzid from msgf+ output or user's .tsv table with specified columns. However, -PSM_format parameter should be applied when using other formats except standart.

Columns: 'spectrum' - MS/MS spectrum id for peptide, 'peptide' - peptide sequence, 'protein' - protein name, related to this peptide, 'assumed_charge' - charge of the peptide, 'precursor_neutral_mass' - mass of the neutral peptide calculated by the formula 'precursor_neutral_mass' = mz * charge - charge * 1.00727649, 'RT exp' - experimental Retention Time of the peptide. 

For full quantitation mode PSMs files assumed to be PSMs_full.tsv tables of Scavager output.

### Output files

As an output `iqmma` generates `/feats_matched` directory with .tsv tables that contain information about the peptide and feature matched for it, table with differentially expressed proteins and their fold change for each feature detection method and Mix algorithm, and Venn diagram to show distribution of those DE proteins between feature detection related methods. Also Diffacto raw output for users filtering could be accessed in `/diffacto` directory or in the directory that was passed to `-diffacto_folder` option.

### Reanalyzing

In terms of the amount of time some stages of analysis could consume, `iqmma` tries to use existing files, that may have been left over from past runs, rather than overwriting them. Because of that, some options were added to avoid repeatable stages or unwanted usage.

`-overwrite_features` and `-feature_folder` - The most time-consuming stage often appears to be feature detection. So there are two possibilities to reanalyze data with already existing features. The first is to set `overwrite_features` to 0 (default) and let `iqmma` to find `/features` directory nearby either PSMs files or mzML files if it was already used on that files. And the second is to specify `feature_folder` parameter with directory, where features you need are stored, and also keep `overwrite_features` set to 0 not to overwrite them.

`-overwrite_matching` and `-matching_folder` - Matching is far less time-consuming than feature detection. If some parameters or even PSM or feature files were changed, and it is needed to reanalyze data, the right way to do so is either setting `overwrite_matching` to 1 (default 0) or pass another directory to the `-matching_folder`.

`-overwrite_first_diffacto` and `-diffacto_folder` - The first option overwrites results of the first stage of the quantitation strategy, where diffacto is used only on matched features from one feature detection at a time. Any changes in parameters referred to the previous stages need to turn it on, so it is set to 1 by default. The second changes the directory of storage of unfiltered quantitation results. 

## Links

- Diffacto repo: https://github.com/statisticalbiotechnology/diffacto

- Dinosaur repo: https://github.com/fickludd/dinosaur

- Biosaur2 repo: https://github.com/markmipt/biosaur2

- OpenMS guide: https://abibuilder.cs.uni-tuebingen.de/archive/openms/Documentation/release/latest/html/index.html

- Pypi: https://pypi.org/project/iqmma/

- Github: https://github.com/PostoenkoVI/IQMMA

- Mailing list: v.i.postoenko@gmail.com, garibova.02@gmail.com


## Citing iqmma

IQMMA: an efficient MS1 intensity extraction using multiple feature detection algorithms for DDA proteomics

Valeriy I. Postoenko, Leyla A. Garibova, Lev I. Levitsky, Julia A. Bubis, Mikhail V. Gorshkov, Mark V. Ivanov.

doi: https://doi.org/10.1101/2023.02.03.526776, biorxiv: https://www.biorxiv.org/content/10.1101/2023.02.03.526776v1
