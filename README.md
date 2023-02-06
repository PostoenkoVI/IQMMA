# IQMMA - Intensity-based Quantitative Mix and Match Approach for data-dependent proteomics

`iqmma` consumes LC-MS data (**mzML**) and results of post-search analysis (or just peptide identifications) (**tsv**) and performs multiple tools (Dinosaur, Biosaur and OpenMS FeatureFinderCentroided) feature detection, peptide-intensity mathcing and quantitation.

## Installation

Using pip:

    pip install iqmma

## Usage

`iqmma` has two working mods. First of all, it can be a quantitation workflow with generating peptide features using multiple tools, matching them on peptides, and two Diffacto quantitation stages (separated and mixed, where the algorithm is choosing the best intensities for each peptide between different feature detections). The second one is stopping after peptide-feature matching to allow the user to apply any other quantitative approach on matched intensities.

For `iqmma` to work properly each mzML file must have a related PSM file which name starts with the name of mzML.
For basic usage all PSMs and mzML files should be stored in the same directory, otherwise -PSM_folder parameter must be applied. All PSMs files must be *PSM_full.tsv tables obtained from Scavager output (https://github.com/markmipt/scavager).
Basic command for quantitation mode:

    iqmma -bio2 path_to_Biosaur2 -dino path_to_Dinosaur -openms path_to_openMS -dif path_to_Diffacto -s1 paths_to_mzml_files_from_sample_1_*.mzML -s2 paths_to_mzml_files_from_sample_2_*.mzML -outdir out_dir

Basic command for matching peptide intensities: (all .mzml files goes into first sample without any differences, no quantitation applied)

    iqmma -bio2 path_to_Biosaur2 -dino path_to_Dinosaur -openms path_to_openMS -dif path_to_Diffacto -s1 paths_to_all_mzml_files_*.mzML -outdir out_path

Or both mods could be used with config file for an advanced usage and configuration:

    iqmma -cfg path_to_config_file -cfg_category name_of_category_in_cfg
    
Example config file could be downloaded from here (example.ini). 


Full options description could be obtained with:

    iqmma -h

### Input files

As an input PSMs files multiple formats could be used. For simple matching mode it could be .tsv Identipy output or .pep.xml (.pepxml) from Identipy or MSFragger output or .mzid from msgf+ output or user's .tsv table with specified columns. However, -PSM_format parameter shoud be applied to use other formats except standart.

Columns: 'spectrum' - MS/MS spectrum id for peptide, 'peptide' - peptide sequence, 'protein' - protein name, related to this peptide, 'assumed_charge' - charge of the peptide, 'precursor_neutral_mass' - mass of the neutral peptide, 'RT exp' - experimental Retention Time of the peptide. 

For full quantitation mode PSMs files assumed to be PSMs_full.tsv tables from Scavager output.

### Output files

As an output `iqmma` generates 'feats_matched' dir with .tsv tables that contain information about the peptide and feature matched for it, table with differetially expressed proteins and their fold change for each feature detection method and Mix algorithm, and Venn diagramm to show distribution of those DE proteins between feature detection related metods.

## Links

- Diffacto repo: https://github.com/statisticalbiotechnology/diffacto
- Dinosaur repo: https://github.com/fickludd/dinosaur
- Biosaur2 repo: https://github.com/markmipt/biosaur2

- Mailing list: v.i.postoenko@gmail.com, garibova.02@gmail.com


## Citing iqmma

Will be available soon... 
