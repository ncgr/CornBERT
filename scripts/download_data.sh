#!/usr/bin/bash

if [ "$#" -ne 1 ]; then
    echo "$(basename "$0") DOWNLOAD_DIRECTORY"
    exit 1
fi

# the download directory paths
assemblies_dir=${1}/assemblies
annotations_dir=${1}/annotations

# make the download directories
mkdir ${assemblies_dir}
mkdir ${annotations_dir}

# download assemblies
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/S288C.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/DBVPG6044.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/DBVPG6765.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/SK1.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/Y12.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/YPS128.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/UWOPS034614.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/CBS432.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/N44.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/YPS138.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/UFRJ50816.genome.fa.gz
wget -P ${assemblies_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_Genome/UWOPS919171.genome.fa.gz

# download annotations
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/S288C.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/DBVPG6044.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/DBVPG6765.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/SK1.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/Y12.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/YPS128.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/UWOPS034614.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/CBS432.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/N44.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/YPS138.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/UFRJ50816.all_feature.gff.gz
wget -P ${annotations_dir} http://yjx1217.github.io/Yeast_PacBio_2016/data/Nuclear_GFF/UWOPS919171.all_feature.gff.gz
