#!/usr/bin/bash

# CornBERT considers the 1000bp upstream of a gene to be its regulatory sequence. This
# scripts extracts these sequences from a FASTA using a GFF containing gene annotations
# for the FATSA.

if [ "$#" -ne 2 ]; then
    echo "$(basename "$0") GFF_GZ FASTA"
    exit 1
fi

# a temporary GFF file that will hold the regulatory elements
tmp_gff="$1.regulatory.gff"

# read the GFF gzip file to stdout, filter to only contain gene lines, adjust the gene
# coordinates to be the 1000bp that precede the gene
gunzip -c $1 | grep 'gene' | awk -v OFS='\t' '{$5=$4-1; $4=$4-1000; print}' > $tmp_gff

# extract the regulatory sequences into a new FASTA
bedtools getfasta -fi $2 -bed $tmp_gff

# remove the temporary GFF
rm $tmp_gff
