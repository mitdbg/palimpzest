#!/bin/bash
#This script can be used to download and extract the test data for the palimpzest demos
# Usage: bash testdata/download.sh
# Requirements: wget, tar

# Move to the testdata directory
pushd testdata
# Download the test data
echo "Downloading the test data..."
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/askem.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/askem-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/bdf-usecase3-pdf.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/bdf-usecase3-references.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/bdf-usecase3-references-pdf.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/bdf-usecase3-references-pdffull.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/bdf-usecase3-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-html.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-matching.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-medium.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-tiny-filtered.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-urls.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/enron-eval.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/enron-eval-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/enron-small.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/enron-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/equation-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/groundtruth.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/images-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/pdfs-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-10.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-15.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-20.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-5.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-tiny.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/vldbdownload.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-25.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/real-estate-eval-30.tar.gz 
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/biofabric-pdf.tar.gz
wget -nc https://people.csail.mit.edu/gerarvit/PalimpzestData/enron-tiny.csv

echo "Extracting the test data..."
# Extract the test data
for f in *.tar.gz; do
  tar -xzf $f
done
rm *.tar.gz
popd
echo "Done!"
