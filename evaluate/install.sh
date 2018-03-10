#!/usr/bin/env bash

echo `date`

# remove existing resources
rm -rf resources && mkdir resources
rm -rf WordNetMapper

# download model
cd resources
wget http://kyoto.let.vu.nl/~minh/wsd/model-h2048p512.zip
unzip model-h2048p512.zip
cd ..

# wordnet 171
cd resources
mkdir wordnet_171
cd wordnet_171/
wget http://wordnetcode.princeton.edu/1.7.1/WordNet-1.7.1.tar.gz
tar -zxvf WordNet-1.7.1.tar.gz
cd ../..

# download competition df files
cd resources
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se2-aw.p
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/sem2013-aw.p
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se2-aw-framework.p
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se13-aw-framework.p
unzip WSD_Unified_Evaluation_Datasets.zip
cd WSD_Unified_Evaluation_Datasets
javac Scorer.java
cd ..
cd ..

# download annotated corpora
cd resources
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip
unzip WSD_Training_Corpora.zip
cd ..

# git clone WordNetMapper
git clone --depth=1 https://github.com/MartenPostma/WordNetMapper

echo `date`


