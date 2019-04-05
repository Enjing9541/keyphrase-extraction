
To be able to run each Keyprahse Extraction Methods:


INSTALLATION
------------------------------------------------------------------------------------------

1. Make sure all the required packages have been sucessfully installed (Please refer to requirement.txt)

2. Make sure the standford POS tagger is downloaded and included in the standford folder
Full Stanford Tagger version 3.8.0 can be downloaded from:https://nlp.stanford.edu/software/tagger.shtml

3. Make sure sent2vec is installed from https://github.com/epfml/sent2vec
Make sure sent2vec pretrained model is downloaded and included in the sent2vec folder

4. Make sure the source codes for EmbedRank from https://github.com/swisscom/ai-research-keyphrase-extraction is included in the swisscom_ai folder

5. Make sure all the dataset is appropriate downloaded and included in the Dataset folder.
Inspec can be downloaded from: https://github.com/boudinfl/hulth-2003-pre
Semeval 2017 can be downloaded from: https://scienceie.github.io/resources.html


USAGE
------------------------------------------------------------------------------------------

6. To evaluate and obtain keyphrase extraction results, simply input python model_name.py 
* All the evaluations are based on the Inspec Dataset.


There are in total 6 different embedding methods.
	python EmbedRank.py
	python Flair.py
	python BERT.py
	python Fasttext.py
	python Glove.py
	python GloveAndFlair.py