## Using Semantic Similarity in Crawling-based Web Application Testing

### Experimental 1 results

Download from Google Drive:
[experiment.zip](https://drive.google.com/file/d/0BzB2SjSX7m4yLWtxQUlsQmNiaVE/view?usp=sharing) (1.24 GB)

### Statistics

`results.ods`: The statistics

`forms.csv`: A list of the subject forms

### Programs

`train_and_test.py`: For running the proposed natural-language method

`rule_match.py`: For running the rule-based method

`combine.py`: For getting results for the RB+NL-n, RB+NL-m, and RB+NL-b methods

`preprocess.py`: A library for feature extraction

### Corpus and the labeled topics

`forms`: Offline cache of the subject forms

`corpus`: Preprocessed corpus from `forms`

`corpus/label-all-corpus.json`: Labeled topics of all 985 input fields from the 100 subject forms used in the experiments

