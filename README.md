# ccks2021_track3_baseline
 a baseline to practice
路径可能会有问题，自己改改


torch==1.7.1

pyhton==3.7.1

transformers==4.7.0

cuda==11.0


this is a baseline, you can fix params to improve your score

#
some useful tricks:

1.https://github.com/alphadl/lookahead.pytorch

2.https://github.com/timgaripov/swa

3.https://github.com/lonePatient/multi-sample_dropout_pytorch

4.https://github.com/lonePatient/albert_pytorch/blob/master/prepare_lm_data_ngram.py

#
pretrain model download:
https://github.com/lonePatient/NeZha_Chinese_PyTorch

#
address: https://tianchi.aliyun.com/competition/entrance/531901/score

#
steps:

1.run pretrain_code/run_pretrain.py

2.run run_classify.py

3.run run_predictor.py

4.upload your submit file