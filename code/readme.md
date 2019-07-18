 # 代码参考及引用
  
  -  使用并修改[huggingface-pytorch-transformers/examples/](https://github.com/huggingface/pytorch-transformers) 中分类脚本及语言模型预训练脚本
  
  -  借鉴参考[google-Unsupervised Data Augmentation](https://github.com/google-research/uda) 中tsa和vat的写法
  
  -  借鉴参考[kaggle-Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python) 中stacking部分代码


## model
  - **bert_get_oof.py**: 使用bert输出集成需要的oof文档
  
  - **custbert.py** 自定义bert（bert之上加bilstm和attention）
  
  - **run_classifier_dataset_utils.py**: 封装bert需要的数据处理
  
  - **simple_classifier.py**：包含tsa和多句子拼接的代码
  
  - **modeling.py**：添加了类bert_context_classification，用于训练多句子拼接
  
  - **stacking.py**: 使用stacking集成

## lm_finetune

  - finetune_on_pregenerated：执行语言模型预训练
  
  - pregenerate_training_data：处理语料生成语言模型预训练的训练样本
