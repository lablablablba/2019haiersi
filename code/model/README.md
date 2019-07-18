# model说明
  - **bert_get_oof**: 使用bert输出集成需要的oof文档
  
  - **custbert** 自定义bert（bert之上加bilstm和attention）
  
  - **run_classifier_dataset_utils.py**: 封装bert需要的数据处理
   
  - **stacking**: 使用stacking集成
  
  - **simple_classifier.py**：包含tsa和多句子拼接的代码
  
  - **modeling.py**：添加了类bert_context_classification，用于训练多句子拼接
