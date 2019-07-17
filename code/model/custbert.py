
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertModel
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
class cusBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(cusBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*800, num_labels)
        self.rnn = nn.LSTM(config.hidden_size,800,num_layers = 1,bidirectional=True,dropout=0.5)
        self.attn = nn.Linear(2*800,1)
        self.f = nn.Softmax()
        self.apply(self.init_bert_weights)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        #encoded_layers,_  = torch.utils.checkpoint.checkpoint(self.bert, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        encoded_layers,_ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        encoded_layers = self.dropout(encoded_layers)
        output, (hidden, _) = self.rnn(encoded_layers.transpose(0,1))
        hidden = self.dropout(output.transpose(0,1))
        attn = self.f(self.attn(hidden).squeeze(-1)).unsqueeze(-1)
        logits = self.classifier(torch.sum(attn*hidden, dim = 1))

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits