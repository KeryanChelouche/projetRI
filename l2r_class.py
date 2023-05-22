import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

class LTRDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, document, relevance = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            query,
            document,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relevance': torch.tensor(relevance, dtype=torch.float)
        }

 
class BertLTRModel(nn.Module):
    def __init__(self, bert_model):
        super(BertLTRModel, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        relevance_score = self.linear(pooled_output)
        return relevance_score


def create_one_hot(target):
    # Ensure the target is a 2D tensor of shape (batch_size, 1)
    if len(target.shape) == 1:
        target = target.view(-1, 1)
    # Create a one-hot tensor of shape (batch_size, 2)
    one_hot = torch.zeros(target.size(0), 2).to(target.device)
    # Fill the one-hot tensor with (score, 1-score) values
    one_hot[:, 0] = target.squeeze()
    one_hot[:, 1] = 1 - target.squeeze()

    return one_hot

class WSLSCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super(WSLSCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        with torch.no_grad():
            # We first create a tensor with same shape as input filled with smoothing value
            true_dist = torch.ones_like(target)
            # This is to ensure that the non-relevant documents (target != 1) have the (1-target*smoothing, target*smoothing) condition
            non_relevant_indices = (target != 1).squeeze()
            relevant_indices = (target == 1).squeeze()


            # target = target.view(-1, 1)
            true_dist[non_relevant_indices] = target[non_relevant_indices].squeeze() * self.smoothing

            # Next, scatter the remaining confidence to the correct class index
            true_dist[relevant_indices] = 1-self.smoothing
            true_dist = create_one_hot(true_dist)
        # We then calculate the log probability of the inputs (this is done internally in CrossEntropyLoss)
        log_prob = F.log_softmax(input, dim=-1)

        # Our final loss is then the mean negative log likelihood
        return torch.mean(torch.sum(-true_dist * log_prob, dim=-1))