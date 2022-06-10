from argparse import ArgumentParser
import json
from collections import OrderedDict

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary

from model_utils import TimeDistributed

class SEXLNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        config = AutoConfig.from_pretrained(self.hparams.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, do_lower_case=True)
        self.model = AutoModel.from_pretrained(self.hparams.model_name).to('cuda')
        self.pooler = SequenceSummary(config).to('cuda')
        # self.pooler_concept = SequenceSummary(config).to('cuda')

        self.classifier = nn.Linear(config.d_model, self.hparams.num_classes)

        self.concept_idx = OrderedDict()
        with open('data/temp_with_parse.json', 'r') as input_file:
            for i, line in enumerate(input_file):
                json_line = json.loads(line)
                sentence = json_line["sentence"].strip().strip(' .')
                self.concept_idx[i] = sentence
        self.tokenized_concepts = self.tokenizer(list(self.concept_idx.values()), padding=True, return_tensors="pt")
        for key, value in self.tokenized_concepts.items():
            self.tokenized_concepts[key] = value.to('cuda')

        self.phrase_logits = TimeDistributed(nn.Linear(config.d_model,
                                                        self.hparams.num_classes))
        self.sequence_summary = SequenceSummary(config)

        self.topk =  self.hparams.topk
        # self.topk_gil_mlp = TimeDistributed(nn.Linear(config.d_model,
        #                                               self.hparams.num_classes))

        self.topk_gil_mlp = nn.Linear(config.d_model,self.hparams.num_classes)

        self.multihead_attention = torch.nn.MultiheadAttention(config.d_model,
                                                               dropout=0.2,
                                                               num_heads=8)

        self.activation = nn.ReLU()

        self.lamda = self.hparams.lamda
        self.gamma = self.hparams.gamma

        self.dropout = nn.Dropout(config.dropout)
        self.loss = nn.CrossEntropyLoss()
        
                    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--min_lr", default=0, type=float,
                            help="Minimum learning rate.")
        parser.add_argument("--h_dim", type=int,
                            help="Size of the hidden dimension.", default=768)
        parser.add_argument("--n_heads", type=int,
                            help="Number of attention heads.", default=1)
        parser.add_argument("--kqv_dim", type=int,
                            help="Dimensionality of the each attention head.", default=256)
        parser.add_argument("--num_classes", type=float,
                            help="Number of classes.", default=2)
        parser.add_argument("--lr", default=2e-5, type=float,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay rate.")
        parser.add_argument("--warmup_prop", default=0.01, type=float,
                            help="Warmup proportion.")
        return parser

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99),
                     eps=1e-8)
        
    def add_pretrained(self):
        self.model_pretrained = AutoModel.from_pretrained('xlnet-base-cased')
        self.model_pretrained.to('cuda')
        self.model_pretrained.eval()
        
    def generate_concept_emb(self):
        
        outputs = self.model(**self.tokenized_concepts)
        self.concept_reps = self.pooler(outputs['last_hidden_state'])
        # outputs = self.model_pretrained(**self.tokenized_concepts)
        # self.concept_reps = outputs['last_hidden_state'][:, -1]
        
    def forward_classifier(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """Returns the pooled token
        """
        outputs = self.model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True)
        cls_hidden_state = self.dropout(self.pooler(outputs["last_hidden_state"]))
        # outputs = self.model_pretrained(input_ids=input_ids,
        #                      token_type_ids=token_type_ids,
        #                      attention_mask=attention_mask,
        #                      output_hidden_states=True)
        # cls_hidden_state = outputs["last_hidden_state"][:, -1]
        return cls_hidden_state, outputs["last_hidden_state"]

    def forward(self, batch):
        
        self.generate_concept_emb()
        # self.concept_store = self.concept_store.to(self.model.device)
        # print(self.concept_store.size(), self.hparams.concept_store)
        tokens, tokens_mask, padded_ndx_tensor, labels = batch
        tokens, tokens_mask, padded_ndx_tensor, labels = tokens.to('cuda'), tokens_mask.to('cuda'), padded_ndx_tensor.to('cuda'), labels.to('cuda')

        # step 1: encode the sentence
        sentence_cls, hidden_state = self.forward_classifier(input_ids=tokens,
                                                             token_type_ids=tokens_mask,
                                                             attention_mask=tokens_mask)
        logits = self.classifier(sentence_cls)

        lil_logits = self.lil(hidden_state=hidden_state,
                              nt_idx_matrix=padded_ndx_tensor)  # [batch size, phrase num, cls num]
        lil_logits_mean = torch.mean(lil_logits, dim=1)  # [batch size, cls num]
        gil_logits, topk_indices = self.gil(pooled_input=sentence_cls)  # [batch size, cls num]

        logits = logits + self.lamda * lil_logits_mean + self.gamma * gil_logits
        predicted_labels = torch.argmax(logits, -1)
        if labels is not None:
            acc = torch.true_divide(
                (predicted_labels == labels).sum(), labels.shape[0])
        else:
            acc = None

        return logits, acc, {"topk_indices": topk_indices,
                             "lil_logits": lil_logits}
        
    def lil(self, hidden_state, nt_idx_matrix):  # [batch size, seq length, 768] and [batch size, phrase number, seq length]
        phrase_level_rep = self.activation(torch.bmm(nt_idx_matrix, hidden_state))  # [batch size, phrase number, 768]
        pooled_seq_rep = self.sequence_summary(hidden_state).unsqueeze(1)  # [batch size, 768] -> [batch size, 1 , 768]
        phrase_level_activations = phrase_level_rep - pooled_seq_rep  # [batch size, phrase number, 768] When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
        phrase_level_logits = self.phrase_logits(phrase_level_activations)  # [batch size, phrase number, class num]
        return phrase_level_logits

    def gil(self, pooled_input):  # [batch size, 768]
        batch_size = pooled_input.size(0)
        inner_products = torch.mm(pooled_input, self.concept_reps.T)  # [batch size, 768] * [768, #concepts] = [batch size, #concepts]
        concept_norm, input_norm = torch.norm(self.concept_reps, dim=1), torch.norm(pooled_input, dim=1)
        cos_sim = torch.div(torch.div(inner_products, concept_norm.T).T, input_norm).T  # [batch size, #concepts]
        # print(cos_sim)
        _, topk_indices = torch.topk(cos_sim, k=self.topk)
        topk_concepts = torch.index_select(self.concept_reps, 0, topk_indices.view(-1))  # [k, 768]
        topk_concepts = topk_concepts.view(batch_size, self.topk, -1).contiguous()

        concat_pooled_concepts = torch.cat([pooled_input.unsqueeze(1), topk_concepts], dim=1)  # [batch size, 1 + k, 768]
        attended_concepts, _ = self.multihead_attention(query=concat_pooled_concepts,  # [batch size, 1 + k, 768]
                                                     key=concat_pooled_concepts,
                                                     value=concat_pooled_concepts)

        gil_topk_logits = self.topk_gil_mlp(attended_concepts[:,0,:])  # [batch size, 768] -> [batch size, cls num]
        # print(gil_topk_logits.size())
        # gil_logits = torch.mean(gil_topk_logits, dim=1)
        return gil_topk_logits, topk_indices

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, _ = self(batch)
        loss = self.loss(logits, batch[-1])
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, _ = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])

        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, _ = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])
        return {"loss": loss}

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("val_loss_step", None)
        tqdm_dict.pop("val_acc_step", None)
        return tqdm_dict


if __name__ == "__main__":
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
