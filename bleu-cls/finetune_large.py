import transformers
import torch
import torch.nn as nn
from dataset import load_data, EmbDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import tqdm
from tqdm import trange
from evaluate import evaluate, tokenize
import os, math
import logging
logging.disable(logging.WARNING)

class BleurtModel(nn.Module):
    """
    bleurt model with cls layer
    """

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.mlp = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, input_masks, token_type_ids, labels):
        output = self.bert(input_ids=input_ids,
                           attention_mask=input_masks,
                           token_type_ids=token_type_ids,
                           output_hidden_states=True,
                           return_dict=True)

        last_hidden = output.last_hidden_state
        feat = last_hidden[:,0,:]
        feat = self.dropout(feat)
        pred = self.mlp(feat)
        loss = self.loss(pred, labels)
        return loss, pred

def train():

    BATCH_SIZE = 12
    MAX_SEQ_LENGTH = 512

    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 70
    LEARNING_RATE = 1e-6
    #LEARNING_RATE = 5e-6
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 5

    OUTPUT_DIR = "./checkpoints-large"
    MODEL_FILE_NAME = "best-model.ckpt"

    data_file = "./data/finetuning_data.jsonl"
    activity_file = "./data/activities.txt"
    train_data, val_data, test_data = load_data(data_file, activity_file)
    train_dataset = EmbDataset(train_data, split="train")
    val_dataset = EmbDataset(val_data, split="val")
    test_dataset = EmbDataset(test_data, split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    state_dict = torch.load("./bleurt-large-512/pytorch_model.bin")
    tokenizer = BertTokenizer.from_pretrained("./bleurt-large-512/")
    print(tokenizer.all_special_tokens)
    special_tokens_dict = {"additional_special_tokens": ["switch-on", "switch-off", \
                               "place_inside", "place_on_top", \
                               "walk_to"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.all_special_tokens)

    config = transformers.BertConfig.from_pretrained("bert-large-uncased", output_hidden_states = True, output_attentions = True)
    model = BleurtModel(config)

    named_parameters = [k for k, v in model.state_dict().items()]
    model.load_state_dict(state_dict, strict=False)
    model.bert.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available():
        model = model.cuda()
 
    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    #scheduler = (optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)

    loss_history = []
    acc_history = []
    best_epoch = 0
    best_val_acc = 0.0
    best_test_acc = 0.0
    fw = open('log_large.txt', 'w')
    for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            sentence_1, sentence_2, tag, label = batch
            input_ids, input_masks, token_type_ids = tokenize(sentence_1, sentence_2, tokenizer, MAX_SEQ_LENGTH)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_masks = input_masks.cuda()
                token_type_ids = token_type_ids.cuda()
                label = label.cuda()

            outputs = model(input_ids, input_masks, token_type_ids, label)
            loss = outputs[0]
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM) 

                optimizer.step()
                optimizer.zero_grad()
                #scheduler.step()
            print("Epoch: {}, step: {}, loss: {}".format(epoch, step, loss.item()))

        val_loss, val_acc, val_GPT4_MCC, val_GPT4_acc, val_Gemini_MCC, val_Gemini_acc = evaluate(model, val_dataloader, tokenizer, epoch, split="val")
        test_loss, test_acc, test_GPT4_MCC, test_GPT4_acc, test_Gemini_MCC, test_Gemini_acc = evaluate(model, test_dataloader, tokenizer, epoch, split="test")

        #print("Acc history: ", loss_history)
        
        print("Dev acc: ", val_acc, val_GPT4_MCC, val_GPT4_acc, val_Gemini_MCC, val_Gemini_acc)
        print("Test acc: ", test_acc, test_GPT4_MCC, test_GPT4_acc, test_Gemini_MCC, test_Gemini_acc)
        fw.write("epoch: {} dev: {:.3f} val_G_mcc: {:.3f} val_G_acc: {:.3f} val_Ge_mcc: {:.3f} val_Ge_acc: {:.3f} test: {:.3f} test_G_mcc: {:.3f} test_G_acc: {:.3f} test_Ge_mcc: {:.3f} test_Ge_acc: {:.3f}\n".format(epoch, val_acc, val_GPT4_MCC, val_GPT4_acc, val_Gemini_MCC, val_Gemini_acc, test_acc, test_GPT4_MCC, test_GPT4_acc, test_Gemini_MCC, test_Gemini_acc))
        
        if len(acc_history) > 0 and val_loss < min(loss_history):
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            best_epoch = epoch
            best_val_acc = val_acc
            best_test_acc = test_acc

        loss_history.append(val_loss)
	acc_history.append(val_acc)

        print("best epoch: ", best_epoch)
        print("best dev acc: ", best_val_acc)
        print("best test acc: ", best_test_acc)

if __name__ == "__main__":
    train()
