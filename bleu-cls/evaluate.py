import torch.nn
import tqdm
import numpy as np

def tokenize(sentence_1, sentence_2, tokenizer, MAX_SEQ_LENGTH=512):
    #import pdb
    #pdb.set_trace()

    tokens = tokenizer(sentence_1, sentence_2, truncation="longest_first", \
                       padding="longest", max_length=MAX_SEQ_LENGTH, \
                       add_special_tokens=True, return_tensors="pt")
    input_ids = tokens.input_ids
    input_masks = tokens.attention_mask
    token_type_ids = tokens.token_type_ids
    return input_ids, input_masks, token_type_ids

def MCC(preds, corrects, tags):
    N = 0
    M = 0
    GPT4 = [0., 0., 0., 0.]
    Gemini = [0., 0., 0., 0.]

    GPT4_info = []
    Gemini_info = []

    for label_pred, label_gold, tag in zip(preds, corrects, tags):
        if label_pred == 0 and label_gold == 0:
            if tag == "gpt":
                GPT4[0] += 1
                GPT4_info.append([0, 0])
            else:
                Gemini[0] += 1
                Gemini_info.append([0, 0])
        elif label_pred == 1 and label_gold == 0:
            if tag == "gpt":
                GPT4[1] += 1
                GPT4_info.append([1, 0])
            else:
                Gemini[1] += 1
                Gemini_info.append([1, 0])
        elif label_pred == 0 and label_gold == 1:
            if tag == "gpt":
                GPT4[2] += 1
                GPT4_info.append([0, 1])
            else:
                Gemini[2] += 1
                Gemini_info.append([0, 1])
        else:
            if tag == "gpt":
                GPT4[3] += 1
                GPT4_info.append([1, 1])
            else:
                Gemini[3] += 1
                Gemini_info.append([1, 1])

    print(GPT4)
    #TP, FP, FN, TN
    p_o = GPT4[0]*GPT4[3] - GPT4[1]*GPT4[2]
    p_e = np.sqrt((GPT4[0]+GPT4[1])*(GPT4[0]+GPT4[2])*(GPT4[3]+GPT4[1])*(GPT4[3]+GPT4[2]))
    GPT4_MCC = p_o/p_e
    print("GPT4 MCC score: ", GPT4_MCC)
    GPT4_info = np.array(GPT4_info)
    GPT4_acc = np.mean(GPT4_info[:,0]==GPT4_info[:,1])
    print("GPT4 acc score: ", np.mean(GPT4_info[:,0]==GPT4_info[:,1]))

    print(Gemini)
    #TP, FP, FN, TN
    p_o = Gemini[0]*Gemini[3] - Gemini[1]*Gemini[2]
    p_e = np.sqrt((Gemini[0]+Gemini[1])*(Gemini[0]+Gemini[2])*(Gemini[3]+Gemini[1])*(Gemini[3]+Gemini[2]))
    Gemini_MCC = p_o/p_e
    print("Gemini MCC score: ", Gemini_MCC)
    Gemini_info = np.array(Gemini_info)
    Gemini_acc = np.mean(Gemini_info[:,0]==Gemini_info[:,1])
    print("Gemini acc score: ", np.mean(Gemini_info[:,0]==Gemini_info[:,1]))
    return GPT4_MCC, GPT4_acc, Gemini_MCC, Gemini_acc

def evaluate(model, dataloader, tokenizer, epoch, split="val"):
    model.eval()

    eval_loss = 0
    np_eval_steps = 0
    predicted_labels, correct_labels = [], []
    tags = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            sentence_1, sentence_2, tag, label = batch
            input_ids, input_masks, token_type_ids = tokenize(sentence_1, sentence_2, tokenizer)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_masks = input_masks.cuda()
                token_type_ids = token_type_ids.cuda()
                label = label.cuda()
            tmp_eval_loss, logits = model(input_ids, input_masks, token_type_ids, label)

            outputs = np.argmax(logits.to('cpu'), axis=1)
            label_ids = label.to('cpu').numpy()

            predicted_labels += list(outputs)
            correct_labels += list(label_ids)
            tags += list(tag)

            eval_loss += tmp_eval_loss.mean().item()
            np_eval_steps += 1

    eval_loss = eval_loss / np_eval_steps

    fw = open("./results_large/pred_{}_{}.txt".format(epoch, split), "w")
    #fw = open("./results/pred_{}_{}.txt".format(epoch, split), "w")
    for pred, correct, tag in zip(predicted_labels, correct_labels, tags):
        fw.write("{}\t{}\t{}\n".format(pred, correct, tag))
    fw.close()

    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    eval_acc = np.mean(correct_labels==predicted_labels)
    #print("accuracy: ", np.mean(correct_labels==predicted_labels))
    GPT4_MCC, GPT4_acc, Gemini_MCC, Gemini_acc = MCC(predicted_labels, correct_labels, tags)
    return eval_loss, eval_acc, GPT4_MCC, GPT4_acc, Gemini_MCC, Gemini_acc


