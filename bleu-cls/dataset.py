import json
from torch.utils.data import Dataset
import numpy as np
import random

def load_data(data_file, activity_file):
    train_data = []
    val_data = []
    test_data = []

    train_activity = set()
    val_activity = set()
    test_activity = set()
    for line in open(activity_file):
        info = line.strip().split("\t")
        if info[1] == "train":
            train_activity.add(info[0])
        elif info[1] == "val":
            val_activity.add(info[0])
        elif info[1] == "test":
            test_activity.add(info[0])
    print("train activity: ", len(list(train_activity)))
    print("val activity: ", len(list(val_activity)))
    print("test activity: ", len(list(test_activity)))
    for line in open(data_file):
        subdata = json.loads(line.strip())
        if subdata["activity"] in train_activity:
            train_data.append(subdata)
        elif subdata["activity"] in val_activity:
            val_data.append(subdata)
        elif subdata["activity"] in test_activity:
            test_data.append(subdata)
    print("train instance: ", len(train_data))
    print("val instance: ", len(val_data))
    print("test instance: ", len(test_data))
    return train_data, val_data, test_data


class EmbDataset(Dataset):
    def __init__(self, data, split="train"):
        self.data = data
        self.split = split
        if split != "train":
            self.sample_list = []
            for instance in data:
                #print(instance.keys())
                #dict_keys(['path', 'activity', 'pos_gpt', 'neg_gpt', 'gold_gpt', 'pos_gemini', 'neg_gemini', 'gold_gemini'])
                gold = instance["gold_gpt"][0]
                gold_gemini = instance["gold_gemini"]
                for i in range(len(instance["pos_gpt"])):
                    if len(gold_gemini) > 0:
                        self.sample_list.append([gold_gemini[0], '[SEP] '+instance["pos_gpt"][i], "gpt", 1])
                    else:
                        self.sample_list.append([gold, '[SEP] '+instance["pos_gpt"][i], "gpt", 1])
                for j in range(len(instance["neg_gpt"])):
                    if len(gold_gemini) > 0:
                        self.sample_list.append([gold_gemini[0], '[SEP] '+instance["neg_gpt"][j], "gpt", 0])
                    else:
                        self.sample_list.append([gold, '[SEP] '+instance["neg_gpt"][j], "gpt", 0])
                for i in range(len(instance["pos_gemini"])):
                    self.sample_list.append([gold, '[SEP] '+instance["pos_gemini"][i], "gemini", 1])
                for j in range(len(instance["neg_gemini"])):
                    self.sample_list.append([gold, '[SEP] '+instance["neg_gemini"][j], "gemini", 0])

    def __len__(self):
        if self.split == "train":
            return len(self.data)
        else:
            return len(self.sample_list)


    def __getitem__(self, idx):
        if self.split != "train":
            return self.sample_list[idx]

        #rand_ind = np.random.randint(0, len(self.data))
        sample_data = self.data[idx]
        pos_list = sample_data["gold_gpt"] + sample_data["pos_gemini"] + sample_data["gold_gemini"]
        neg_list = sample_data["neg_gpt"] + sample_data["neg_gemini"]
        seed = np.random.rand()
        #if seed < 0.7:
        #if seed < 0.6:
        #if seed < 0.5:
        #if seed < 0.4:
        if seed < 0.3:
            sentences = random.sample(pos_list, 2)
            return [sentences[0], '[SEP] '+sentences[1], "mix", 1]
        #elif seed >= 0.7 and seed < 0.9 and len(neg_list) > 0:
        #elif seed >= 0.6 and seed < 0.9 and len(neg_list) > 0:
        #elif seed >= 0.5 and seed < 0.9 and len(neg_list) > 0:
        #elif seed >= 0.4 and seed < 0.9 and len(neg_list) > 0:
        #elif seed >= 0.3 and seed < 0.8 and len(neg_list) > 0:
        #elif seed >= 0.3 and seed < 0.7 and len(neg_list) > 0:
        #elif seed >= 0.3 and seed < 0.8 and len(neg_list) > 0: roberta-large old, lr=2e-5
        #elif seed >= 0.3 and seed < 0.6 and len(neg_list) > 0: roberta-large old2, lr=2e-5
        #elif seed >= 0.3 and seed < 0.6 and len(neg_list) > 0: roberta-large old3, lr=5e-5
        elif seed >= 0.3 and seed < 0.8 and len(neg_list) > 0:
            sentence_1 = random.sample(pos_list, 1)
            sentence_2 = random.sample(neg_list, 1)
            return [sentence_1[0], '[SEP] '+sentence_2[0], "mix", 0]
        else:
            sentence_1 = random.sample(pos_list, 1)
            items = sentence_1[0].split("\n")
            random.shuffle(items)
            sentence_2 = "\n".join(items)
            return [sentence_1[0], '[SEP] '+sentence_2, "random", 0] 


if __name__ == "__main__":
    data_file = "./data/finetuning_data.jsonl"
    activity_file = "./data/activities.txt"
    train_data, val_data, test_data = load_data(data_file, activity_file)
    train_dataset = EmbDataset(train_data, split="train")
    val_dataset = EmbDataset(val_data, split="val")
    test_dataset = EmbDataset(test_data, split="test")
    print(train_dataset[0])
    print(val_dataset[0])
    print(test_dataset[0])
