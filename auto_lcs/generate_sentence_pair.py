import json
import os

dirs = os.listdir("./annotation")

data = []
for current_dir, subdirs, files in os.walk("./annotation"):
    #print(current_dir)
    #for dirname in subdirs:
    #    print("\t" + dirname)

    if "plan_gpt-4v.txt" not in files:
        continue

    print(current_dir)
    print(files)
    subdata = dict()
    subdata["dir"] = current_dir

    fread_1 = open(os.path.join(current_dir, "plan_gpt-4v.txt"))
    # same for claude_v3_haiku and claude_v3_sonnet
    #fread_1 = open(os.path.join(current_dir, "plan_claude_v3_haiku.txt"))
    #fread_1 = open(os.path.join(current_dir, "plan_claude_v3_sonnet.txt"))
 
    candidate = ""
    for line in fread_1:
        candidate += line
    subdata["candidate"] = candidate

    reference = ""
    #plan_gold_1 for gpt-4v and claude
    if "plan_gold_1.txt" in files:
        fread_2 = open(os.path.join(current_dir, "plan_gold_1.txt"))
        for line in fread_2:
            reference += line
    else:
        reference = candidate
    subdata["reference"] = reference

    print(subdata)
    data.append(subdata)

fw = open("gpt-4v_pairs.jsonl", 'w')
#fw = open("claude_v3_haiku_pairs.jsonl", 'w')
#fw = open("claude_v3_sonnet_pairs.jsonl", 'w')
for subdata in data:
    fw.write(json.dumps(subdata)+'\n')
fw.close()
