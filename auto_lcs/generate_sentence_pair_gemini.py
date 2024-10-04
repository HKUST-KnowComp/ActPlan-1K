import json
import os

dirs = os.listdir("./annotation")

data = []
for current_dir, subdirs, files in os.walk("./annotation"):
    #print(current_dir)
    #for dirname in subdirs:
    #    print("\t" + dirname)

    if "plan_gemini.txt" not in files:
        continue

    print(current_dir)
    print(files)
    subdata = dict()
    subdata["dir"] = current_dir

    fread_1 = open(os.path.join(current_dir, "plan_gemini.txt"))
    candidate = ""
    for line in fread_1:
        candidate += line
    subdata["candidate"] = candidate

    reference = ""
    #plan_gold_2 for gemini-pro
    if "plan_gold_2.txt" in files:
        fread_2 = open(os.path.join(current_dir, "plan_gold_2.txt"))
        for line in fread_2:
            reference += line
    else:
        reference = candidate
    subdata["reference"] = reference

    print(subdata)
    data.append(subdata)
    #break

fw = open("gemini_pairs.jsonl", 'w')
for subdata in data:
    fw.write(json.dumps(subdata)+'\n')
fw.close()
