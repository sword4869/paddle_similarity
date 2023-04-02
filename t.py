import json

json_path = 'data/qa.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(len(data['keyword']['*']))
input()
input()
data2 = {
    'keyword':{
    '*': []
    }
}
t = [True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, False, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, False, True]
for index, qa in enumerate(data['keyword']['*']):
    if t[index] == True:
        data2['keyword']['*'].append({'questions': [qa['questions']], 'ans_txt': qa['ans_txt'], 'ans_wav': qa['ans_wav']})

with open('data/qa.json', 'w', encoding='utf-8') as f:
    json.dump(data2, f, indent=4, ensure_ascii=False)