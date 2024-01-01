import json
import random

with open('me_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# print(data)
# 需要导出的数据格式
final_output_data = []

for each_ques in data.values():
    output_data = {}
    # print(each_ques)
    question = each_ques['question']  # 每一个总问题的question
    output_data['question'] = question
    ans_evidence = each_ques['evidences']
    for each_ans in ans_evidence.values():
        output_data['answer'] = each_ans['answer'][0]
        output_data['evidence'] = each_ans['evidence']
        final_output_data.append(output_data.copy())


# 无法回答时，模型给出的回答样例
cant_answer_template = [
    '抱歉，根据您所给的内容，我无法找到有关问题的答案',
    '给定的信息中似乎没有提到问题的答案',
    '根据提供的内容，我无法找到问题的相关信息',
    '根据您提供的上下文，我找不到与问题相关的答案',
    '给定的信息中似乎没有与问题有关的信息',
    '根据上述内容，我难以找到问题的解答',
    '据我所知，问题的答案不在提供的信息中',
    '根据上述信息，问题的答案似乎不可得',
    '给定的上下文似乎没有包含问题的答案',
    '给定的信息中似乎没有与问题有关的线索'
]


for item in final_output_data:
    if item['answer'] == 'no_answer':
        randon_index = random.randint(0, len(cant_answer_template) - 1)
        item['answer'] = cant_answer_template[randon_index]
    # print(item)

with open('llama_raw_data.json', 'w', encoding='utf-8') as file:
    json.dump(final_output_data, file, ensure_ascii=False, indent=4)


with open('llama_raw_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
'''
Stanford Alpaca 训练数据格式如下：
[
  {"instruction" : ...,
   "input" : ...,
   "output" : ...},
  ...
]
'''

cant_answer = []

for item in data:
    # print(index, item)
    output_data = {"instruction": "", "input": "", "output": ""}
    output_data['instruction'] = "请根据给定下文：" + item['evidence'] + '\n' + "告诉我" + item['question'] + '\n'
    output_data['input'] = ""
    output_data['output'] = item['answer']
    cant_answer.append(output_data.copy())

with open('train_data/cant_answer_llama_train_data.json', 'w', encoding='utf-8') as file:
    json.dump(cant_answer, file, ensure_ascii=False, indent=4)






