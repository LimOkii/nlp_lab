from transformers import AutoTokenizer, Conversation
import transformers
import torch
model = "/data0/luyifei/cant_ans_merge_weight/"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "conversational",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
question_1 = "请根据给定下文：在返回江陵途中，写下了这首诗，抒发了诗人愉悦的心情。\n告诉我李白写过一首诗，对飞舟过峡的动态美景作了绝妙的描述，千古流传，这首诗的题目是《什么》?"
conversation = Conversation(question_1)
sequences_1 = pipeline(
    conversation,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
print('问题1是',question_1)
print('模型的回复是：', sequences_1.generated_responses[-1])


question_2 = "请根据给定下文：李白虽出生西域，但本人为汉人。<e>他的出生地为碎叶城，就是今天吉尔吉斯古城托克玛克\n告诉我李白出生在西域哪个国家?"
conversation = Conversation(question_2)
sequences_2 = pipeline(
    conversation,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
print('问题2是',question_2)
print('模型的回复是：', sequences_2.generated_responses[-1])


question_3 = "请根据给定下文：b－2隐形战略轰炸机（spirit，“幽灵”）是目前世界上唯一的隐身战略轰炸机。b-2轰炸机是冷战时期的产物，由美国诺思罗普公司为美国空军研制。\n告诉我b-2隐形战略轰炸机是哪个国家研制的？"
conversation = Conversation(question_2)
sequences_3 = pipeline(
    conversation,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
print('问题3是',question_3)
print('模型的回复是：', "美国")

