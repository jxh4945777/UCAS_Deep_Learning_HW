import json

input_file_path = './dataset/' #CCPC/ccpc_train_v1.0.json
output_file_path = './dataset/'
#Five: 19758    Seven: 89969


def json_process(input_file, output_file, poem_type):#preprocess_file 5/7
	poem_list = []
	poem_type_char = {5: "five", 7: "seven"}
	txt_poem_file = open(output_file, "a+", encoding= "utf-8")
	for each_data in open(input_file, encoding="utf-8"):
		json_data = json.loads(each_data)
		data_content = json_data['content']
		if len(data_content) == (poem_type * 4 + 3):
			poem_list.append(data_content)
	for item in poem_list:
		item = item.replace("|","，")
		txt_poem_file.write("<" + item + "。>" + "\n")
	print("Num_of_" + poem_type_char[poem_type] + "poems: " + str(len(poem_list)) + "\n")

def init_index():
	index_file = open("./dataset/char_index.txt", "w", encoding = "utf-8")
	list_of_char = ['<', '>', '，', '。']
	index_file.write("0 <\n")
	index_file.write("1 >\n")
	index_file.write("2 ，\n")
	index_file.write("3 。\n")
	num = 4
	for poem_line in open("./dataset/Five_Poem_Data.txt", "r", encoding = "utf-8"):
		poem_line = poem_line.replace("\n","")
		for poem_char in poem_line:
			if poem_char not in list_of_char:
				list_of_char.append(poem_char)
				index_file.write(str(num) + " " + poem_char + "\n")
				num += 1
	for poem_line in open("./dataset/Seven_Poem_Data.txt", "r", encoding = "utf-8"):
		poem_line = poem_line.replace("\n","")
		for poem_char in poem_line:
			if poem_char not in list_of_char:
				list_of_char.append(poem_char)
				index_file.write(str(num) + " " + poem_char + "\n")
				num += 1


# #生成五言和七言古诗数据集
json_process(input_file_path + 'CCPC/ccpc_train_v1.0.json', output_file_path + 'Seven_Poem_Data.txt', 7)
json_process(input_file_path + 'CCPC/ccpc_test_v1.0.json', output_file_path + 'Seven_Poem_Data.txt', 7)
json_process(input_file_path + 'CCPC/ccpc_valid_v1.0.json', output_file_path + 'Seven_Poem_Data.txt', 7)

json_process(input_file_path + 'CCPC/ccpc_train_v1.0.json', output_file_path + 'Five_Poem_Data.txt', 5)
json_process(input_file_path + 'CCPC/ccpc_test_v1.0.json', output_file_path + 'Five_Poem_Data.txt', 5)
json_process(input_file_path + 'CCPC/ccpc_valid_v1.0.json', output_file_path + 'Five_Poem_Data.txt', 5)


#生成id与word对照
init_index()