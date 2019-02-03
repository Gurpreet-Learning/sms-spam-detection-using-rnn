import random

class convert_numerical:

	def __init__(self):
		self.word_list = []

	def load_dataset_vals(self):

		data = []
		dataset_file = open('files/word_dataset.txt','r')
		dataset_raw = dataset_file.read().split('\n')

		for data_line in dataset_raw:
			word_list = data_line.split(',')
			if len(word_list) != 1:
				word_list = word_list[1].split(' ')
				data.append(word_list)

		random.shuffle(data)
		return data

	def convert_numeric(self):

		dataset_raw = self.load_dataset_vals()
		for row in dataset_raw:
			for word in row:
				if word not in self.word_list:
					self.word_list.append(word)


convert = convert_numerical()
convert.convert_numeric()