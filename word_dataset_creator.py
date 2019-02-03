import re

class dataset_creator:

	def __init__(self):
		self.stop_words = self.load_stopwords()
		self.all_words = []

	def load_stopwords(self):
		file = open('files/stop-word-list.csv','r')
		return file.read().split('\n')

	def write_word_dataset(self,sentence_total):

		word_dataset = open('files/word_dataset.csv','a+')
		word_dataset.write('category,sentence\n')
		for sentence_compiled in sentence_total:
			category = sentence_compiled[0]
			setence = sentence_compiled[1]
			# if category == 'ham':
			# 	word_dataset.write('{},'.format(str(0)))
			# elif category == 'spam':
			word_dataset.write('{},'.format(category))
			for word in setence:
				word_dataset.write('{} '.format(str(word)))

			word_dataset.write('\n')



	def extract_sentence(self):
		sentence_total = []
		sms_lines = self.load_sms_lines()

		for line in sms_lines:
			if len(line) != 0:
				category,pre_words = self.extract_words(line)
			if len(pre_words) != 0:
				sentence_total.append([category,pre_words[0:len(pre_words)]])

		self.write_word_dataset(sentence_total)

	def load_sms_lines(self):
		data = open('files/SMSSpamCollection.txt','r')
		return data.read().split('\n')

	def extract_words(self,summary):
		symbols = [	'`','~','!','@','#','$','%','^','&','*','(',')','_','-',
					'+','=','{','[','}','}','|','\',<',',','>','.','?','/',
					',',"'",'``','\\\\','--','1','2','3','4','5','6','7','8'
					,'9','0','\\',"''","'''",'\n'
				]
		word_list = []
		string_filter = r'[^\w\s]*'
		summary = summary.split('\t')

		category = summary[0]
		words = summary[1].split(' ')
		new_word_list = []

		for word in words:
			word = word.lower()
			word = re.sub(string_filter,'',word)

			if word not in symbols:
				word = re.sub('[^a-zA-z0-9\s]','',word)
				word_list.append(word)

		return category,word_list

	def convert_to_number(self,s):
		return int.from_bytes(s.encode(), 'little')


dataset_creator = dataset_creator()
dataset_creator.extract_sentence()
print('Done.')