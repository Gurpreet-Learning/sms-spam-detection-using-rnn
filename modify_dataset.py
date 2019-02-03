import random
dataset= open('files/word_dataset.csv','r').read().split('\n')
new_dataset = open('files/spam_dataset.csv','a+')
data_all = []
for data in dataset:
	if data.split(',')[0] == 'spam':
		for i in range(0,50):
			data_all.append(data)

	else:
		data_all.append(data)

random.shuffle(data_all)
new_dataset.write('category,summary\n')
for row in data_all:
	new_dataset.write('{}\n'.format(row))

new_dataset.close()
