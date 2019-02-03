
predicted = open('files/spam_predicted.txt','r').read().split('\n')
main_dataset = open('files/word_dataset.csv','r').read().split('\n')
spam_count = 0
ham_count = 0
no_dup_spam = []
no_dup_predicted = []

for row in main_dataset:
	category = row.split(',')[0]
	if category == 'spam':
		if row not in no_dup_spam:
			no_dup_spam.append(row)

for row in predicted:
	if row not in no_dup_predicted:
		no_dup_predicted.append(row)

for row in no_dup_spam:
	if row in no_dup_predicted:
		spam_count += 1
	else:
		ham_count += 1

spam_percent = spam_count/len(no_dup_spam)
ham_percent = ham_count/len(no_dup_spam)
print('{}% spam predicted'.format(spam_percent))
print('{}% ham predicted as spam'.format(ham_percent))