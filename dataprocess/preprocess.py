import nltk
import string
import contractions
import n2w
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

def expand_contractions(text):
	expanded_words = []   
	for word in text.split():
		expanded_words.append(contractions.fix(word))  
		
	expanded_text = ' '.join(expanded_words)
	return " ".join(expanded_words)

def remove_punctuation(text):
	text = text.translate(str.maketrans('','',string.punctuation))
	return text

def lower_case(text):
	return text.lower()

def remove_stop_words(text):
	stop_words = set(stopwords.words('english'))
	
	word_tokens = word_tokenize(text)
	
	filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
	
	filtered_sentence = []
	
	for w in word_tokens:
		if w not in stop_words:
			filtered_sentence.append(w)

	return " ".join(filtered_sentence)

def num2word(text):
	for i in text.split():
		if i.isdigit():
			text = text.replace(i, n2w.convert(i))
	return text

def preprocessing(text,num):

	lower_case_true = (num & 0b0001) != 0
	expand_con_true = (num & 0b0010) != 0
	remove_pun_true = (num & 0b0100) != 0
	remove_s_w_true = (num & 0b1000) != 0

	if lower_case_true:
		text = num2word(text)
		text = lower_case(text)
	if expand_con_true:
		text = expand_contractions(text)
	if remove_pun_true:
		text = remove_punctuation(text)
	if remove_s_w_true:
		text = remove_stop_words(text)

	return text
