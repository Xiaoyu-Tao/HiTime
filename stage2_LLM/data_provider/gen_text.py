import pickle

prompts = {}
label_map = {}
target_texts = {}
split_texts = {}
			
label_map['CR'] = {}
for idx, label in enumerate(["a", "b", "c", "d", "e", "g", "h", "l", "m", "n", "o", "p", "q", "r", "s", "u", "v", "w", "y", "z"]):
    label_map['CR'].update({idx: label})

prompts['CR'] = "The dataset consists of 20 different characters captured using a WACOM tablet: a, b, c, d, e, g, h, l, m, n, o, p, q, r, s, u, v, w, y, z. \nPlease choose one character from the previously mentioned options based on the provided information. "
target_texts['CR'] = "The character being analyzed is "
split_texts['CR'] = "being analyzed is "
   
# CT
label_map['CT'] = {}
for idx, label in enumerate(['a','b','c','d','e','g','h','l','m','n','o','p','q','r','s','u','v','w','y','z']):
    label_map['CT'].update({idx:label})

prompts['CT'] = "When pronouncing 20 letters: a; b; c; d; e; g; h; l; m; n; o; p; q; r; s; u; v; w; y; z, the tablet captures the individual's facial expression. \nPlease choose one letter from the previously mentioned 20 letters and analyze the individual's pronounce based on the provided information. "
target_texts['CT'] = "The individual is currently pronouncing letter "
split_texts['CT'] = "is currently pronouncing letter "

# EP
label_map['EP'] = {}
for idx, label in enumerate(["WALKING", "RUNNING", "SAWING", "SEIZURE MIMICKING"]):
    label_map['EP'].update({idx:label})

prompts['EP'] = "When conducting 4 activities: WALKING (various paces and gestures), RUNNING (40-meter corridor), SAWING (30 seconds), and SEIZURE MIMICKING (with a 30-second seizure), the accelerometer captures the individual's movements.  \nPlease choose one activity from the aforementioned four options and analyze the individual's movements based on the provided information. "
target_texts['EP'] = "The individual is currently performing activity "
split_texts['EP'] = "is currently performing activity "


# ER
label_map['ER'] = {}
for idx, label in enumerate(["hand open","fist","two","pointing","ring","grasp"]):
    label_map['ER'].update({idx:label})

prompts['ER'] = "Hand gestures such as hand open, fist, two, pointing, ring, and grasp are recorded using the eRing prototype ring. eRing detects hand and finger gestures through electric field sensing. \nPlease choose one gesture from the aforementioned six options and analyze the individual's hand gesture based on the provided information. "
target_texts['ER'] = "The individual is currently performing "
split_texts['ER'] = "is currently performing "

# FD
label_map['FD'] = {}
for idx, label in enumerate(["face","scramble"]):
    label_map['FD'].update({idx:label})

prompts['FD'] = "When looking at two kinds of photo, face and scramble, the researchers recorded the electrical activity in the observers' brains. \nPlease choose one kind of photo from the previously mentioned two options based on the provided information. "
target_texts['FD'] = "The observer is currently looking at "
split_texts['FD'] = "is currently looking at "

# HB
label_map['HB'] = {}
for idx, label in enumerate(["normal","abnormal"]):
    label_map['HB'].update({idx:label})

prompts['HB'] = "Heart sounds are collected from various parts of the body, including typical areas such as the aorta, pulmonary artery, tricuspid valve, and mitral valve. The sounds are divided into two categories: normal and abnormal. \nPlease choose one category from the previously mentioned two options and analyze the individual's heart sound based on the provided information. "
target_texts['HB'] = "The recording is currently classified as "
split_texts['HB'] = "is currently classified as "

# JV
label_map['JV'] = {}
for idx, label in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
    label_map['JV'].update({idx: label})

prompts['JV'] = "Nine Japanese-male speakers were recorded saying the vowels 'a' and 'e'.  \nPlease choose one speaker from the previously mentioned nine options based on the provided information. "
target_texts['JV'] = "The utterance is currently classified as "
split_texts['JV'] = "is currently classified as "

# LI
label_map['LI'] = {}
for idx, label in enumerate([str(i) for i in range(1, 16)]):
    label_map['LI'].update({idx: label})

prompts['LI'] = "When analyzing 15 types of hand movements in LIBRAS, the researchers recorded the bi-dimensional curves of the hand movements from videos. \nPlease choose one hand movement type from the previously mentioned fifteen options based on the provided information."
target_texts['LI'] = "The hand movement is currently classified as "
split_texts['LI'] = "is currently classified as "

# NATOPS
label_map['NATOPS'] = {}
for idx, label in enumerate(["I have command", "All clear", "Not clear", "Spread wings", "Fold wings", "Lock wings"]):
    label_map['NATOPS'].update({idx: label})

prompts['NATOPS'] = "When performing six different actions: 'I have command', 'All clear', 'Not clear', 'Spread wings', 'Fold wings', and 'Lock wings', sensors capture the individual's movements from the hands, elbows, wrists, and thumbs. \nPlease choose one motion from the previously mentioned six options based on the provided information."
target_texts['NATOPS'] = "The motion is currently classified as "
split_texts['NATOPS'] = "is currently classified as "

# PD
label_map['PD'] = {}
for idx, label in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]):
    label_map['PD'].update({idx: label})

prompts['PD'] = "When the individual is writing ten digits: zero, one, two, three, four, five, six, seven, eight and nine, the pen-tip across the screen would be traced. \nPlease choose one kind of digit from the previously mentioned ten options based on the provided information."
target_texts['PD'] = "The digit is currently classified as "
split_texts['PD'] = "is currently classified as "

# PEMS-SF
label_map['PEMS_SF'] = {}
for idx, label in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]):
    label_map['PEMS_SF'].update({idx: label})

prompts['PEMS_SF'] = "When analyzing daily occupancy rates of different car lanes on San Francisco bay area freeways, the data is recorded from sensors every 10 minutes. The task is to classify each observed day as the correct day of the week, from Monday to Sunday. \nPlease choose one day of the week from the previously mentioned seven options based on the provided information."
target_texts['PEMS_SF'] = "The day is currently classified as "
split_texts['PEMS_SF'] = "is currently classified as "

# RS
label_map['RS'] = {}
for idx, label in enumerate(["Squash Forehand","Squash Backhand","Badminton Clear","Badminton Smash"]):
    label_map['RS'].update({idx:label})

prompts['RS'] = "When performing four different strokes: Squash Forehand, Squash Backhand, Badminton Clear, and Badminton Smash, sensors capture the individual's movements using a smart watch (Sony SmartWatch 3). The watch records the x, y, z coordinates from both the gyroscope and accelerometer, relayed to an Android phone (OnePlus 5). \nPlease choose one stroke from the previously mentioned four strokes and analyze the individual's performance based on the provided information. "
target_texts['RS'] = "The individual is currently performing "
split_texts['RS'] = "is currently performing "

# SAD
label_map['SAD'] = {}
for idx, label in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]):
    label_map['SAD'].update({idx: label})

prompts['SAD'] = "When the individual is pronouncing ten digits: zero, one, two, three, four, five, six, seven, eight and nine, the sound would be recorded. \nPlease choose one kind of digit from the previously mentioned ten options based on the provided information. "
target_texts['SAD'] = "The digit is currently classified as "
split_texts['SAD'] = "is currently classified as "

# SRS1
label_map['SRS1'] = {}
for idx, label in enumerate(["Cortical Positivity","Cortical Negativity"]):
    label_map['SRS1'].update({idx:label})
    
prompts['SRS1'] = "When performing two different tasks: 'Cortical Positivity' and 'Cortical Negativity', EEG sensors capture the individual's brain signals. The data collected are from six EEG channels, with 896 samples per channel per trial, recorded at a sampling rate of 256 Hz over a 3.5-second interval. \nPlease choose one task from the previously mentioned two tasks and analyze the individual's performance based on the provided information."
target_texts['SRS1'] = "The individual is currently performing task "
split_texts['SRS1'] = "is currently performing task "

# SRS2
label_map['SRS2'] = {}
for idx, label in enumerate(["Cortical Positivity", "Cortical Negativity"]):
    label_map['SRS2'].update({idx: label})

prompts['SRS2'] = "When performing two different tasks: 'Cortical Positivity' and 'Cortical Negativity', EEG sensors capture the brain signals of an artificially respirated ALS patient. \nPlease choose one task from the previously mentioned two tasks and analyze the individual's performance based on the provided information."
target_texts['SRS2'] = "The individual is currently performing task "
split_texts['SRS2'] = "is currently performing task "

# UWG
label_map['UWG'] = {}
for idx, label in enumerate([str(i) for i in range(1, 9)]):
    label_map['UWG'].update({idx: label})

prompts['UWG'] = "When performing eight simple gestures, accelerometers capture the individual's movements. The data consists of the x, y, z coordinates of each motion, with each series being 315 data points long. \nPlease choose one gesture from the previously mentioned eight gestures and analyze the individual's performance based on the provided information."
target_texts['UWG'] = "The individual is currently performing gesture "
split_texts['UWG'] = "is currently performing gesture "

def get_prompt(datasetname):
	prompt = prompts[datasetname]
	return prompt

class TextGenerator():
	def __init__(self, datasetname):
		self.dataset = datasetname
		self.label_map = label_map[datasetname]
		self.prompt = prompts[datasetname]
		self.target_text = target_texts[datasetname]
		self.split_text = split_texts[datasetname]
  
		print(f'Probably length of {datasetname} prompt: ', len(self.prompt.split(' ')))
		
	def get_str_from_label(self, label):
		return self.label_map[label]
  
	def get_prompt_and_target(self, label):
		prompt = self.prompt
		target = self.target_text + self.label_map[label.item()] + '.'
		return prompt, target
  
	def extract_key_labels(self, text):
		try:
			useful_msage = text.split(self.split_text, 1)[-1]
		except:
			raise ValueError(f"No such dataset@ {self.dataset}")
			
			# print(sequence)
			# print('==================================')
			
		label_num = -1
		for idx, label_str in self.label_map.items():
			if useful_msage.lower().startswith(label_str.lower()):
			# if label_str in useful_msage:
				label_num = idx
				break

		# if label_map[label_num] != gt:
		# 	print(useful_msage)
		# 	print(label_map[label_num], gt)
		# 	print('==================================')

		return label_num
