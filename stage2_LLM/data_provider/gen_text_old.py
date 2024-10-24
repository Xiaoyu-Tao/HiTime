import pickle

prompts = {}
label_map = {}
target_texts = {}
split_texts = {}

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

# RS
label_map['RS'] = {}
for idx, label in enumerate(["Squash Forehand","Squash Backhand","Badminton Clear","Badminton Smash"]):
    label_map['RS'].update({idx:label})

prompts['RS'] = "When performing four different strokes: Squash Forehand, Squash Backhand, Badminton Clear, and Badminton Smash, sensors capture the individual's movements using a smart watch (Sony SmartWatch 3). The watch records the x, y, z coordinates from both the gyroscope and accelerometer, relayed to an Android phone (OnePlus 5). \nPlease choose one stroke from the previously mentioned four strokes and analyze the individual's performance based on the provided information. "
target_texts['RS'] = "The individual is currently performing "
split_texts['RS'] = "is currently performing "

# SRS1
label_map['SRS1'] = {}
for idx, label in enumerate(["Cortical Positivity","Cortical Negativity"]):
    label_map['SRS1'].update({idx:label})
    
prompts['SRS1'] = "When performing two different tasks: 'Cortical Positivity' and 'Cortical Negativity', EEG sensors capture the individual's brain signals. The data collected are from six EEG channels, with 896 samples per channel per trial, recorded at a sampling rate of 256 Hz over a 3.5-second interval. \nPlease choose one task from the previously mentioned two tasks and analyze the individual's performance based on the provided information."
target_texts['SRS1'] = "The individual is currently performing task "
split_texts['SRS1'] = "is currently performing task "

# prompts['BasicMotions'] = "There are four physical activities including standing, walking, running and playing badminton. \nPlease choose one activity from it.\nThe individual is currently engaged in "

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
			if useful_msage.startswith(label_str):
			# if label_str in useful_msage:
				label_num = idx
				break

		# if label_map[label_num] != gt:
		# 	print(useful_msage)
		# 	print(label_map[label_num], gt)
		# 	print('==================================')

		return label_num
