from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import model_from_json

def loadmodel():
	global classifier
	json_file = open('dogcatmodel.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	classifier = model_from_json(loaded_model_json)
	# load weights into new model
	classifier.load_weights('dogcatmodel.h5')
	print("Loaded model from disk")
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def uploadphoto():
	global filename
	global label1
	filename = str(tkinter.filedialog.askopenfile(parent=root,mode='rb',title='Upload a Photo'))
	file_path_index = filename.find('name')
	filename = filename[(file_path_index+5):len(filename)-1]
	filename = filename[1:len(filename)-1]
	print(filename)
	predict()

def predict():
	global label1
	global filename
	test_image = image.load_img(filename, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	#training_set.class_indices
	if result[0][0] == 1:
		prediction = "That's a dog!"
	else:
		prediction = "That's a cat!"
	label1.config(text=prediction)

def configure_window(x, title_text):
	x.geometry("750x500")
	x.title(title_text)
	x.resizable(False, False)
	x.configure(background="White")
	x.focus_force()

def root_launch():
	global root
	global labelvalue
	global uploadbutton
	global label1
	labelvalue = ""
	root = Tk()
	configure_window(root, "Dog Cat Predictor")
	uploadbutton = Button(root, text="Upload Photo", font = ("Arial", 12),
				width=15, height=3, bg="white", 
				activebackground="light grey", 
				bd="3", highlightbackground="black", 
				command = uploadphoto)
	uploadbutton.place(relx=0.5, rely=0.3, anchor=CENTER)
	label1 = Label(root, text=labelvalue, width=80, height=2,
                      font=('Arial',11))
	label1.place(relx=0.5, rely=0.7, anchor=CENTER)

root_launch()
loadmodel()
root.mainloop()
