import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import math
import random
import os
import torch, torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import os.path
import shutil
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.nn.functional as func


def sp_noise(image, prob):
	output = np.zeros(image.shape,np.uint8)
	thres = 1 - prob
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			rdn = random.random()
			if rdn < prob:
				output[i][j] = 0
			elif rdn > thres:
				output[i][j] = 255
			else:
				output[i][j] = image[i][j]
	return output

def check_path(kanjiname):
	pathname = "E:\\MiKanji\\Images\\"+kanjiname
	try:
		os.mkdir(pathname)
	except FileExistsError:
		pass
	return pathname



class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, 10)
		self.conv2 = nn.Conv2d(16, 48, 8)
		self.conv3 = nn.Conv2d(48, 64, 5)
		self.conv4 = nn.Conv2d(64, 256, 2)
		#self.conv4 = nn.Conv2d(16, 12, 5)
		#self.dropout1 = nn.Dropout2d(0.25)
		#self.dropout2 = nn.Dropout2d(0.5)

		self.fc1 = nn.Linear(4096, 3000)
		self.fc2 = nn.Linear(3000, 2337)
		#self.fc3 = nn.Softmax(2337)

	def forward(self, x):
		x = self.conv1(torch.nn.functional.max_pool2d(x, 2))
		x = torch.nn.functional.relu(self.conv2(x))
		#x = torch.nn.functional.relu(self.conv3(x))
		x = torch.nn.functional.relu(self.conv3(x))
		x = torch.nn.functional.relu(self.conv4(x))
		#x = self.dropout1(x)
		x = torch.flatten(x, 1)
		
		x = torch.nn.functional.relu(self.fc1(x))
		#x = self.dropout2(x)
		x = self.fc2(x)
		
		#x = torch.nn.functional.relu(self.fc2(x))
		x = torch.nn.functional.log_softmax(x, dim=1)

		return x


if __name__ == "__main__":

	# Read the Kanji file with
	kanji = open("joyo.txt", "r", encoding="utf8")
	kanjilist = kanji.readline()
	kanji.close()

	createDataset = False
	randomGray = False
	makeTest = False
	makeTrain = False

	if createDataset:
		for i in range(0, len(kanjilist)):
			for j in range(0, 100):
				filename = kanjilist[i] + str(j) + ".png"

				# Use the cinecaption font size 40
				fnt = ImageFont.truetype('cinecaption226.ttf', 40)

				# Create 50x50 images with random color backgrounds
				image = Image.new(mode="RGB", size=(50, 50), color="rgb("+str(random.randint(0, 255))+","+str(random.randint(0, 255))+","+str(random.randint(0, 255))+")")

				draw = ImageDraw.Draw(image)

				# Draw each Kanji on the image
				draw.text((5, 5), kanjilist[i], font=fnt, fill=(255,255,255,255))

				# Save the file
				image.save("E:\\MiKanji\\Images\\"+filename)


	i = 0
	if randomGray:
		if makeTrain:
			for filename in os.listdir("E:\\MiKanji\\Images\\"):
				if filename.endswith(".png"):
					# print(filename)
					file = open("E:\\MiKanji\\Images\\"+filename, "rb")
					bytes = bytearray(file.read())
					numpyarray = np.asarray(bytes, dtype=np.uint8)
					img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
					noisy = sp_noise(img, 0.01)
					success, buffer = cv2.imencode(".png", noisy)
					buffer.tofile("E:\\MiKanji\\Images\\Grays\\"+filename)
					i += 1
					if i % 1000 == 0:
						print("grayscale# "+str(i))
					file.close()
		if makeTest:
			i = 0
			for filename in os.listdir("E:\\MiKanji\\Images\\"):
				if filename.endswith(".png"):
					if i % 10 == 0:
						# print(filename)
						file = open("E:\\MiKanji\\Images\\"+filename, "rb")
						bytes = bytearray(file.read())
						numpyarray = np.asarray(bytes, dtype=np.uint8)
						img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
						noisy = sp_noise(img, 0.01)
						success, buffer = cv2.imencode(".png", noisy)
						buffer.tofile("E:\\MiKanji\\Images\\Test\\"+filename)
						file.close()
					if i % 1000 == 0:
						print("grayscale# "+str(i))
					i += 1

	# Use this to add each image to a class folder, needed for Pytorch ImageFolder
	ImageSort = False
	if ImageSort:
		for filename in os.listdir("E:\\MiKanji\\Images\\Test"):
			shutil.move("E:\\MiKanji\\Images\\Test\\"+filename, check_path("Test\\"+filename[0]))
		for filename in os.listdir("E:\\MiKanji\\Images\\Grays"):
			shutil.move("E:\\MiKanji\\Images\\Grays\\"+filename, check_path("Grays\\"+filename[0]))

	
	training = True
	if training:
		transform_settings = transforms.Compose([
			transforms.Grayscale(),
			transforms.ToTensor()
		])

		train_data = torchvision.datasets.ImageFolder("E:\\MiKanji\\Images\\Grays", transform=transform_settings)
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=5)
		test_data = torchvision.datasets.ImageFolder("E:\\MiKanji\\Images\\Test", transform=transform_settings)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=5)

		device = torch.device("cuda:0")

		print(len(train_data))
		print(len(test_data))

		network = CNN().to(device)
		optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
		J = torch.nn.CrossEntropyLoss()
		network.train()
		losses_001 = np.array([])
		losses_003 = np.array([])
		losses_01 = np.array([])
		losses_03 = np.array([])
		losses_1 = np.array([])
		learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
		losses = [losses_1, losses_03, losses_01, losses_003, losses_001]
		
		for epoch in range(0,5):
			optimizer = torch.optim.Adam(network.parameters(), lr=learning_rates[epoch])
			for i, data in enumerate(train_loader):
				inputs, labels = data[0].to(device), data[1].to(device)
				#inputs, labels = torch.autograd.Variable(inputs).to(device), torch.autograd.Variable(labels).to(device)
				
				optimizer.zero_grad()

				outputs = network(inputs)
				
				loss = J(outputs, labels)
				loss.backward()
				optimizer.step()
				if (i % 1000 == 0):
					losses[epoch] = np.append(losses[epoch], loss.cpu().detach().numpy())
					#np.savetxt(str(learning_rates[epoch])+'.txt', losses[epoch])
					#print(losses[epoch])
				print("Epoch: ", epoch, "Loss: ", loss.cpu().detach().numpy())

				
			torch.save(network.state_dict(), "Model\\model.cnn")
	
		for epoch in range(0,5):
			np.savetxt(str(learning_rates[epoch])+'.txt', losses[epoch])
		

		print("Training complete")

	























