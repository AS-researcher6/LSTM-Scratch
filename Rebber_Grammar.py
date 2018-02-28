import os
import random

def create_data(sequenceNumber):
	trainingNumber = int(.8*sequenceNumber)
	remaining = sequenceNumber-trainingNumber
	validationNumber = int(remaining/2)
	testingNumber = sequenceNumber-trainingNumber-validationNumber
	trainingFile = open('trainingFile.txt', 'w')
	for index in range(trainingNumber):
		print>>trainingFile, embedded_rebber_grammar()
	trainingFile.close()
	validationFile = open('validationFile.txt','w')
	for index in range(validationNumber):
		print>>validationFile, embedded_rebber_grammar()
	validationFile.close()
	testingFile = open('testingFile.txt','w')
	for index in range(testingNumber):
		print>>testingFile, embedded_rebber_grammar()
	testingFile.close()

	trainingList = [None]*trainingNumber
	validationList = [None]*validationNumber
	testingList = [None]*testingNumber

def embedded_rebber_grammar():
	sequence = "B"
	if random.random()>=0.5:
		sequence += "T"
		sequence += "B"
		sequence = state_start(sequence)
		sequence += "T"
		sequence += "E"
	else:
		sequence += "P"
		sequence += "B"
		sequence = state_start(sequence)
		sequence += "P"
		sequence += "E"
	sequence = "A" + sequence + "A" #Padding with start and end indicators
	return(sequence)

def state_start(sequence):
	if random.random()>=0.5:
		sequence += "T"
		sequence = state_t1(sequence)
	else:
		sequence += "P"
		sequence = state_b1(sequence)
	return(sequence)

def state_t1(sequence):
	if random.random()>=0.5:
		sequence += "S"
		sequence = state_t1(sequence)
	else:
		sequence += "X"
		sequence = state_t2(sequence)
	return(sequence)

def state_t2(sequence):
	if random.random()>=0.5:
		sequence += "S"
		sequence = state_end(sequence)
	else:
		sequence += "X"
		sequence = state_b1(sequence)
	return(sequence)

def state_b1(sequence):
	if random.random()>=0.5:
		sequence += "T"
		sequence = state_b1(sequence)
	else:
		sequence += "V"
		sequence = state_b2(sequence)
	return(sequence)

def state_b2(sequence):
	if random.random()>=0.5:
		sequence += "V"
		sequence = state_end(sequence)
	else:
		sequence += "P"
		sequence = state_t2(sequence)
	return(sequence)

def state_end(sequence):
	sequence += "E"
	return(sequence)




