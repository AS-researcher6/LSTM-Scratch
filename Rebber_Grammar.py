import os
import random

def create_data(sequenceNumber,testingNumber):
	trainingNumber = int(.8*sequenceNumber)
	remaining = sequenceNumber-trainingNumber
	validationNumber = int(remaining/2)
	trainingFile = open('trainingFile.txt', 'w')
	for index in range(trainingNumber):
		trainingFile.write(embedded_rebber_grammar() + '\n')
	trainingFile.close()
	validationFile = open('validationFile.txt','w')
	for index in range(validationNumber):
		validationFile.write(embedded_rebber_grammar() + '\n')
	validationFile.close()
	testingFile = open('testingFile.txt','w')
	for index in range(testingNumber):
		testingFile.write(embedded_rebber_grammar() + '\n')
	testingFile.close()

	# trainingList = [None]*trainingNumber
	# validationList = [None]*validationNumber
	# testingList = [None]*testingNumber

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

create_data(1000,20)


