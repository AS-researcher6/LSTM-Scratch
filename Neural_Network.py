import os
import random

exec(open("./LSTM_Cell.py").read())
exec(open("./Rebber_Grammar.py").read())


def initialize_neural_network(numCells,dataDim): #numCells is a list with number of cells for sequential layers. dataDim is the vector size of one input. Will always add an extra layer of dataDim size as output layer
	initialSeqLength = 1
	numLayers = len(numCells)+1
	global dataLength
	global totalCellNumber
	global ledgerLength
	global cellList	
	dataLength = dataDim
	totalCellNumber = sum(numCells)+dataDim
	ledgerLength = totalCellNumber+dataDim
	cellList = [None]*totalCellNumber
	cellId = 0
	for layerIndex in range(0,numLayers): #Using numCells and dataDim, initializes a network using LSTM_Cell and creates fully connected layers
		if layerIndex < numLayers-2:
			previousCellNumber = sum(numCells[0:layerIndex+1])
			outputIds = range(previousCellNumber,previousCellNumber+numCells[layerIndex+1])
		elif layerIndex == numLayers-2:
			previousCellNumber = sum(numCells[0:layerIndex+1])
			outputIds = range(previousCellNumber,previousCellNumber+dataDim)
		else:
			pass
		if layerIndex != numLayers-1:
			for x in range(0,numCells[layerIndex]):
				cellList[cellId] = LSTM_Cell(cellId,outputIds,initialSeqLength)
				cellId = cellId+1
		else:
			for x in range(0,dataDim):
				cellList[cellId] = LSTM_Cell(cellId,[cellId+dataDim],initialSeqLength)
				cellId = cellId+1
			for outputCellIndex in range(totalCellNumber-dataDim,totalCellNumber): #Last dataDim elements are just meant for storing output of final layer. Makes sure output weights are 1
				cellList[outputCellIndex].woMatrix = np.array([[1.0]])
	inputIds = range(0,numCells[0])
	global feederUnit
	feederUnit = Feeder(inputIds,dataDim)


fileName = "trainingFile.txt"
oneHotDic =  {"T":0,"B":1,"E":2,"P":3,"S":4,"X":5,"V":6,"A":7} #;)
def convert_data(rebberSequence): #T,B,E,P,S,X,V,(A)
	oneHot = np.array([[0.0]*len(rebberSequence)]*8)
	for sequenceIndex in range(0,len(rebberSequence)):
		oneHot[oneHotDic[rebberSequence[sequenceIndex]],sequenceIndex] = 1.0
	return(oneHot)

def reset_network(seqLength):
	for cell in cellList:
		cell.reset(seqLength)

def forward_propogation(inputSequence,sequenceLength):
	outputLedger = np.array([[0.0]*(ledgerLength-totalCellNumber)]*sequenceLength)
	for sequenceIndex in range(0,sequenceLength):
		forwardPropLedger = [0]*ledgerLength
		inputData = inputSequence[:,sequenceIndex]
		forwardPropLedger = feederUnit.forward_pass(inputData,forwardPropLedger)
		for cellIndex in range(0,totalCellNumber):
			forwardPropLedger = cellList[cellIndex].forward_pass(forwardPropLedger)
		for outputIndex in range(0,ledgerLength-totalCellNumber):
			outputLedger[sequenceIndex,outputIndex] = forwardPropLedger[totalCellNumber+outputIndex]
	return(outputLedger)


def calculate_loss(outputSequence,outputLedger,sequenceLength): #Also applies softmax
	for sequenceIndex in range(0,sequenceLength):
		trueOutput = outputSequence[:,sequenceIndex]
		networkOutput = softmax(outputLedger[sequenceIndex,:])
		#crossEntropy = cross_entropy(trueOutput,networkOutput)
		error = [None]*dataLength
		for item in range(dataLength):
			error[item] = -cross_entropy(trueOutput[item],networkOutput[item])
		totalError = 0
		for item in error:
			totalError = totalError + np.abs(item)
		for cellIndex in range(dataLength):
			backwardPropLedger[sequenceIndex,ledgerLength-dataLength+cellIndex] = error[cellIndex]
	#return(crossEntropy)
	return(totalError)

def backward_propogation(inputSequence):
	for cellIndex in range(len(cellList)-1,-1,-1):
		cellList[cellIndex].backward_pass()
	feederUnit.backward_pass(inputSequence)
	for outputCellIndex in range(totalCellNumber-dataLength,totalCellNumber):
		cellList[outputCellIndex].woMatrix = np.array([[1.0]])

def train_network(trainingEnd):
	file = open("trainingFile.txt",'r')
	trainingData = file.readlines() #Assuming I can store everything in memory
	file.close()
	end = False
	epochIndex = 0
	lineNumber = 0
	fileLength = len(trainingData)
	cycleIndex = 0
	cycleCheck = 10
	trainingLoss = -1
	validationLoss = -1
	while end==False:
		if lineNumber >= fileLength-2:
			lineNumber = 0
			epochIndex = epochIndex + 1
			print("Epoch Number ",epochIndex," Completed")
			if epochIndex == trainingEnd:
				end = True
		if cycleIndex >=cycleCheck:
			cycleCheck = 0
			print("Training Loss: ",trainingLoss, " Line Number: ", lineNumber+1)
		inputSequence = convert_data(trainingData[lineNumber][:-2])
		outputSequence = convert_data(trainingData[lineNumber][1:-1])
		sequenceLength = len(inputSequence[0])
		global backwardPropLedger
		backwardPropLedger = np.array([[0.0]*ledgerLength]*sequenceLength)
		reset_network(sequenceLength)
		outputLedger = forward_propogation(inputSequence,sequenceLength)
		trainingLoss = calculate_loss(outputSequence,outputLedger,sequenceLength)
		backward_propogation(inputSequence)
		lineNumber = lineNumber+1
		cycleIndex = cycleIndex + 1


def validate_network(epochIndex):
	totalLoss = 0
	file = open("validationFile.txt",'r')
	testingData = file.readlines() #Assuming I can store everything in memory
	file.close()
	for lineNumber in range(len(testingData)):
		textInput = testingData[lineNumber][:-2]
		inputSequence = convert_data(testingData[lineNumber][:-2])
		outputSequence = convert_data(testingData[lineNumber][1:-1])
		sequenceLength = len(inputSequence[0])
		global backwardPropLedger
		backwardPropLedger = np.array([[0.0]*ledgerLength]*sequenceLength)
		reset_network(sequenceLength)
		outputLedger = forward_propogation(inputSequence,sequenceLength)
		textOutput = ""
		for x in range(0,sequenceLength):
			outputVector = outputLedger[x,:]
		sequenceLoss = calculate_loss(outputSequence,outputLedger,sequenceLength)
		totalLoss = totalLoss + sequenceLoss
	print("Validation Loss for Epoch " + str(epochIndex) + ": " + str(totalLoss))

oneHotDic2 =  {0:"T",1:"B",2:"E",3:"P",4:"S",5:"X",6:"V",7:"A"}
def test_network():
	file = open("testingFile.txt",'r')
	testingData = file.readlines() #Assuming I can store everything in memory
	file.close()
	for lineNumber in range(len(testingData)):
		textInput = testingData[lineNumber][:-2]
		inputSequence = convert_data(testingData[lineNumber][:-2])
		outputSequence = convert_data(testingData[lineNumber][1:-1])
		sequenceLength = len(inputSequence[0])
		reset_network(sequenceLength)
		outputLedger = forward_propogation(inputSequence,sequenceLength)
		textOutput = ""
		for x in range(0,sequenceLength):
			outputVector = outputLedger[x,:]
			textOutput = textOutput + oneHotDic2[np.where(outputVector==max(outputVector))[0][0]]
		print("Test #" + str(lineNumber))
		print("This is the text input:")
		print(textInput)
		print("This is the text output:")
		print(textOutput)


initialize_neural_network([5,5,5],8)
train_network(1)
test_network()












