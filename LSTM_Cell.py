import os
import random

import numpy as np
import math

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return sigmoid(values)*(1-sigmoid(values))

def tanh_derivative(values): 
    return 1. - np.tanh(values) ** 2

def rand_arr(a, b, *args): 
    #np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

def softmax(x): #Compute softmax of vector x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy(true,estimate):
	return(-1*np.dot(true,np.log(estimate)))

def loss_function(true,estimate): #Returns vector of error with respect to softmax input
	#return(-true/estimate)
	return(estimate-true)
	#return(-1*np.dot(true,np.log(estimate)))

# numCells = 10

# forwardPropLedger = [0]*numCells
# backwardPropLedger = np.array([[0.0]*numCells]*seqLength)

class LSTM_Cell:
	def __init__(self,id,outputIds,seqLength):
		self.wMatrix = rand_arr(-0.1,0.1,4,1) #[Wgx;Wix;Wfx;Wox] input weights
		self.uMatrix = rand_arr(-0.1,0.1,4,1) #[Wgh;Wih;Wfh;Woh] recurrent weights
		self.bMatrix = rand_arr(-0.1,0.1,4,1) #[Bg;Bi;Bf;Bo]
		self.id = id
		self.seqLength = seqLength
		self.stateMem = np.array([[0.0]*(seqLength+1)]*12) #[g;i;f;o;s;h;dg;di;df;do,ds,input] #Initializing Array to hold values for backpropogation. Earlier values are older time steps
		self.forwardIndex = 0
		self.outputIds = outputIds
		self.woMatrix = rand_arr(-0.1,0.1,len(outputIds),1)
		self.sPrev = 0 #sPrev and hPrev need to be initialized randomly and trained with backpropogation...later
		self.hPrev = 0
	def reset(self,seqLength): #Must be done after a forward_pass of a single sequence to prep for a different sequence of possibly different seqLength
		self.seqLength = seqLength
		self.sPrev = 0
		self.hPrev = 0
		self.forwardIndex = 0
		self.stateMem = np.array([[0.0]*(self.seqLength+1)]*12)
	def forward_pass(self,forwardPropLedger):
		totalInput = forwardPropLedger[self.id]
		stateArray = (self.wMatrix*totalInput)+(self.uMatrix*self.hPrev)+self.bMatrix
		self.g = np.tanh(stateArray[0,0])
		self.i = sigmoid(stateArray[1,0])
		self.f = sigmoid(stateArray[2,0])
		self.o = sigmoid(stateArray[3,0])
		self.s = (self.g*self.i) + (self.sPrev*self.f)
		self.h = np.tanh(self.s)*self.o
		self.stateMem[0,self.forwardIndex] = self.g
		self.stateMem[1,self.forwardIndex] = self.i
		self.stateMem[2,self.forwardIndex] = self.f
		self.stateMem[3,self.forwardIndex] = self.o
		self.stateMem[4,self.forwardIndex] = self.s
		self.stateMem[5,self.forwardIndex] = self.h
		self.stateMem[11,self.forwardIndex] = totalInput
		self.forwardIndex = self.forwardIndex+1
		self.hPrev = self.h
		self.sPrev = self.s
		self.outputs = self.h*self.woMatrix
		for idIndex in range(0,len(self.outputIds)):
			forwardPropLedger[self.outputIds[idIndex]] = (forwardPropLedger[self.outputIds[idIndex]] + self.outputs[idIndex])[0]
		return(forwardPropLedger)
	def backward_pass(self):
		for t in range(self.seqLength-1,-1,-1):
			delH = np.dot([[self.stateMem[6,t+1],self.stateMem[7,t+1],self.stateMem[8,t+1],self.stateMem[9,t+1]]],self.uMatrix)[0,0]
			delT = 0
			for idIndex in range(0,len(self.outputIds)):
				currentId = self.outputIds[idIndex]
				delT = delT + backwardPropLedger[t,currentId]*self.woMatrix[idIndex,0]
			dH = delT + delH
			dO = dH*np.tanh(self.stateMem[4,t])*self.stateMem[3,t]*(1-self.stateMem[3,t])
			dS = dH*self.stateMem[3,t]*tanh_derivative(self.stateMem[4,t])+(self.stateMem[10,t+1]*self.stateMem[2,t+1])
			dG = dS*self.stateMem[1,t]*(1-(self.stateMem[4,t]*self.stateMem[4,t]))
			dI = dS*self.stateMem[0,t]*self.stateMem[1,t]*(1-self.stateMem[1,t])
			if t>0:
				dF = dS*self.stateMem[4,t-1]*self.stateMem[2,t]*(1-self.stateMem[2,t])
			else:
				dF = 0
			self.stateMem[6,t] = dG
			self.stateMem[7,t] = dI
			self.stateMem[8,t] = dF
			self.stateMem[9,t] = dO
			self.stateMem[10,t] = dS
			dX = np.dot([[self.stateMem[6,t],self.stateMem[7,t],self.stateMem[8,t],self.stateMem[9,t]]],self.wMatrix)[0,0]
			backwardPropLedger[t,self.id] = dX
		dW = np.array([[0.0],[0.0],[0.0],[0.0]])
		dU = np.array([[0.0],[0.0],[0.0],[0.0]])
		dB = np.array([[0.0],[0.0],[0.0],[0.0]])
		for saveIndex in range(0,4):
			accessIndex = saveIndex+6
			for t in range(0,self.seqLength):
				dW[saveIndex,0] = dW[saveIndex,0] + (self.stateMem[accessIndex,t]*self.stateMem[11,t])
				dU[saveIndex,0] = dU[saveIndex,0] + (self.stateMem[accessIndex,t+1]*self.stateMem[5,t])
		for saveIndex in range(0,4):
			accessIndex = saveIndex+6
			for t in range(0,self.seqLength-1):
				dB[saveIndex,0] = dB[saveIndex,0] + (self.stateMem[accessIndex,t+1])
		dW = -0.05*dW/self.seqLength #Takes the average partial derivative and mulitplies by learning rate
		dU = -0.05*dU/self.seqLength
		dB = -0.05*dB/self.seqLength
		self.wMatrix = self.wMatrix-dW
		self.uMatrix = self.uMatrix-dU
		self.bMatrix = self.bMatrix-dB
		dWO = np.array([[0.0]]*len(self.outputIds))
		for idIndex in range(0,len(self.outputIds)):
				currentId = self.outputIds[idIndex]
				for t in range(0,self.seqLength):
					dWO[idIndex,0] = dWO[idIndex,0] + (backwardPropLedger[t,currentId]*self.stateMem[5,t])
		dWO = -0.05*dWO/self.seqLength
		self.woMatrix = self.woMatrix-dWO

class Feeder: ####NEED TO TEST
	def __init__(self,outputIds,dataLength):
		self.forwardIndex = 0
		self.outputIds = outputIds
		self.woMatrix = rand_arr(-0.1,0.1,len(outputIds),dataLength)
	def forward_pass(self,inputData,forwardPropLedger):
		outputIndex = 0
		for output in self.outputIds:
			forwardPropLedger[output] = forwardPropLedger[output] + np.dot(self.woMatrix[outputIndex,:],inputData)
			outputIndex = outputIndex + 1
		return(forwardPropLedger)
	def backward_pass(self,inputSequence):
		#Basically look at node deltas in backprop ledger and multiply by input data to find dW
		for outputIndex in self.outputIds:
			dWO = np.array([[0.0]*dataLength])
			for inputIndex in range(0,len(inputSequence[0])):
				dWO = dWO + backwardPropLedger[inputIndex,self.outputIds[outputIndex]]*inputSequence[:,inputIndex]
			dWO = -0.05*dWO/len(inputSequence[0])
			self.woMatrix[outputIndex,:] = self.woMatrix[outputIndex,:]-dWO

#self = LSTM_Cell(1,[0],[2],seqLength)

