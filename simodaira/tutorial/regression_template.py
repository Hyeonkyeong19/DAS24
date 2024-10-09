# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
class linearRegression():
	#------------------------------------
	def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
		self.x = x
		self.y = y
		self.xDim = x.shape[0]
		self.dNum = x.shape[1]
		
		self.kernelType = kernelType
		self.kernelParam = kernelParam
	#------------------------------------

	#------------------------------------
	# 学習(For文)ここでは割愛
	def train(self):
		self.w = np.zeros([self.xDim,1])
	#------------------------------------

	#------------------------------------
	# 最小二乗法における重みの計算
	def trainMat(self):
		#self.x.shape #(N,200) (入力次元: 1, データ数：200)
		ones = np.ones((1,self.dNum))
		x_dash = np.concatenate([self.x, ones], axis=0) # バイアスも計算できるように説明変数に１を追加
		self.w_dash = np.matmul(np.linalg.inv(np.matmul(x_dash, x_dash.T)), np.sum((self.y * x_dash), axis=1))
	#------------------------------------
	
	#------------------------------------
	# 与えられた重みから予測値を求める
	def predict(self,x):
		return np.dot(self.w_dash[:-1], x) + self.w_dash[-1]
	#------------------------------------

	#------------------------------------
	# 損失関数
	def loss(self,x,y):
		loss = 0.0
		loss = np.sum(y - self.predict(x))
		return loss
	#------------------------------------
# �N���X�̒�`�I���
#-------------------

#-------------------
if __name__ == "__main__":
	
	# regresionData(学習データ数, 評価データ数, データ種類)
	myData = rg.artificial(200,100, dataType="1D")
	#myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	
	#regression = linearRegression(myData.xTrain,myData.yTrain)
	pdb.set_trace()
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	predict = regression.predict(myData.xTest)
	myData.plot(predict,isTrainPlot=False)
	
#-------------------
	