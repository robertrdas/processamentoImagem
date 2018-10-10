import numpy as np 
import cv2
import math as mt

def printM(mat):
	for i in range(mat.shape[0]):
		for j in range (mat.shape[1]):
			print(mat[i,j])
		
def algoritimo_basico(img):

	linhas,colunas = img.shape

	limiar = int((0+255)/2)
	
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(linhas):
		for j in range(colunas):
			if img[i,j] >= limiar:
				imagem_resultado[i,j] = 255

	return imagem_resultado

def modulacao_aleatoria(img):

	linhas,colunas = img.shape

	limiar = int((0+255)/2)

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(linhas):
		for j in range(colunas):
			temp = img[i,j] + np.random.randint(-127,128)

			if temp >= limiar:
				imagem_resultado[i,j] = 255

	return imagem_resultado

def aglomeracao_pontos(img, nMat):
	

	linhas,colunas = img.shape
	if nMat == 1:
		matD = np.array([1, 7, 4, 5, 8, 3, 6, 2, 9]).reshape((3,3))
		n = 3
	else:
		matD = np.array([8, 3, 4, 6, 1, 2, 7, 5, 9]).reshape((3,3))
		n = 3

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for x in range(linhas):
		for y in range(colunas):
			i = x % n
			j = y % n
			vlrNimagem = ((img[x,y]*1.0)/255)
			vlrNmat = ((matD[i,j]*1.0)/10)

			if vlrNimagem > vlrNmat:
				imagem_resultado[x,y] = 255
	
	return imagem_resultado

def dispersao_pontos(img, nMat):

	linhas,colunas = img.shape
	if nMat== 1:
		matD = np.array([2, 3, 4, 1]).reshape((2,2))
		n = 2
	else:
		matD = np.array([2, 16, 3, 13, 10, 6, 11, 7, 4, 14, 1, 15, 12, 8, 9, 6]).reshape((4,4))
		n = 4

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for x in range(linhas):
		for y in range(colunas):
			i = x % n
			j = y % n
			coef = mt.pow(n,2)+1
			temp1 = ((img[x,y]*1.0)/255)
			temp2 = ((matD[i,j]*1.0)/coef)

			if temp1 > temp2:
				imagem_resultado[x,y] = 255

	return imagem_resultado

def floyd_steinberg(img):
	img = np.float32(img)
	linhas,colunas = img.shape

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for x in range(linhas):
		for y in range(colunas):
			if img[x,y] >= 128:
				imagem_resultado[x,y] = 255

			erro = img[x,y] - imagem_resultado[x,y]

			if x + 1 < linhas:
				img[x+1,y] = img[x+1,y] + erro * 7/16
			if y+1 < colunas:
				img[x,y+1] = img[x,y+1] + erro *  5/16
				if x + 1 < linhas:
					img[x+1,y+1] = img[x+1,y+1] + erro *  1/16
				if x-1 > 0 :
					img[x-1,y+1] = img[x-1,y+1] + erro *  3/16


	return imagem_resultado

def rum_dithering(algoritmo):

	img = cv2.imread('./imagens/lena_original.jpg',0)


	if algoritmo == 'basico':
		resultado = algoritimo_basico(img.copy())
	elif algoritmo == 'modulacao_aleatoria':
		resultado = modulacao_aleatoria(img.copy())
	elif algoritmo == 'aglomeracao':
		resultado = aglomeracao_pontos(img.copy(),1)
	elif algoritmo == 'dispersao':
		resultado = dispersao_pontos(img.copy(),1)
	elif algoritmo == 'floyd':
		resultado = floyd_steinberg(img.copy())
	else:
		return None

	cv2.imshow('ImagemEntrada',img)

	titulo = 'algoritmo {tituloJanela}'.format(tituloJanela = algoritmo)
	cv2.imshow(titulo,resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

