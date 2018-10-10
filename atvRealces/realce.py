import numpy as np
import cv2
import math as mt

def negativo(img):
	linhas,colunas = img.shape

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for x in range(linhas):
		for y in range(colunas):
			imagem_resultado[x,y] = 255 - img[x,y]

	return imagem_resultado

def normaliza_contraste(img,c,d):
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	a = img.min()

	b = img.max()

	for x in range(linhas):
		for y in range(colunas):
			imagem_resultado[x,y] = (img[x,y]-a)*((d-c)/(b-a))+c

	return imagem_resultado

def correcao_gama(img,gama):
	constante  = 1
	linhas,colunas = img.shape
	
	img = img/255

	for x in range(linhas):
		for y in range(colunas):

			img[x,y] = (constante*img[x,y])**gama

	return np.uint8(img * 255)

def realce_linear(img,D,G):
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)


	for x in range(linhas):
		for y in range(colunas):
			imagem_resultado[x,y] = novo_valor = (G * img[x,y]) + D

	return imagem_resultado

def realce_log(img):
	G = 105.9612
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)


	for x in range(linhas):
		for y in range(colunas):
			imagem_resultado[x,y] = G * mt.log10(img[x,y]+1)
			
	return imagem_resultado

def realce_quad(img):
	G = 1/255
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)


	for x in range(linhas):
		for y in range(colunas):
			imagem_resultado[x,y] = G * (img[x,y]**2)
			
	return imagem_resultado

def realce_sqrt(img):
	G = 15.9687
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)


	for x in range(linhas):
		for y in range(colunas):
			imagem_resultado[x,y] = G * mt.sqrt(img[x,y])
			
	return imagem_resultado



def rum_realce(metodo):
	img = cv2.imread('./imagens/lena.jpg',0)

	if metodo == 'negativo':
		resultado = negativo(img)
	elif metodo == 'normalizacao':
		resultado = normaliza_contraste(img,100,255)
	elif metodo == 'gama':
		resultado = correcao_gama(img.copy(), 1.2)
	elif metodo == 'linear':
		resultado = realce_linear(img.copy(),0,2)
	elif metodo == 'log':
		resultado = realce_log(img.copy())
	elif metodo == 'quad':
		resultado = realce_quad(img.copy())
	elif metodo == 'raiz':
		resultado = realce_sqrt(img.copy())
	else:
		return None

	cv2.imshow('ImagemEntrada',img)

	titulo = '{tituloJanela}'.format(tituloJanela = metodo)
	cv2.imshow(titulo,resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

