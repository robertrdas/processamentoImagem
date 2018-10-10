import numpy as np
import cv2
import math as mt
import matplotlib.pyplot as plt

#função que gera o histograma dado uma imagem
def gera_histograma(img):
	linhas,colunas = img.shape

	histograma = [0] * 256

	for i in range(linhas):
		for j in range(colunas):
			histograma[img[i,j]] = histograma[img[i,j]] + 1

	return histograma

#função que ger um histograma acumulado, dado um histograma
def histograma_acumulado(histograma_):
	linhas,colunas = img.shape

	histograma_acumulado = [0]*256

	histograma_acumulado[0] = histograma_[0]
	for i in range(1,len(histograma_)):
		histograma_acumulado[i] = histograma_[i] + histograma_acumulado[i-1]

	return histograma_acumulado

#função que gera um histograma equalizado e equaliza a imagem.
def histograma_equalizado(img,histograma_acumulado_):
	linhas,colunas = img.shape

	fator  = 255/(linhas*colunas) #calcula o fator

	histograma_equalizado = [0]*256
	nova_imagem = np.zeros((linhas,colunas),np.uint8)

	for i in range(len(histograma_acumulado_)):
		histograma_equalizado[i] = round(histograma_acumulado_[i]*fator)

	
	for x in range(linhas):
		for y in range(colunas):
			nova_imagem[x,y] = histograma_equalizado[img[x,y]]

	return nova_imagem,histograma_equalizado

#função que gera uma imagem com histograma alongado
def alongamento_histograma(img, Linferior, Lsuperior):
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)
	minimo = img.min()
	maximo = img.max()
	nivel = 255

	for i in range(linhas):
		for j in range(colunas):
			if img[i,j] > Lsuperior:
				imagem_resultado[i,j] = 255
			if img[i,j] < Linferior :
				imagem_resultado[i,j] = 0
			else:
				imagem_resultado[i,j] = nivel * (img[i,j] - minimo)/(maximo - minimo)

	return imagem_resultado

def especificacao_histograma(img, img_ref): #imagem a ser ajustada , imagem de referencia
	linhas,colunas = img.shape
	nova_imagem = np.zeros((linhas,colunas),np.uint8)

	histograma_img_ = gera_histograma(img) # histograma da imagem a ser ajustada
	histograma_ref_ = gera_histograma(img_ref) #histograma da imagem de referencia

	histograma_acumulado_img_ = histograma_acumulado(histograma_img_) # histograma_acumulado da imagem a ser ajustada
	histograma_acumulado_img_ref_ = histograma_acumulado(histograma_ref_) #histograma acumulado da imagem de referencia

	img_,histograma_equalizado_img_ = histograma_equalizado(img, histograma_acumulado_img_) # histograma_acumulado da imagem a ser ajustada
	img_,histograma_equalizado_img_ref = histograma_equalizado(img_ref, histograma_acumulado_img_ref_) #histograma acumulado da imagem de referencia

	histograma_especificado_ = [0]*256

	for i in range(len(histograma_equalizado_img_)):
		valor_minimo = 320000
		indice = -1
		for j in range(len(histograma_equalizado_img_ref)):
				temp =  abs(histograma_equalizado_img_[i] - histograma_equalizado_img_ref[j])
				if temp < valor_minimo:
					valor_minimo = temp
					indice = j

		histograma_especificado_[i] = indice

	for x in range(linhas):
		for y in range(colunas):
			nova_imagem[x,y] = histograma_especificado_[img[x,y]]

	return nova_imagem #retorna a imagem com histograma especificaado com base na imagem de referencia

#funçõ para gerar um grafico, dado um histograma
def grafico(histograma_plot):
	labels = [0]*256
	for h in range(256):
		labels[h] = h

	x = labels
	y = histograma_plot

	x1 = [str(d)for d in x]

	y_pos = [idx for idx, i in enumerate(y)]

	plt.figure(figsize=(10,5))


	plt.bar(y_pos, y, align='center', color="black", width=0.5) 
	plt.title("HISTOGRAMA ")
	plt.ylabel("FREQUENCIA")
	plt.xlabel('BINS')

	plt.xticks(y_pos[::4], x1[::4], size='small',rotation=45, ha="right")
	plt.yticks(np.arange(0,max(y),150))
	plt.xlim(xmin=-1)
	plt.ylim(ymax=sorted(y)[-1]+0.1) # valor maximo do eixo y
	plt.show() 

def rum_histograma():

	img = cv2.imread('lena.jpg',0) #carrega a imagem 1
	img_ref = cv2.imread('mediana.jpg',0) #carrega a imagem 2, necessria para a especificação de histograma

	_histograma_img = gera_histograma(img) #gera o histograma daa imagem
	
	#_histograma_acumulado_ = histograma_acumulado(_histograma_img) #gera histograma acumulado da imagem principal
	#_imagem_equalizada_,_histograma_equalizado_ = histograma_equalizado(img,_histograma_acumulado_) #gera  imagem equlizada e o histograma equalizado
	#img_alongada = alongamento_histograma(img.copy(),50,150)

	'''
		founção de espcificação foi feita apenas para estudo, não era necessaria paara o trabalho
	'''
	#_histograma_iref = gera_histograma(img_ref) #gera histograma da imagem de referencia
	#img_corrigida = especificacao_histograma(img.copy(), img_ref.copy()) # gera imagem com histograma especificada com base na imagem de referencia
	
	

	cv2.imshow('imagem_original', img)
	#cv2.imshow('imagem_referencia', img_ref)

	cv2.imshow('resultado', img_corrigida)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()	