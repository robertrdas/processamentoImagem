import numpy as np
import cv2

def _and_(img1,img2):

	linha_img1,linha_img2 = img1.shape[0],img2.shape[0]

	col_img1,col_img2 = img1.shape[1],img2.shape[1]

	if linha_img1==linha_img2 and col_img1==col_img2:
		img_resultado = np.zeros((linha_img1,col_img1))

		for linha in range(linha_img1):
			for coluna in range(col_img1):
				if img1[linha][coluna] == img2[linha][coluna] and img1[linha][coluna] == 255 and img2[linha][coluna] == 255 :
					img_resultado[linha][coluna] = 255
			
	return img_resultado

def _or_(img1,img2):

	linha_img1,linha_img2 = img1.shape[0],img2.shape[0]

	col_img1,col_img2 = img1.shape[1],img2.shape[1]

	if linha_img1==linha_img2 and col_img1==col_img2:
		img_resultado =np.zeros((img1.shape[0],img2.shape[1]))
		#img_resultado = np.full((img1.shape[0],img2.shape[0]),255)

		for linha in range(linha_img1):
			for coluna in range(col_img1):
				if img1[linha][coluna] == img2[linha][coluna] and img1[linha][coluna] == 0 and img2[linha][coluna] == 0 :
					img_resultado[linha][coluna] = 0
				else:
					img_resultado[linha][coluna] = 255
			
	return img_resultado

def _xor_(img1,img2):

	linha_img1,linha_img2 = img1.shape[0],img2.shape[0]

	col_img1,col_img2 = img1.shape[1],img2.shape[1]

	if linha_img1==linha_img2 and col_img1==col_img2:
		img_resultado =np.zeros((img1.shape[0],img2.shape[1]))
		#img_resultado = np.full((img1.shape[0],img2.shape[0]),255)

		for linha in range(linha_img1):
			for coluna in range(col_img1):
				if img1[linha][coluna] == img2[linha][coluna]  :
					img_resultado[linha][coluna] = 0
				else:
					img_resultado[linha][coluna] = 255
			
	return img_resultado

def _not_(img1):

	linha_img1,col_img1 = img1.shape[0],img1.shape[1]

	img_resultado = np.zeros((linha_img1,col_img1))
		#img_resultado = np.full((img1.shape[0],img2.shape[0]),255)

	for linha in range(linha_img1):
		for coluna in range(col_img1):
			if img1[linha][coluna]==255:
				img_resultado[linha][coluna] = 0
			else:
				img_resultado[linha][coluna] = 255
			
	return img_resultado

def rum_operacoes_conjuntos(operacao):
	img1 = cv2.imread('./imagens/quadrado1.png',0)
	img2 = cv2.imread('./imagens/quadrado2.png',0)

	if operacao == 'and':
		resultado = _and_(img1,img2)
	elif operacao == 'or':
		resultado = _or_(img1,img2)
	elif operacao == 'xor':
		resultado = _xor_(img1)
	elif operacao == 'not':
		resultado = _not_(img1)
	else:
		return None


	cv2.imshow('ImagemEntrada1',img1)
	cv2.imshow('ImagemEntrada2',img2)

	titulo = '{tituloJanela}'.format(tituloJanela = operacao)
	cv2.imshow(titulo,resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()