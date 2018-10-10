import numpy as np
import cv2
import math 

def _soma_(img1, img2):

	linha, coluna, extra = img1.shape

	if img1.shape == img2.shape:
		img_resultado = np.zeros((linha, coluna, extra), np.uint8)

		for l in range(linha):
			for c in range(coluna):
				b = (img1[l, c, 0] + img2[l, c, 0])/2
				g = (img1[l, c, 1] + img2[l, c, 1])/2
				r = (img1[l, c, 2] + img2[l, c, 2])/2
				if(b > 255): b = 255
				if(g > 255): g = 255
				if(r > 255): r = 255

				img_resultado[l, c] = [b, g, r]
	return img_resultado

def _sub_(img1, img2):

	linha, coluna, extra = img1.shape

	if img1.shape == img2.shape:
		img_resultado = np.zeros((linha, coluna, extra), np.uint8)

		for l in range(linha):
			for c in range(coluna):
				b = (img1[l, c, 0] - img2[l, c, 0])/2
				g = (img1[l, c, 1] - img2[l, c, 1])/2
				r = (img1[l, c, 2] - img2[l, c, 2])/2
				if(b < 0): b = 0
				if(g < 0): g = 0
				if(r < 0): r = 0

				img_resultado[l, c] = [b, g, r]
	return img_resultado

def _multi_(img1, alpha):

	linha, coluna, extra = img1.shape

	img_resultado = np.zeros((linha, coluna, extra), np.uint8)

	for l in range(linha):
		for c in range(coluna):
			b = (img1[l, c, 0] * alpha)
			g = (img1[l, c, 1] * alpha)
			r = (img1[l, c, 2] * alpha)
			if(b > 255): b = 255
			if(g > 255): g = 255
			if(r > 255): r = 255

			img_resultado[l, c] = [b, g, r]
	return img_resultado

def _div_(img1, img2):

	linha, coluna, extra = img1.shape

	if img1.shape == img2.shape:
		img_resultado = np.zeros((linha, coluna, extra), np.uint8)

		for l in range(linha):
			for c in range(coluna):
				if img2[l, c, 0] != 0:
					b = (img1[l, c, 0] / img2[l, c, 0])
				else:
					b = 0
				if img2[l, c, 1] != 0:
					g = (img1[l, c, 1] / img2[l, c, 1])
				else:
					g = 0
				if img2[l, c, 1] != 0:
					r = (img1[l, c, 2] / img2[l, c, 2])
				else:
					r = 0

				img_resultado[l, c] = [b, g, r]
	return img_resultado

def _mix_(img1, alpha1, img2, beta, gama):

	linha, coluna, extra = img1.shape

	if img1.shape == img2.shape:
		img_resultado = np.zeros((linha, coluna, extra), np.uint8)

		for l in range(linha):
			for c in range(coluna):
				b1 = (img1[l, c, 0] * alpha1)
				g1 = (img1[l, c, 1] * alpha1)
				r1 = (img1[l, c, 2] * alpha1)

				b2 = (img2[l, c, 0] * beta)
				g2 = (img2[l, c, 1] * beta)
				r2 = (img2[l, c, 2] * beta)

				br = b1 + b2 + gama
				gr = b1 + b2 + gama
				rr = b1 + b2 + gama

				if(br > 255): br = 255
				if(gr > 255): gr = 255
				if(rr > 255): rr = 255

				img_resultado[l, c] = [br, gr, rr]
	return img_resultado
def dist_euclidiana(x1, y1, x2, y2):
	return math.pow(math.pow((x2-x1),2)+math.pow((y2-y1),2), 0.5)

def dist_quarteirao(x1, y1, x2, y2):
	return math.abs(x1 - x2) + math.abs(y1 - y2)

def dist_xadrez(x1, y1, x2, y2):
	return max(math.abs(x1 - x2),math.abs(y1 - y2))
	
def rum_operacoes_aritimeticas(operacao):

	img1 = cv2.imread('./imagens/teste1.png',1)
	img2 = cv2.imread('./imagens/teste2.png',1)

	if operacao == 'soma':
		resultado = _soma_(img1, img2)
	elif operacao == 'sub':
		resultado = _sub_(img1, img2)
	elif operacao == 'mult':
		resultado = _multi_(img1, 0.1)
	elif operacao == 'div':
		resultado = _div_(img1, img2)
	elif operacao == 'mix':
		resultado = _mix_(img1, 0.5, img2, 0.7, 0)
	else:
		return None


	cv2.imshow('ImagemEntrada1',img1)
	cv2.imshow('ImagemEntrada2',img2)

	titulo = '{tituloJanela}'.format(tituloJanela = operacao)
	cv2.imshow(titulo,resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()