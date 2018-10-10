import numpy as np 
import cv2
import math as mt


def translacao_pixel(x, y, xt, yt):

	mat_translacao = np.array([1, 0, xt, 0, 1, yt, 0, 0, 1]).reshape((3, 3))
	mat_ponto = np.array([x, y, 1]).reshape((3,1))

	mat_resultado = np.dot(mat_translacao,mat_ponto)

	return mat_resultado[0,0],mat_resultado[1,0]

def escalar_pixel(x, y, xe, ye):

	mat_escalar = np.array([xe, 0, 0, 0, ye, 0, 0, 0, 1]).reshape((3, 3))
	mat_ponto = np.array([x, y, 1]).reshape((3,1))

	mat_resultado = np.dot(mat_escalar,mat_ponto)

	return mat_resultado[0,0], mat_resultado[1,0]

def rotacao_pixel(x, y, angulo):

	mat_rotacao = np.array([mt.cos(mt.radians(angulo)), -mt.sin(mt.radians(angulo)), 0, mt.sin(mt.radians(angulo)), mt.cos(mt.radians(angulo)), 0, 0, 0, 1]).reshape((3, 3))

	mat_ponto = np.array([x, y, 1]).reshape((3,1))

	mat_resultado = np.dot(mat_rotacao,mat_ponto)
	return mat_resultado[0,0], mat_resultado[1,0]

def translacao(img, xt, yt):

	nova_imagem = np.zeros((img.shape[0],img.shape[1], img.shape[2]),np.uint8)
	linhas, colunas, pixel = img.shape

	for i in range(linhas):
		for j in range(colunas):

			novo_x, novo_y = translacao_pixel(i, j, yt, xt)

			if novo_x < linhas - 1 and novo_y < colunas - 1 and novo_x >= 0 and novo_y >= 0: #pra sinais iguais
				nova_imagem[novo_x][novo_y] = img[i][j]

	return nova_imagem

def escalonamento(img, xe, ye):

	nova_imagem = np.zeros((int(img.shape[0]*xe), int(img.shape[1]*ye), img.shape[2]), dtype = np.uint8)

	linhas, colunas, pixel = img.shape

	for i in range(linhas):
		for j in range(colunas):

			novo_x, novo_y = escalar_pixel(i, j, xe, ye)

			#if novo_x < linhas - 1 and novo_y < colunas - 1 and novo_x >= 0 and novo_y >= 0:
			nova_imagem[int(novo_x)][int(novo_y)] = img[i][j]

	return nova_imagem

def rotacao(img, angulo, x = 0, y = 0):

	linhas, colunas, pixel = img.shape
	nova_imagem = np.zeros((linhas, colunas, pixel), np.uint8)

	for i in range(linhas):
		for j in range(colunas):

			novo_x, novo_y = rotacao_pixel(i, j, angulo)

			if novo_x < linhas - 1 and novo_y < colunas - 1 and novo_x >= 0 and novo_y >= 0:
				nova_imagem[int(novo_x)][int(novo_y)] = img[i][j]

	return nova_imagem

def rotacao_em_pixel(img, angulo, x, y):

	linhas, colunas, pixel = img.shape
	nova_imagem = np.zeros((linhas, colunas, pixel), np.uint8)

	for i in range(linhas):
		for j in range(colunas):

			trs_x, trs_y = translacao_pixel(i, j, -x , -y)
			rot_x, rot_y = rotacao_pixel(trs_x, trs_y, angulo)
			novo_x, novo_y = translacao_pixel(rot_x, rot_y, x, y)

			if novo_x < linhas - 1 and novo_y < colunas - 1 and novo_x >= 0 and novo_y >= 0:
				nova_imagem[int(novo_x)][int(novo_y)] = img[i][j]
				
	return nova_imagem


def rum_operacoes_geometricas():
	img1 = cv2.imread('venge.jpg')
	img2 = cv2.imread('teste2.png')

	#img1 = cv2.resize(img1 ,(200,200))

	resultado = rotacao_em_pixel(img1, 45, int(img1.shape[0]/2), int(img1.shape[1]/2)) #rotaciona em torno do pixel central?
	#-23
	#resultado = escalonamento(img1, 0.2, 0.2)

	#cv2.imwrite('escalar.jpg', resultado)
	cv2.imshow('treta', resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()		

if __name__ == "__main__":

	#rum_amostragem()
	#rum_quantizacao()
	#rum_operacoes_conjuntos()
	#rum_operacoes_aritimeticas()
	rum_operacoes_geometricas()