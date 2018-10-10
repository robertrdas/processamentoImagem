import numpy as np
import cv2,os,math

'''
	contem instruções de uso apos a main.
'''

def amostragem(img,n):
	amostra = [lin[::n] for lin in img[::n]]
	return np.array(amostra)

def rum_amostragem():
	filename = 'venge.jpg'

	fator = [2,4]
	for ft in fator:
		img = cv2.imread(filename,0)
		amostra = amostragem(img,ft)

	#cv2.imwrite('vengeAmostrada.jpg',amostra)
	cv2.imshow('teste',amostra)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

def quantizacao_uniforme(img, k):

	img = np.float32(img)
	quantized = img.copy()

	rows = img.shape[0]
	cols = img.shape[1]

	for i in range(rows):
		for j in range(cols):
			quantized[i,j]=((math.pow(2,k)-1)*np.float32((img[i,j])-img.min())/(img.max()-img.min()))
			quantized[i,j]=np.round(quantized[i,j])*int(256/math.pow(2,k))

	return quantized

def quantizacao_uniforme2(img,k):
	a=np.float32(img)
	bucket = 256/k
	quantizado = (a/(256/k))
	return np.uint8(quantizado)*bucket

def rum_quantizacao():
	filename = 'venge.jpg'
	cores =[2,8]

	for cor in cores:
		img = cv2.imread(filename,0)
		resultado = quantizacao_uniforme2(img,cor)
		name,extension = os.path.splitext(filename)
		new_filename= '{name}-quantizado-{k}{ext}'.format(name=name,k=cor,ext=extension)
		cv2.imwrite(new_filename,resultado)

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
	
def rum_operacoes_logicas():
	img1 = cv2.imread('quadrado1.png',0)
	img2 = cv2.imread('quadrado2.png',0)

	#resultado = _and_(img1,img2)
	#resultado = _or_(img1,img2)
	#resultado = _xor_(img1,img2)
	#resultado = _not_(img1)

	#cv2.imwrite('and.jpg',resultado)
	cv2.imshow('and',resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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
	
def rum_operacoes_aritimeticas():
	img1 = cv2.imread('teste1.png')
	img2 = cv2.imread('teste2.png')

	#resultado = _soma_(img1, img2)
	#resultado = _sub_(img1, img2)
	#resultado = _multi_(img1, 0.1)
	#resultado = _div_(img1, img2)
	#resultado = _mix_(img1, 0.5, img2, 0.7, 0)

	#cv2.imwrite('soma.jpg', resultado)
	cv2.imshow('Soma', resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()		


def translacao_pixel(x, y, xt, yt):

	mat_translacao = np.array([1, 0, xt, 0, 1, yt, 0, 0, 1]).reshape((3, 3))
	mat_ponto = np.array([x, y, 1]).reshape((3,1))

	mat_resultado = np.dot(mat_translacao,mat_ponto)

	return mat_resultado[0,0],mat_resultado[1,0]

def escalar_pixel(x, y, xe, ye):

	mat_escalar = np.array([xe, 0, 0, 0, ye, 0, 0, 0, 1]).reshape((3, 3)).astype(np.uint8)
	mat_ponto = np.array([x, y, 1]).reshape((3,1))

	mat_resultado = np.dot(mat_escalar,mat_ponto)

	return mat_resultado[0,0], mat_resultado[1,0]

def rotacao_pixel(x, y, angulo):

	mat_rotacao = np.array([math.cos(math.radians(angulo)), -math.sin(math.radians(angulo)), 0, math.sin(math.radians(angulo)), math.cos(math.radians(angulo)), 0, 0, 0, 1]).reshape((3, 3))

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

	nova_imagem = np.zeros((img.shape[0] * math.ceil(xe), img.shape[1] * math.ceil(ye), img.shape[2]), np.uint8)
	linhas, colunas, pixel = img.shape

	for i in range(linhas):
		for j in range(colunas):

			novo_x, novo_y = escalar_pixel(i, j, math.ceil(xe), math.ceil(ye))

			#if novo_x < linhas - 1 and novo_y < colunas - 1 and novo_x >= 0 and novo_y >= 0:
			nova_imagem[novo_x][novo_y] = img[i][j]

	return nova_imagem

def rotacao(img, angulo):

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
	img1 = cv2.imread('download.jpg')
	img2 = cv2.imread('teste2.png')

	#resultado = escalonamento(img1, 2, 2)
	#resultado = translacao(img1, 20, 20)
	#resultado = rotacao(img1,45)
	#resultado = rotacao_em_pixel(img1, 90, int(img1.shape[0]/2), int(img1.shape[1]/2)) #rotaciona em torno do pixel central?
	#-23

	#cv2.imwrite('translacao.jpg', resultado)
	cv2.imshow('operacoes_geometricas', resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

if __name__ == "__main__":

	#rum_amostragem()
	#rum_quantizacao()
	#rum_operacoes_logicas()
	#rum_operacoes_aritimeticas()
	#rum_operacoes_geometricas()

'''
Arquivo contem todas as atividades pedidas em sala de aula.
É dividido em : AMOSTRAGEM, QUANTIZAÇÃO, OPERAÇÕES LOGICAS, OPERAÇÕES ARITIMETICAS E OPERAÇÕES GEOMETRICAS
Cada item dentro de um conjunto de operações tem uma função propria
Juntando todas as funções de um conjunto existe uma função para executar o conjunto desejado, a qual é chammada pela main.
	{exemplo: Se quiser executar operações logicas: 
		passo 1: va na main e descomente a linha #rum_operacoes_logicas()
		passo 2: va na função rum_operacoes_logicas() e descomente a operação especifica
		ex: operação and: 
			resultado = _and_(img1,img2)
			Dessa forma irá execultar a operação and do conjunto operações_logicas
		assim por diante.
	}
	O que foi explicado se aplica a todo o resto erxeceto para quantização e amostragem que so é preciso descomentar na main
OBS: todas as imagens são obtidas do caminho relativo sendo que as imagens que são carregadas ja se encontram na pasta enviada
obrigado! Robert Douglas de A. Santos
'''