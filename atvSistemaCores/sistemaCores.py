import numpy as np 
import cv2

#função para converter de rgb para cmy
def rgb_cmy (img):
	linha,coluna,canais = img.shape

	imagem_cmy = np.zeros((linha,coluna,canais),np.uint8)

	branco = [255,255,255] #vetor auxiliar usado nas subtrações

	for i in range(linha):
		for j in range(coluna):
			imagem_cmy[i][j] =  list( np.array(branco) - np.array(img[i,j]) )

	return imagem_cmy

#converte rgb para YCrCb
def rgb_yCrCb (img):
	delta = 128 #128 pois esta sendo trabalhado em imgens de 128 bits
	

	linha,coluna,canais = img.shape
	imagem_yCrCb = np.zeros((linha,coluna,canais),np.uint8) #declara vetor que servirá como imagem resultado

	for i in range(linha):
		for j in range(coluna):

			y = 0.299 * img[i,j][2] + 0.587 * img[i,j][1] + 0.114 * img[i,j][0] #calculando vlor do pixel para o canl y
			Cr = (img[i,j][2] - y) * 0.713 + delta #calculando o valor do pixel para o canal Cr
			Cb = (img[i,j][0] - y) * 0.564 + delta #calculaando o valor do pixel para o canal Cb

			imagem_yCrCb[i][j] = [y,Cr,Cb]  #monta o pixel com os tres valores dos canais

	return imagem_yCrCb

#converte rgb par yuv
def rgb_yuv (img):
	
	linha,coluna,canais = img.shape
	imagem_yuv = np.zeros((linha,coluna,canais),np.uint8)#declara vetor que servirá como imagem resultado

	for i in range(linha):
		for j in range(coluna):
			y = 0.299 * img[i,j][2] + 0.587 * img[i,j][1] + 0.114 * img[i,j][0] #calcula o valor do pixel para o canal y
			u = img[i,j][2] - y #calcula o valor do pixel para o canal u
			v = img[i,j][0] - y #calcula o valor do pixel para o canal v

			imagem_yuv[i][j] = [y,u,v]  

	return imagem_yuv

#função auxiliar para mutiplicar matriz necessaaria na conversãao de rgb para yiq	
def mult_yiq(b, g, r):

	#mat_coeficiente = np.array([0.299, 0.587, 0.144, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape((3,3))  #matriz de coeficientes
	mat_coeficiente = np.array([0.144,0.587,0.299,-0.321,-0.275,0.596,0.311,-0.523,0.212]).reshape((3,3))  #matriz de coeficientes
	mat_bgr = np.array([b, g, r]).reshape((3,1)) #matiz do pixel bgr
	mat_resultado = np.dot(mat_coeficiente,mat_bgr) #multiplica a matriz

	return int(mat_resultado[0][0]),int(mat_resultado[1][0]),int(mat_resultado[2][0]) #retorna o valor equivalente r,g,b

def rgb_yiq(img):

	rows, cols, channels = img.shape

	normalized = img.copy()
	normalized = np.divide(normalized, 255)

	_returnImg = np.zeros((rows, cols, channels), np.uint8)
	
	matrix = np.array([0.144, 0.587, 0.299, 
					-0.321, -0.275, 0.596,
					0.311, -0.523, 0.212]).reshape(3, 3)

	for i in range(rows):
		for j in range(cols):			
			YIQ = np.dot(matrix, normalized[i, j])
			YIQ = np.multiply(YIQ, 255)
			_returnImg[i, j] = YIQ

	return _returnImg

def rum_sistema_cores(metodo):
	img1 = cv2.imread('./imagens/lena.jpg',1)

	if metodo == 'rgb_cmy':
		resultado = rgb_cmy(img1) #para converter de rgb para cmy
	elif metodo == 'rgb_yCrCb':
		resultado = rgb_yCrCb(img1) #converter de rgb para yCrCb
	elif metodo == 'rgb_yuv':
		resultado = rgb_yuv(img1) #converter de rgb para yuv
	elif metodo == 'rgb_yiq':
		resultado = rgb_yiq(img1) #converter de rgb para yiq
	else:
		return None


	cv2.imshow('ImagemEntrada1',img1)

	titulo = '{tituloJanela}'.format(tituloJanela = metodo)
	cv2.imshow(titulo,resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
'''
Arquivo contem todas as atividades pedidas em sala de aula.
Cada convrsão entre sistemaa de cores tem su propria função
Juntando todas as funções de um conjunto existe uma função para executar o conjunto desejado, a qual é chammada pela main.
	{exemplo: Se quiser executar a converção  rgb pra cmy 
		Va na main e descomente a linha #resultado = rgb_cmy(img1)
	}
OBS: todas as imagens são obtidas do caminho relativo sendo que as imagens que são carregadas ja se encontram na pasta enviada
obrigado! Robert Douglas de A. Santos
'''