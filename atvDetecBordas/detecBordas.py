import numpy as np
import cv2
import math as mt


def filtro_gaussiano(img,n,sig):
	print("suavizando....")
	valor = 0
	aresta  = n//2
	arestas =[]

	if n == 3:
		mascara = list(np.array([1,2,1,2,4,2,1,2,1])/16) #usado para mascara 3x3
	elif n==5:
		mascara = list(np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1])/273) #usado para mascar 5x5
	else:
		return None
		
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(n):
				for y in range(n):
					arestas.append(img[i-aresta+x,j-aresta+y])					
			
			for k in range(len(arestas)):
				valor += arestas[k] * mascara[k]

			resultado = round((valor))
			
			if resultado < 0 : 
				resultado = 0
			if resultado > 255: 
				resultado = 255

			imagem_resultado[i,j] = resultado

			arestas.clear()
			valor = 0
	print("suaviza d+....")
	return imagem_resultado


def detec_ponto(img,limiar):
	valor = 0
	aresta  = 3//2
	arestas =[]

	mascara = list(np.array([-1,-1,-1,-1,8,-1,-1,-1,-1])) #usado para mascara 3x3 paara detcçãao de ponto
	
	linhas,colunas = img.shape

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(3):
				for y in range(3):
					arestas.append(img[i-aresta+x,j-aresta+y])					
			
			for k in range(len(arestas)):
				valor += arestas[k] * mascara[k]

			resultado = round((valor))
			
			if resultado > limiar : 
				imagem_resultado[i,j] = 255


			arestas.clear()
			valor = 0
	return imagem_resultado

def inicia_detec_ponto():
	img = cv2.imread('./imagens/estrelas.jpg',0)

	resultado = filtro_gaussiano(img.copy(),5,1)

	resultado1 = detec_ponto(resultado,127);

	cv2.imshow('original', img)
	cv2.imshow('resultado', resultado1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def detec_retas(img,limiar,ang):
	print("vai comecar....")
	valor = 0
	aresta  = 3//2
	arestas =[]

	if ang == 0:
		mascara = list(np.array([-1,-1,-1, 2, 2, 2,-1,-1,-1])) #usado para mascara 3x3 paara detcçãao de ponto
	elif ang == 90:
		mascara = list(np.array([-1, 2,-1,-1, 2,-1,-1, 2,-1])) #usado para mascara 3x3 paara detcçãao de ponto
	elif ang == 45:
		mascara =  list(np.array([-1,-1, 2,-1, 2,-1, 2,-1,-1]))#usado para mascara 3x3 paara detcçãao de ponto	
	elif ang == -45:
		mascara = list(np.array([2,-1,-1,-1, 2,-1,-1,-1, 2])) #usado para mascara 3x3 paara detcçãao de ponto
	
	linhas,colunas = img.shape

	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(3):
				for y in range(3):
					arestas.append(img[i-aresta+x,j-aresta+y])					
			
			#print(mascara)
			#print(arestas)

			for k in range(len(arestas)):
				valor += arestas[k] * mascara[k]
			#print(valor)
			
			if valor > limiar : 
				imagem_resultado[i,j] = 255


			arestas.clear()
			valor = 0
	print("sucesso d+....")
	return imagem_resultado

def algoritimo_basico(img):

	linhas,colunas = img.shape

	limiar = int((0+255)/2)
	
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(linhas):
		for j in range(colunas):
			if img[i,j] >= limiar:
				imagem_resultado[i,j] = 255

	return imagem_resultado


def inicia_detec_retas():

	img = cv2.imread('./imagens/tabuleiro.jpg',0)

	#resultado = filtro_gaussiano(img.copy(),5,1)

	resultado1 = detec_retas(img,127,0);

	cv2.imshow('original', img)
	cv2.imshow('resultado', resultado1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def prewitt(img,limiar):
	aresta  = 3//2
	arestas =[]

	mX = list(np.array([-1,-1,-1,0,0,0,1,1,1])) #kernel da marcara para calcular o gradiente x
	mY = list(np.array([-1,0,1,-1,0,1,-1,0,1])) #kernel da marcara para calcular o gradiente y

	linhas,colunas = img.shape

	valorX = 0
	valorY = 0

	gradienteX = np.zeros((linhas,colunas)) #imagem do gradiente x
	gradienteY = np.zeros((linhas,colunas)) #imagem do gradiente y

	saida = np.zeros((linhas,colunas),np.uint8) #imagem de saida

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(3):
				for y in range(3):
					arestas.append(img[i-aresta+x,j-aresta+y])					
	
			#calcula para o gradiente x
			for k in range(len(arestas)):
				valorX += arestas[k] * mX[k]
			gradienteX[i,j] = valorX				
			
			#calcula para o gradiente y
			for k in range(len(arestas)):
				valorY += arestas[k] * mY[k]
			gradienteY[i,j] = valorY

			arestas.clear()

			valorY = 0
			valorX = 0

	cv2.imshow('gradientex', gradienteX)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow('gradientey', gradienteY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	for i in range (linhas):
		for j in range(colunas):
			aux = abs(gradienteX[i,j]) + abs(gradienteY[i,j])
			if aux > limiar:
				saida[i,j] = 255


	return saida #retorna a imagem com as bordas identificadas

def roberts(img,limiar):
	aresta  = 2//2
	arestas =[]

	mX = list(np.array([1, 0, 0, -1]))
	mY = list(np.array([0, -1, 1, 0]))

	linhas,colunas = img.shape

	valorX = 0
	valorY = 0

	gradienteX = np.zeros((linhas,colunas)) #imagem do gradiente x
	gradienteY = np.zeros((linhas,colunas)) #imagem do gradiente y

	saida = np.zeros((linhas,colunas),np.uint8) #imagem de saida

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(2):
				for y in range(2):
					arestas.append(img[i-aresta+x,j-aresta+y])					
	
			#calcula para o gradiente x
			for k in range(len(arestas)):
				valorX += arestas[k] * mX[k]
			gradienteX[i,j] = valorX		
			
			#calcula para o gradiente y
			for k in range(len(arestas)):
				valorY += arestas[k] * mY[k]
			gradienteY[i,j] = valorY

			arestas.clear()

			valorX = 0	
			valorY = 0

	cv2.imshow('gradientex', gradienteX)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow('gradientey', gradienteY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	for i in range (linhas):
		for j in range(colunas):
			aux = abs(gradienteX[i,j]) + abs(gradienteY[i,j])
			if aux > limiar:
				saida[i,j] = 255
				
	return saida

def sobel(img,limiar):
	aresta  = 3//2
	arestas =[]

	mX = list(np.array([-1,0,1,-2,0,2,-1,0,1]))
	mY = list(np.array([-1,-2,-1,0,0,0,1,2,1]))

	linhas,colunas = img.shape

	valorX = 0
	valorY = 0

	gradienteX = np.zeros((linhas,colunas)) #imagem do gradiente x
	gradienteY = np.zeros((linhas,colunas)) #imagem do gradiente y

	saida = np.zeros((linhas,colunas),np.uint8) #imagem de saida

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(3):
				for y in range(3):
					arestas.append(img[i-aresta+x,j-aresta+y])					
	
			#calcula para o gradiente x
			for k in range(len(arestas)):
				valorX += arestas[k] * mX[k]
			gradienteX[i,j] = valorX		
			
			#calcula para o gradiente y
			for k in range(len(arestas)):
				valorY += arestas[k] * mY[k]
			gradienteY[i,j] = valorY

			arestas.clear()

			valorX = 0	
			valorY = 0

	cv2.imshow('gradientex', gradienteX)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow('gradientey', gradienteY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	for i in range (linhas):
		for j in range(colunas):
			aux = abs(gradienteX[i,j]) + abs(gradienteY[i,j])
			if aux > limiar:
				saida[i,j] = 255
				
	return saida

def laplaciano(img,limiar):
	aresta  = 3//2
	arestas =[]

	mX = list(np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]))
	mY = list(np.array([2, -1, 2, -1, -4, -1, 2, -1, 2]))

	linhas,colunas = img.shape

	valorX = 0
	valorY = 0

	gradienteX = np.zeros((linhas,colunas)) #imagem do gradiente x
	gradienteY = np.zeros((linhas,colunas)) #imagem do gradiente y

	saida = np.zeros((linhas,colunas),np.uint8) #imagem de saida

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(3):
				for y in range(3):
					arestas.append(img[i-aresta+x,j-aresta+y])					
	
			#calcula para o gradiente x
			for k in range(len(arestas)):
				valorX += arestas[k] * mX[k]
			gradienteX[i,j] = valorX		
			
			#calcula para o gradiente y
			for k in range(len(arestas)):
				valorY += arestas[k] * mY[k]
			gradienteY[i,j] = valorY

			arestas.clear()

			valorX = 0	
			valorY = 0

	cv2.imshow('gradientex', gradienteX)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow('gradientey', gradienteY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	for i in range (linhas):
		for j in range(colunas):
			aux = abs(gradienteX[i,j]) + abs(gradienteY[i,j])
			if aux > limiar:
				saida[i,j] = 255
				
	return saida

def inicia_deteccao_borda(opc):
	img = cv2.imread('./imagens/tabuleiro.jpg',0)

	resultado = filtro_gaussiano(img,5,1)

	if opc == 1:
		resultado1 = roberts(algoritimo_basico(img),127);
		nome = "resultado - roberts"
	elif opc == 2 :
		resultado1 = prewitt(img,50);
		nome = "resultado - prewitt"
	elif opc == 3:
		resultado1 = sobel(img,50);
		nome = "resultado - sobel"
	elif opc == 4 :
		resultado1 = laplaciano(img,50)
		nome = "resultado - laplaciano"

	cv2.imshow('Imagem de Entrada', img)
	cv2.imshow(nome, resultado1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":

	'''
	Essas duas funções realizam a detecção de ponto e reta , com imagens pre definidas
	caso queira altera-las dirija-se ate as funções inicia_detec_ponto ou inicia_detc_reta
	'''
	#inicia_detec_ponto()
	#inicia_detec_retas()

	'''
	Essas duas funções realizam a detecção de ponto e reta , com imagens pre definidas
	caso queira altera-las dirija-se ate a função inicia_deteccao_bordas.
	O parametro da função diz respeito ao algoritmo a ser usado

		1 - algoritmo de ROBERTS
		2 - algoritmo PREWITT
		3 - algoritmo SOBEL
		4 - algoritmo LAPLACIANO
	'''
	inicia_deteccao_borda(3)
	

	