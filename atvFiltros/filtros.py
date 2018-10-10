import numpy as np
import cv2
import math as mt

def filtro_media(img,n):
	valor = 0
	aresta  = int(n/2)

	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(n):
				for y in range(n):
					valor += img[i-aresta+x,j-aresta+y]					

			novo_pixel = round((valor*1.0)/(n*n))
			imagem_resultado[i,j] = novo_pixel
			valor = 0

	return imagem_resultado

def gaussXY(x,y,sig):
	fator = (1/(2*mt.pi*(sig**2)))

	g = fator*mt.exp( -((x**2 + y**2)/(2*(sig**2))) )

	return g

def mascara_gauss(n, sig):
	resultado = np.zeros((n,n))

	a = n//2
	b = n//2

	for x in range(-a,a):
		for y in range(-b,b):
			resultado[x+a+1,y+b+1] = gaussXY(x,y,sig) 
	
				
	resultado = resultado/np.sum(resultado)
	return list(resultado.flatten())

def filtro_gaussiano(img,n,sig):
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
	return imagem_resultado

def filtro_mediana(img,n):
	valor = 0
	aresta  = n//2
	arestas =[]


	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(n):
				for y in range(n):
					arestas.append(img[i-aresta+x,j-aresta+y])					

			arestas.sort()
			pos = len(arestas)//2
			resultado = arestas[pos]
			imagem_resultado[i,j] = resultado

			arestas.clear()
			valor = 0
	return imagem_resultado

if __name__ == "__main__":

	img = cv2.imread('lena1.jpg',0)
	
	#resultado = filtro_media(img,15)
	#resultado = filtro_gaussiano(img.copy(),5,1)
	resultado = filtro_mediana(img,5)

	cv2.imshow('original', img)
	cv2.imshow('resultado', resultado)


	cv2.waitKey(0)
	cv2.destroyAllWindows()