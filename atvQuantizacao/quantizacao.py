import cv2,math
import numpy as np

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

def quantizacao_uniforme_2(img,k):
	a=np.float32(img)
	bucket = 256/k
	quantizado = (a/(256/k))
	return np.uint8(quantizado)*bucket

def rum_quantizacao(quantizacaoEscolhida):
	filename = './imagens/venge.jpg'
	cores =[2,8]

	for cor in cores:
		img = cv2.imread(filename,0)

		if quantizacaoEscolhida == 'metodo1':
			resultado = quantizacao_uniforme(img,cor)
		else:
			resultado = quantizacao_uniforme_2(img,cor)

		cv2.imshow('resultado',resultado)
		cv2.waitKey(0)
		cv2.destroyAllWindows()