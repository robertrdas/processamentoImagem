import numpy as np
import cv2,os,math

def amostragem(img,n):
	amostra = [lin[::n] for lin in img[::n]]
	return np.array(amostra)

def rum_amostragem():
	filename = 'venge.jpg'

	fator = [2,4]
	for ft in fator:
		img = cv2.imread(filename,0)
		amostra = amostragem(img,ft)

	cv2.imwrite('vengeAmostrada.jpg',amostra)
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
	
def rum_operacoes_conjuntos():
	img1 = cv2.imread('quadrado1.png',0)
	img2 = cv2.imread('quadrado2.png',0)

	#resultado = _and_(img1,img2)
	#resultado = _or_(img1,img2)
	#resultado = _xor_(img1)
	resultado = _not_(img1)

	cv2.imwrite('and.jpg',resultado)
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
	#resultado = _div_(img1, img2)
	#resultado = _mix_(img1, 0.5, img2, 0.7, 0)

	cv2.imwrite('soma.jpg', resultado)
	cv2.imshow('Soma', resultado)
	cv2.waitKey(0)
	cv2.destroyAllWindows()		

if __name__ == "__main__":

	#rum_amostragem()
	#rum_quantizacao()
	#rum_operacoes_conjuntos()
	#rum_operacoes_aritimeticas()