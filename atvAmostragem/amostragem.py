import numpy as np
import cv2

def amostragem(img,n):
	amostra = [lin[::n] for lin in img[::n]]
	return np.array(amostra)


def rum_amostragem():
	filename = './imagens/venge.jpg'

	fator = [2,4]
	for ft in fator:
		img = cv2.imread(filename,0)
		amostra = amostragem(img,ft)
		cv2.imshow('resultado',amostra)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
