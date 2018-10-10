import numpy as np
import cv2
import math
import os,copy

def dice(im1, im2, empty_score=1.0):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute dice_val coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

#executa a função dice, para todas as imagens
def calcScoreMedium(list1,list2):
	score  = 0
	for x in range(len(list1)):
		score += dice(list1[x],list2[x])

	return score/len(list1)

#carrega as imagens das marcações do especialista
def readBaseMarking():
	marking = []
	pathMarking = os.listdir('./base_pulmao/marcacao')
	for x in range(len(pathMarking)):
		marking.append(cv2.imread("./base_pulmao/marcacao/{arquivo}".format(arquivo = pathMarking[x]),0))

	return marking

#caarrega as imagens segmentadas para melhoramento
def readBaseSegmentation():
	segmentations = []
	pathSegmentations = os.listdir('./base_pulmao/segmentacao')
	for x in range(len(pathSegmentations)):
		segmentations.append(cv2.imread("./base_pulmao/segmentacao/{arquivo}".format(arquivo = pathSegmentations[x]),0))

	return segmentations

#realia os melhoramentos nas imagens segmentadas
def morphologicalAplications(segmentations,sizeKernel1,sizeKernel2,sizeKernel3):
	result = copy.deepcopy(segmentations)
	kernel1 = np.ones(sizeKernel1,np.uint8)
	kernel2 = np.ones(sizeKernel2,np.uint8)
	kernel3 = np.ones(sizeKernel3,np.uint8)
	for x in range(len(segmentations)):
		result[x] = cv2.medianBlur(segmentations[x],17) #aplicauma suavisação com filtro da mediada e kernel 17x17
		result[x] = cv2.erode(result[x],kernel1,1) #aplica erosão 
		result[x] = cv2.medianBlur(result[x],5) #aplica uma segunda suaviazação com filtro da mediana kernel 5x5
		result[x] = cv2.morphologyEx(result[x],cv2.MORPH_CLOSE,kernel3) # realiza fechamento de buracos

	return result

if __name__ == "__main__":
	result = []
	marking = readBaseMarking() #carrega as imagens de marcação
	segmentations = readBaseSegmentation() #carrega as imagens segmentadas

	scoreMediumInitial = calcScoreMedium(marking,segmentations) #calcula a media entre o dice de todas as iamgens da marcaação do especialista
	print("SCORE INICIAL: {sInitial}".format(sInitial = scoreMediumInitial))

	result = morphologicalAplications(segmentations,(17,17),(15,15),(29,29))

	scoreMediumResult = calcScoreMedium(marking,result) #calcula a media entre o dice de todas as imagens após o melhoramento
	print("SCORE MELHORADO: {sResult}".format(sResult= scoreMediumResult))

	scoreIncrease = scoreMediumResult - scoreMediumInitial 
	print("AUMENTO NO SCORE : {Increase}".format(Increase= scoreIncrease))
