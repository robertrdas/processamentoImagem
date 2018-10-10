from atvAmostragem import amostragem as am
from atvQuantizacao import quantizacao as qt
from atvConjuntos import conjuntos as cj
from atvAritimetica import aritimetica as art
from atvSistemaCores import sistemaCores as sc
from atvDithering import dithering as dt
from atvRealces import realce as rc

if __name__ == "__main__":

	#execulta a amostragem
	#am.rum_amostragem()

	'''
	executa a quantização
		se quiser o metodo 1, passe como parametro a string 'metodo1'
		se quiser o metodo 2, passe como parametro a sting 'metodo2'
	'''
	#qt.rum_quantizacao('metodo2')

	'''
	executa operações de conjuntos
		parametros : 'and' , 'or' , 'xor' e 'not'
	'''
	#cj.rum_operacoes_conjuntos('and')

	'''
	executa operações de conjuntos
		parametros : 'soma' , 'sub' , 'mult' , 'div'  e 'mix'
	'''
	#art.rum_operacoes_aritimeticas('soma')

	'''
	executa operações de conversão entre sistemas de cores
		parametros : 'rgb_cmy' , 'rgb_yCrCb' , 'rgb_yuv' , 'mult_yiq'  e 'rgb_yiq'
	'''
	#sc.rum_sistema_cores('rgb_cmy')

	'''
	executa operações de conversão entre sistemas de cores
		parametros : 'basico' , 'modulacao_aleatoria' , 'aglomeracao' , 'dispersao'  e 'floyd'
	'''
	#dt.rum_dithering('modulacao_aleatoria')

	'''
	executa operações de conversão entre sistemas de cores
		parametros : 'negativo' , 'normalizacao' , 'gama' , 'linear' , 'log' , 'quad' , 'raiz'
	'''
	#rc.rum_realce('negativo')