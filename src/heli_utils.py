import numpy as np
#from casadi import *

def get_state_space():

	state_space = np.zeros((29,2))
	state_space[0] = [-10,10]
	state_space[1] = [-10,10]
	state_space[2] = [-20,20]
	state_space[3] = [-10,10]
	state_space[4] = [-10,10]
	state_space[5] = [-300, 300]
	state_space[6] = [-350, 350]
	state_space[7] = [-100, 100]
	state_space[8] = [-20,20]
	state_space[9] = [-20,20]
	state_space[10] = [-40,40]
	state_space[11] = [-30,30]
	state_space[12] = [-30,30]
	state_space[13] = [-600,600]
	state_space[14] = [-600,600]
	state_space[15] = [-200,200]
	state_space[16] = [-10,10]
	state_space[17] = [-130, 130]
	state_space[18] = [-50,50]
	state_space[19] = [-25,25]
	state_space[20] = [-20,20]
	state_space[21] = [-10,10]
	state_space[22] = [-30,30]
	state_space[23] = [-10,10]
	state_space[24] = [-15,15]
	state_space[25] = [-30,30]
	state_space[26] = [-15,15]
	state_space[27] = [-35,35]
	state_space[28] = [0,100]

	return state_space


def derivative(x):
		dx = np.zeros(len(x))
		dx[0] = (0.998573780060 * x[3] + 0.053384274244 * x[4])
	
		dx[1] = (x[2] - 0.003182219341 * x[3] + 0.059524655342 * x[4])
	
		dx[2] = (-11.570495605469 * x[2] - 2.544637680054 * x[3] 
											 - 0.063602626324 * x[4] + 0.106780529022 * x[5]
											 - 0.094918668270 * x[6] + 0.007107574493 * x[7]
											 - 3.700790207851 * x[8] - 16.213284674534 * x[9] 
											 - 2.984968535139 * x[10] - 0.493137919288 * x[11] 
											 - 1.186954196152 * x[12] - 0.031106608756 * x[13] 
											 + 0.024595252653 * x[14] + 0.008231369923 * x[15] 
											 + 0.231787619674 * x[16] + 0.745302732591 * x[17]
											 + 7.895709940231 * x[18] + 2.026930360369 * x[19]  
											 + 0.271792657736 * x[20] + 0.315196108541 * x[21]  
											 + 0.015876847710 * x[22] + 0.009288507454 * x[23] 
											 + 0.087920280806 * x[24] - 0.103727794204 * x[25]  
											 - 4.447282126346 * x[26] + 0.016271459306 * x[27])
	
		dx[3] = (0.439356565475 * x[2] - 1.998182296753 * x[3] 
											 + 0.016651883721 * x[5] + 0.018462046981 * x[6]
											 - 0.001187470742 * x[7] - 7.517319654181 * x[8]
											 + 0.236494174025 * x[9] - 0.028725044803 * x[10]
											 - 2.442989538035 * x[11] + 0.034510550810 * x[12]
											 - 0.004683216652 * x[13] - 0.005154038690 * x[14]
											 - 0.002104275246 * x[15] - 0.079935853309 * x[16]
											 + 1.420125114638 * x[17] - 0.117856066698 * x[18]
											 - 0.226142434271 * x[19] - 0.002585832387 * x[20]
											 - 0.001365917341 * x[21] + 0.035962654791 * x[22]
											 + 0.028993699893 * x[23] - 0.045896888864 * x[24]
											 + 0.716358354284 * x[25] + 0.029085601036 * x[26]
											 - 0.001242728387 * x[27])
	
		dx[4] = (-2.040895462036 * x[2]- 0.458999156952 * x[3]
											 - 0.735027790070 * x[4] + 0.019255757332 * x[5]
											 - 0.004595622420 * x[6] + 0.002120360732 * x[7]
											 - 0.740775522612 * x[8] - 2.555714688932 * x[9]
											 - 0.339301128908 * x[10] - 0.033104023297 * x[11]
											 - 1.446467788369 * x[12] - 0.007442776396 * x[13]
											 - 0.000012314482 * x[14] + 0.030657946816 * x[15]
											 + 1.002118140789 * x[16] + 0.153644862643 * x[17]
											 + 1.273828227991 * x[18] + 1.983204935524 * x[19]
											 + 0.048757213739 * x[20] + 0.060295617991 * x[21]
											 + 0.001605314985 * x[22] + 0.000554368427 * x[23]
											 + 0.475422075598 * x[24] - 0.010880647601 * x[25]
											 - 0.775712358056 * x[26]- 0.408545111762 * x[27])
	
		dx[5] = (-32.103607177734 * x[0] - 0.503355026245 * x[2] 
											 + 2.297859191895 * x[3] - 0.021215811372 * x[5]
											 - 0.021167919040 * x[6] + 0.015811592340 * x[7]
											 + 8.689411857722 * x[8] - 0.215429806172 * x[9]
											 + 0.063500560122 * x[10] + 2.847523923644 * x[11]
											 - 0.297021616015 * x[12] + 0.001323463163 * x[13]
											 + 0.002124820781 * x[14] + 0.068860932948 * x[15]
											 + 1.694077894544 * x[16] - 1.639571645676 * x[17]
											 + 0.110652545728 * x[18] + 0.728735301618 * x[19]
											 + 0.003107735169 * x[20] + 0.003335187976 * x[21]
											 - 0.042347579477 * x[22] - 0.034247794709 * x[23]
											 + 0.469091132962 * x[24] - 0.814424502262 * x[25]
											 - 0.018082452136 * x[26]+ 0.016747349252 * x[27])
		
		dx[6] = (0.102161169052 * x[0] + 32.057830810547 * x[1]
											 - 2.347217559814 * x[2] - 0.503611564636 * x[3]
											 + 0.834947586060 * x[4] + 0.021226570010 * x[5]
											 - 0.037879735231 * x[6] + 0.000354003860 * x[7]
											 - 0.560681623936 * x[8] - 3.574948145694 * x[9]
											 - 0.788176766644 * x[10] - 0.107590635594 * x[11]
											 + 0.908657075077 * x[12] - 0.008720966051 * x[13]
											 + 0.005669792925 * x[14] + 0.044884407612 * x[15]
											 + 0.788227489086 * x[16] + 0.111065913657 * x[17]
											 + 1.709840089441 * x[18] - 0.946574755181 * x[19]
											 + 0.054255711842 * x[20] + 0.060392345409 * x[21]
											 + 0.003299051857 * x[22] + 0.001965592530 * x[23]
											 - 0.035607238660 * x[24] - 0.021984114632 * x[25]
											 - 0.893130060176 * x[26] + 0.503048977806 * x[27])
	
		dx[7] = (-1.910972595215 * x[0] + 1.713829040527 * x[1]
											 - 0.004005432129 * x[2] - 0.057411193848 * x[3]
											 + 0.013989634812 * x[5] - 0.000906753354 * x[6]
											 - 0.290513515472 * x[7] - 1.440209153996 * x[8]
											 - 1.089782421583 * x[9] - 0.599051729911 * x[10]
											 - 0.930901394778 * x[11] + 5.044060722850 * x[12]
											 + 0.079229241316 * x[13] + 0.074101747848 * x[14]
											 - 1.301808243838 * x[15] - 31.393874531397 * x[16]
											 + 0.233327947688 * x[17] + 0.478559456452 * x[18]
											 - 9.198865975131 * x[19] - 0.002820980233 * x[20]
											 - 0.034669033757 * x[21] + 0.022125233836 * x[22]
											 + 0.019923408940 * x[23] - 8.159414332666 * x[24]
											 - 0.129736796488 * x[25] - 0.298841506489 * x[26]
											 - 0.300193732750 * x[27])
	
		dx[8] = (0.050176870833 * x[0] - 0.003161246171 * x[1]
											 - 0.486165175190 * x[2] + 0.266534777047 * x[3]
											 + 0.003826227932 * x[4] + 0.000001339204 * x[5]
											 + 0.000001199431 * x[6] - 0.000022435600 * x[7]
											 - 0.020657323970 * x[8] + 0.001301453941 * x[9]
											 + 0.213359280279 * x[10] + 0.881596311923 * x[11]
											 + 0.051809053856 * x[12] - 0.000000551337 * x[13]
											 - 0.000000493794 * x[14] + 0.000009236516 * x[15])
	
		dx[9] = (-0.019757788570 * x[0] + 0.009012833714 * x[1]
												 + 0.311015942657 * x[2] + 2.810181204790 * x[3]
												 - 0.001602140073 * x[4] - 0.000000613279 * x[5]
												 - 0.000000549271 * x[6] + 0.000010274224 * x[7]
												 + 0.008134087133 * x[8] - 0.003710494952 * x[9]
												 + 0.863507011470 * x[10] - 1.236460821044 * x[11]
												 + 0.060184240645 * x[12] + 0.000000252481 * x[13]
												 + 0.000000226129 * x[14] - 0.000004229797 * x[15])
	 
		dx[10] = (-0.030385323449 * x[0] + 0.003110159427 * x[1]
												 + 0.312812882924 * x[2] + 0.287354391281 * x[3]
												 - 0.002331730630 * x[4] - 0.000000824205 * x[5]
												 - 0.000000738183 * x[6] + 0.000013807861 * x[7]
												 - 8.414922645141 * x[8] - 36.922139523656 * x[9]
												 - 18.505141519315 * x[10] - 3.793715804769 * x[11]
												 - 2.765572372983 * x[12] + 0.035944961732 * x[13]
												 - 0.038910104720 * x[14] + 0.025846348888 * x[15]
												 + 0.527826299191 * x[16] + 1.697201876759 * x[17]
												 + 17.980094722474 * x[18] + 4.615721721183 * x[19]
												 + 0.618925691035 * x[20] + 0.717763941510 * x[21]
												 + 0.036154725527 * x[22] + 0.021151770407 * x[23]
												 + 0.200211885807 * x[24] - 0.236208723376 * x[25]
												 - 10.127341872304 * x[26] + 0.037053334254 * x[27])
	
		dx[11] = (0.002667394037 * x[0] + 0.004496152836 * x[1]
												 + 0.045956750452 * x[2] + 1.764514260408 * x[3]
												 + 0.000146052012 * x[4] + 0.000000019584 * x[5]
												 + 0.000000017540 * x[6] - 0.000000328097 * x[7]
												 - 17.119523267503 * x[8] + 0.536693033369 * x[9]
												 + 0.353775293385 * x[10] - 8.335731095093 * x[11]
												 + 0.078527228401 * x[12] + 0.005987264162 * x[13]
												 + 0.006725273267 * x[14] - 0.005979187005 * x[15]
												 - 0.182029763642 * x[16] + 3.233906041666 * x[17]
												 - 0.268381596955 * x[18] - 0.514971094398 * x[19]
												 - 0.005888452287 * x[20] - 0.003110464210 * x[21]
												 + 0.081894084826 * x[22] + 0.066024394813 * x[23]
												 - 0.104516302587 * x[24] + 1.631289796960 * x[25]
												 + 0.066233671911 * x[26] - 0.002829938571 * x[27])
	
		dx[12] = (  0.024056576806 * x[0] - 0.001361685819 * x[1]
												 - 0.230715295944 * x[2] + 0.185551143531 * x[3]
												 + 0.001832537128 * x[4] + 0.000000640359 * x[5]
												 + 0.000000573525 * x[6] - 0.000010727892 * x[7]
												 - 1.696796379292 * x[8] - 5.819307733117 * x[9]
												 - 2.712299197847 * x[10] - 0.615817527040 * x[11]
												 - 4.029675752634 * x[12] + 0.002306818331 * x[13]
												 - 0.004623901048 * x[14] + 0.071938991843 * x[15]
												 + 2.282021405408 * x[16] + 0.349879770769 * x[17]
												 + 2.900759066988 * x[18] + 4.516150272075 * x[19]
												 + 0.111029828612 * x[20] + 0.137305059460 * x[21]
												 + 0.003655620040 * x[22] + 0.001262406662 * x[23]
												 + 1.082630189953 * x[24] - 0.024777388732 * x[25]
												 - 1.766450614425 * x[26] - 0.930338103031 * x[27])
	
		dx[13] = (- 1.753103616578 * x[0] + 0.521869609890 * x[1]
												 + 23.319318958026 * x[2] + 145.082271971311 * x[3]
												 - 0.138741289403 * x[4] - 0.000051341929 * x[5]
												 - 0.000045983385 * x[6] + 0.000860128319 * x[7]
												 - 11.594360544437 * x[8] - 0.705424902410 * x[9]
												 - 10.592707880324 * x[10] - 54.888617486514 * x[11]
												 - 0.619258600252 * x[12] - 0.018180886764 * x[13]
												 - 0.016310350542 * x[14] + 0.172267463350 * x[15]
												 + 3.857750758541 * x[16] - 3.733629238750 * x[17]
												 + 0.251977753557 * x[18] + 1.659474556422 * x[19]
												 + 0.007076928248 * x[20] + 0.007594883320 * x[21]
												 - 0.096433822422 * x[22] - 0.077989008913 * x[23]
												 + 1.068213380174 * x[24] - 1.854605830991 * x[25]
												 - 0.041177323469 * x[26] + 0.038137029879 * x[27])
	 
		dx[14] = (1.708539622488 * x[0] + 0.111898315003 * x[1]
												 - 13.174473231922 * x[2] + 91.462755556230 * x[3]
												 + 0.127584976026 * x[4] + 0.000043171229 * x[5]
												 + 0.000038665459 * x[6] - 0.000723245056 * x[7]
												 - 1.878010842263 * x[8] + 23.870898681235 * x[9]
												 + 1.639719754761 * x[10] - 40.888303474223 * x[11]
												 + 2.851614162302 * x[12] + 0.001349430570 * x[13]
												 - 0.024984412428 * x[14] + 0.102862439056 * x[15]
												 + 1.794950045519 * x[16] + 0.252919074168 * x[17]
												 + 3.893644396914 * x[18] - 2.155538119928 * x[19]
												 + 0.123550997381 * x[20] + 0.137525326941 * x[21]
												 + 0.007512594224 * x[22] + 0.004476043338 * x[23]
												 - 0.081084731931 * x[24] - 0.050062181420 * x[25]
												 - 2.033833968448 * x[26] + 1.145542115841 * x[27])
	
		dx[15] = (-0.069753861204 * x[0] + 0.041269247265 * x[1]
												 + 1.243498527057 * x[2] + 13.467483657041 * x[3]
												 - 0.005772466581 * x[4] - 0.000002269708 * x[5]
												 - 0.000002032820 * x[6] + 0.000038024292 * x[7]
												 - 5.161896992464 * x[8] - 0.784811430978 * x[9]
												 - 1.913888711445 * x[10] - 8.087612492321 * x[11]
												 + 11.488701354150 * x[12] + 0.194411237470 * x[13]
												 + 0.167838434014 * x[14] - 3.255004272242 * x[15]
												 - 71.490067651024 * x[16] + 0.531333931032 * x[17]
												 + 1.089774627294 * x[18] - 20.947639012098 * x[19]
												 - 0.006423930487 * x[20] - 0.078948253623 * x[21]
												 + 0.050383537787 * x[22] + 0.045369546582 * x[23]
												 - 18.580601832107 * x[24] - 0.295436370828 * x[25]
												 - 0.680521274763 * x[26] - 0.683600561672 * x[27])
	
		dx[16] = (-0.013549327978 * x[5] - 0.012135188033 * x[6] + 0.226991094595 * x[7])
	
		dx[17] = (-11.385989897412 * x[0])
	
		dx[18] = (-4.554395958965 * x[1])
	
		dx[19] = (0.243569095885 * x[3] - 4.554395958965 * x[4])
	 
		dx[20] = (-4.554395958965 * x[2] - 8.500000000000 * x[20] - 18.000000000000 * x[21])
	
		dx[21] = (1.000000000000 * x[20])
	 
		dx[22] = (-11.385989897412 * x[3] - 8.500000000000 * x[22] - 18.000000000000 * x[23])
	
		dx[23] = (1.000000000000 * x[22])
	
		dx[24] = (0.683186075980 * x[8] + 0.514736886625 * x[9]
												 + 0.282998565164 * x[10] + 0.440668616363 * x[11]
												 - 2.382738811465 * x[11] - 0.037424700426 * x[13]
												 - 0.035002491999 * x[14] + 0.614952694278 * x[15]
												 + 14.829958398888 * x[16] - 0.110759742503 * x[17]
												 - 0.226034186438 * x[18] + 4.345468653096 * x[18]
												 + 0.001333027828 * x[20] + 0.016376955559 * x[20]
												 - 0.010465240818 * x[22] - 0.009422482600 * x[23]
												 - 6.145615181050 * x[24] + 0.061014181775 * x[25]
												 + 0.141165339638 * x[26] + 0.141806743312 * x[27])
	
		dx[25] = (-36.039354729710 * x[8] + 0.767400874818 * x[9] 
												 - 0.190879388177 * x[10] - 11.678174370212 * x[11]
												 - 0.041149877278 * x[12] - 0.026017271417 * x[13]
												 - 0.026698725144 * x[14] + 0.036415219598 * x[15]
												 + 0.738656358350 * x[16] + 6.810845841283 * x[17]
												 - 0.384784957980 * x[18] - 0.708557300741 * x[19]
												 - 0.005524328707 * x[20] + 0.002522572903 * x[21]
												 + 0.171826920583 * x[22] + 0.138368426838 * x[23]
												 + 0.071909684799 * x[24] - 6.567495145681 * x[25]
												 + 0.039293511274 * x[26] + 0.006041152866 * x[27])
	
		dx[26] = (1.997224587333 * x[8] + 13.482210983798 * x[9]
												 + 2.488520358003 * x[10] + 0.076750797248 * x[11]
												 + 0.804972334222 * x[12] + 0.023466195202 * x[13]
												 - 0.022740687251 * x[14] + 0.018646161041 * x[15]
												 + 0.436604617107 * x[16] - 0.414374632569 * x[17]
												 - 6.563020897889 * x[18] - 1.423460802051 * x[19]
												 - 0.224998971426 * x[20] - 0.259852011779 * x[21]
												 - 0.008437464875 * x[22] - 0.003945344110 * x[23]
												 + 0.102235829031 * x[24] + 0.191829027845 * x[25]
												 - 6.312428841540 * x[26] - 0.038075090345 * x[27])
	
		dx[27] = (1.761524247668 * x[8] - 3.415298165208 * x[9]
												 - 1.836225244248 * x[10] - 0.015605131825 * x[11]
												 + 10.486845595600 * x[12] - 0.031379180918 * x[13]
												 + 0.001266746410 * x[14] + 0.525873993847 * x[15]
												 + 9.808565668907 * x[16] - 0.367529750255 * x[17]
												 + 1.370405524130 * x[18] - 12.076970057329 * x[19]
												 + 0.004883176776 * x[20] - 0.015765473705 * x[21]
												 - 0.000399777933 * x[22] - 0.000497333312 * x[23]
												 + 0.199818976539 * x[24] - 0.002648145523 * x[25]
												 - 0.101212258081 * x[26] - 5.199268943788 * x[27])
		if len(x) > 28:
				dx[28] = x[7]

		return dx

def mhe_derivative(x):

	dx = vertcat((0.998573780060 * x[3] + 0.053384274244 * x[4]),

				(x[2] - 0.003182219341 * x[3] + 0.059524655342 * x[4]),

				(-11.570495605469 * x[2] - 2.544637680054 * x[3] 
											 - 0.063602626324 * x[4] + 0.106780529022 * x[5]
											 - 0.094918668270 * x[6] + 0.007107574493 * x[7]
											 - 3.700790207851 * x[8] - 16.213284674534 * x[9] 
											 - 2.984968535139 * x[10] - 0.493137919288 * x[11] 
											 - 1.186954196152 * x[12] - 0.031106608756 * x[13] 
											 + 0.024595252653 * x[14] + 0.008231369923 * x[15] 
											 + 0.231787619674 * x[16] + 0.745302732591 * x[17]
											 + 7.895709940231 * x[18] + 2.026930360369 * x[19]  
											 + 0.271792657736 * x[20] + 0.315196108541 * x[21]  
											 + 0.015876847710 * x[22] + 0.009288507454 * x[23] 
											 + 0.087920280806 * x[24] - 0.103727794204 * x[25]  
											 - 4.447282126346 * x[26] + 0.016271459306 * x[27]),
	
				(0.439356565475 * x[2] - 1.998182296753 * x[3] 
											 + 0.016651883721 * x[5] + 0.018462046981 * x[6]
											 - 0.001187470742 * x[7] - 7.517319654181 * x[8]
											 + 0.236494174025 * x[9] - 0.028725044803 * x[10]
											 - 2.442989538035 * x[11] + 0.034510550810 * x[12]
											 - 0.004683216652 * x[13] - 0.005154038690 * x[14]
											 - 0.002104275246 * x[15] - 0.079935853309 * x[16]
											 + 1.420125114638 * x[17] - 0.117856066698 * x[18]
											 - 0.226142434271 * x[19] - 0.002585832387 * x[20]
											 - 0.001365917341 * x[21] + 0.035962654791 * x[22]
											 + 0.028993699893 * x[23] - 0.045896888864 * x[24]
											 + 0.716358354284 * x[25] + 0.029085601036 * x[26]
											 - 0.001242728387 * x[27]),
	
				(-2.040895462036 * x[2]- 0.458999156952 * x[3]
											 - 0.735027790070 * x[4] + 0.019255757332 * x[5]
											 - 0.004595622420 * x[6] + 0.002120360732 * x[7]
											 - 0.740775522612 * x[8] - 2.555714688932 * x[9]
											 - 0.339301128908 * x[10] - 0.033104023297 * x[11]
											 - 1.446467788369 * x[12] - 0.007442776396 * x[13]
											 - 0.000012314482 * x[14] + 0.030657946816 * x[15]
											 + 1.002118140789 * x[16] + 0.153644862643 * x[17]
											 + 1.273828227991 * x[18] + 1.983204935524 * x[19]
											 + 0.048757213739 * x[20] + 0.060295617991 * x[21]
											 + 0.001605314985 * x[22] + 0.000554368427 * x[23]
											 + 0.475422075598 * x[24] - 0.010880647601 * x[25]
											 - 0.775712358056 * x[26]- 0.408545111762 * x[27]),
	
				(-32.103607177734 * x[0] - 0.503355026245 * x[2] 
											 + 2.297859191895 * x[3] - 0.021215811372 * x[5]
											 - 0.021167919040 * x[6] + 0.015811592340 * x[7]
											 + 8.689411857722 * x[8] - 0.215429806172 * x[9]
											 + 0.063500560122 * x[10] + 2.847523923644 * x[11]
											 - 0.297021616015 * x[12] + 0.001323463163 * x[13]
											 + 0.002124820781 * x[14] + 0.068860932948 * x[15]
											 + 1.694077894544 * x[16] - 1.639571645676 * x[17]
											 + 0.110652545728 * x[18] + 0.728735301618 * x[19]
											 + 0.003107735169 * x[20] + 0.003335187976 * x[21]
											 - 0.042347579477 * x[22] - 0.034247794709 * x[23]
											 + 0.469091132962 * x[24] - 0.814424502262 * x[25]
											 - 0.018082452136 * x[26]+ 0.016747349252 * x[27]),
		
				(0.102161169052 * x[0] + 32.057830810547 * x[1]
											 - 2.347217559814 * x[2] - 0.503611564636 * x[3]
											 + 0.834947586060 * x[4] + 0.021226570010 * x[5]
											 - 0.037879735231 * x[6] + 0.000354003860 * x[7]
											 - 0.560681623936 * x[8] - 3.574948145694 * x[9]
											 - 0.788176766644 * x[10] - 0.107590635594 * x[11]
											 + 0.908657075077 * x[12] - 0.008720966051 * x[13]
											 + 0.005669792925 * x[14] + 0.044884407612 * x[15]
											 + 0.788227489086 * x[16] + 0.111065913657 * x[17]
											 + 1.709840089441 * x[18] - 0.946574755181 * x[19]
											 + 0.054255711842 * x[20] + 0.060392345409 * x[21]
											 + 0.003299051857 * x[22] + 0.001965592530 * x[23]
											 - 0.035607238660 * x[24] - 0.021984114632 * x[25]
											 - 0.893130060176 * x[26] + 0.503048977806 * x[27]),
	
				(-1.910972595215 * x[0] + 1.713829040527 * x[1]
											 - 0.004005432129 * x[2] - 0.057411193848 * x[3]
											 + 0.013989634812 * x[5] - 0.000906753354 * x[6]
											 - 0.290513515472 * x[7] - 1.440209153996 * x[8]
											 - 1.089782421583 * x[9] - 0.599051729911 * x[10]
											 - 0.930901394778 * x[11] + 5.044060722850 * x[12]
											 + 0.079229241316 * x[13] + 0.074101747848 * x[14]
											 - 1.301808243838 * x[15] - 31.393874531397 * x[16]
											 + 0.233327947688 * x[17] + 0.478559456452 * x[18]
											 - 9.198865975131 * x[19] - 0.002820980233 * x[20]
											 - 0.034669033757 * x[21] + 0.022125233836 * x[22]
											 + 0.019923408940 * x[23] - 8.159414332666 * x[24]
											 - 0.129736796488 * x[25] - 0.298841506489 * x[26]
											 - 0.300193732750 * x[27]),
	
				(0.050176870833 * x[0] - 0.003161246171 * x[1]
											 - 0.486165175190 * x[2] + 0.266534777047 * x[3]
											 + 0.003826227932 * x[4] + 0.000001339204 * x[5]
											 + 0.000001199431 * x[6] - 0.000022435600 * x[7]
											 - 0.020657323970 * x[8] + 0.001301453941 * x[9]
											 + 0.213359280279 * x[10] + 0.881596311923 * x[11]
											 + 0.051809053856 * x[12] - 0.000000551337 * x[13]
											 - 0.000000493794 * x[14] + 0.000009236516 * x[15]),
	
				(-0.019757788570 * x[0] + 0.009012833714 * x[1]
												 + 0.311015942657 * x[2] + 2.810181204790 * x[3]
												 - 0.001602140073 * x[4] - 0.000000613279 * x[5]
												 - 0.000000549271 * x[6] + 0.000010274224 * x[7]
												 + 0.008134087133 * x[8] - 0.003710494952 * x[9]
												 + 0.863507011470 * x[10] - 1.236460821044 * x[11]
												 + 0.060184240645 * x[12] + 0.000000252481 * x[13]
												 + 0.000000226129 * x[14] - 0.000004229797 * x[15]),
	 
				(-0.030385323449 * x[0] + 0.003110159427 * x[1]
												 + 0.312812882924 * x[2] + 0.287354391281 * x[3]
												 - 0.002331730630 * x[4] - 0.000000824205 * x[5]
												 - 0.000000738183 * x[6] + 0.000013807861 * x[7]
												 - 8.414922645141 * x[8] - 36.922139523656 * x[9]
												 - 18.505141519315 * x[10] - 3.793715804769 * x[11]
												 - 2.765572372983 * x[12] + 0.035944961732 * x[13]
												 - 0.038910104720 * x[14] + 0.025846348888 * x[15]
												 + 0.527826299191 * x[16] + 1.697201876759 * x[17]
												 + 17.980094722474 * x[18] + 4.615721721183 * x[19]
												 + 0.618925691035 * x[20] + 0.717763941510 * x[21]
												 + 0.036154725527 * x[22] + 0.021151770407 * x[23]
												 + 0.200211885807 * x[24] - 0.236208723376 * x[25]
												 - 10.127341872304 * x[26] + 0.037053334254 * x[27]),
	
				(0.002667394037 * x[0] + 0.004496152836 * x[1]
												 + 0.045956750452 * x[2] + 1.764514260408 * x[3]
												 + 0.000146052012 * x[4] + 0.000000019584 * x[5]
												 + 0.000000017540 * x[6] - 0.000000328097 * x[7]
												 - 17.119523267503 * x[8] + 0.536693033369 * x[9]
												 + 0.353775293385 * x[10] - 8.335731095093 * x[11]
												 + 0.078527228401 * x[12] + 0.005987264162 * x[13]
												 + 0.006725273267 * x[14] - 0.005979187005 * x[15]
												 - 0.182029763642 * x[16] + 3.233906041666 * x[17]
												 - 0.268381596955 * x[18] - 0.514971094398 * x[19]
												 - 0.005888452287 * x[20] - 0.003110464210 * x[21]
												 + 0.081894084826 * x[22] + 0.066024394813 * x[23]
												 - 0.104516302587 * x[24] + 1.631289796960 * x[25]
												 + 0.066233671911 * x[26] - 0.002829938571 * x[27]),
	
				(0.024056576806 * x[0] - 0.001361685819 * x[1]
												 - 0.230715295944 * x[2] + 0.185551143531 * x[3]
												 + 0.001832537128 * x[4] + 0.000000640359 * x[5]
												 + 0.000000573525 * x[6] - 0.000010727892 * x[7]
												 - 1.696796379292 * x[8] - 5.819307733117 * x[9]
												 - 2.712299197847 * x[10] - 0.615817527040 * x[11]
												 - 4.029675752634 * x[12] + 0.002306818331 * x[13]
												 - 0.004623901048 * x[14] + 0.071938991843 * x[15]
												 + 2.282021405408 * x[16] + 0.349879770769 * x[17]
												 + 2.900759066988 * x[18] + 4.516150272075 * x[19]
												 + 0.111029828612 * x[20] + 0.137305059460 * x[21]
												 + 0.003655620040 * x[22] + 0.001262406662 * x[23]
												 + 1.082630189953 * x[24] - 0.024777388732 * x[25]
												 - 1.766450614425 * x[26] - 0.930338103031 * x[27]),
	
				(- 1.753103616578 * x[0] + 0.521869609890 * x[1]
												 + 23.319318958026 * x[2] + 145.082271971311 * x[3]
												 - 0.138741289403 * x[4] - 0.000051341929 * x[5]
												 - 0.000045983385 * x[6] + 0.000860128319 * x[7]
												 - 11.594360544437 * x[8] - 0.705424902410 * x[9]
												 - 10.592707880324 * x[10] - 54.888617486514 * x[11]
												 - 0.619258600252 * x[12] - 0.018180886764 * x[13]
												 - 0.016310350542 * x[14] + 0.172267463350 * x[15]
												 + 3.857750758541 * x[16] - 3.733629238750 * x[17]
												 + 0.251977753557 * x[18] + 1.659474556422 * x[19]
												 + 0.007076928248 * x[20] + 0.007594883320 * x[21]
												 - 0.096433822422 * x[22] - 0.077989008913 * x[23]
												 + 1.068213380174 * x[24] - 1.854605830991 * x[25]
												 - 0.041177323469 * x[26] + 0.038137029879 * x[27]),
	 
				(1.708539622488 * x[0] + 0.111898315003 * x[1]
												 - 13.174473231922 * x[2] + 91.462755556230 * x[3]
												 + 0.127584976026 * x[4] + 0.000043171229 * x[5]
												 + 0.000038665459 * x[6] - 0.000723245056 * x[7]
												 - 1.878010842263 * x[8] + 23.870898681235 * x[9]
												 + 1.639719754761 * x[10] - 40.888303474223 * x[11]
												 + 2.851614162302 * x[12] + 0.001349430570 * x[13]
												 - 0.024984412428 * x[14] + 0.102862439056 * x[15]
												 + 1.794950045519 * x[16] + 0.252919074168 * x[17]
												 + 3.893644396914 * x[18] - 2.155538119928 * x[19]
												 + 0.123550997381 * x[20] + 0.137525326941 * x[21]
												 + 0.007512594224 * x[22] + 0.004476043338 * x[23]
												 - 0.081084731931 * x[24] - 0.050062181420 * x[25]
												 - 2.033833968448 * x[26] + 1.145542115841 * x[27]),
	
				(-0.069753861204 * x[0] + 0.041269247265 * x[1]
												 + 1.243498527057 * x[2] + 13.467483657041 * x[3]
												 - 0.005772466581 * x[4] - 0.000002269708 * x[5]
												 - 0.000002032820 * x[6] + 0.000038024292 * x[7]
												 - 5.161896992464 * x[8] - 0.784811430978 * x[9]
												 - 1.913888711445 * x[10] - 8.087612492321 * x[11]
												 + 11.488701354150 * x[12] + 0.194411237470 * x[13]
												 + 0.167838434014 * x[14] - 3.255004272242 * x[15]
												 - 71.490067651024 * x[16] + 0.531333931032 * x[17]
												 + 1.089774627294 * x[18] - 20.947639012098 * x[19]
												 - 0.006423930487 * x[20] - 0.078948253623 * x[21]
												 + 0.050383537787 * x[22] + 0.045369546582 * x[23]
												 - 18.580601832107 * x[24] - 0.295436370828 * x[25]
												 - 0.680521274763 * x[26] - 0.683600561672 * x[27]),
	
			(-0.013549327978 * x[5] - 0.012135188033 * x[6] + 0.226991094595 * x[7]),
	
			(-11.385989897412 * x[0]),
	
			(-4.554395958965 * x[1]),
	
			(0.243569095885 * x[3] - 4.554395958965 * x[4]),
	 
			(-4.554395958965 * x[2] - 8.500000000000 * x[20] - 18.000000000000 * x[21]),
	
			(1.000000000000 * x[20]),
	 
			(-11.385989897412 * x[3] - 8.500000000000 * x[22] - 18.000000000000 * x[23]),
	
			(1.000000000000 * x[22]),
	
			(0.683186075980 * x[8] + 0.514736886625 * x[9]
												 + 0.282998565164 * x[10] + 0.440668616363 * x[11]
												 - 2.382738811465 * x[11] - 0.037424700426 * x[13]
												 - 0.035002491999 * x[14] + 0.614952694278 * x[15]
												 + 14.829958398888 * x[16] - 0.110759742503 * x[17]
												 - 0.226034186438 * x[18] + 4.345468653096 * x[18]
												 + 0.001333027828 * x[20] + 0.016376955559 * x[20]
												 - 0.010465240818 * x[22] - 0.009422482600 * x[23]
												 - 6.145615181050 * x[24] + 0.061014181775 * x[25]
												 + 0.141165339638 * x[26] + 0.141806743312 * x[27]),
	
			(-36.039354729710 * x[8] + 0.767400874818 * x[9] 
												 - 0.190879388177 * x[10] - 11.678174370212 * x[11]
												 - 0.041149877278 * x[12] - 0.026017271417 * x[13]
												 - 0.026698725144 * x[14] + 0.036415219598 * x[15]
												 + 0.738656358350 * x[16] + 6.810845841283 * x[17]
												 - 0.384784957980 * x[18] - 0.708557300741 * x[19]
												 - 0.005524328707 * x[20] + 0.002522572903 * x[21]
												 + 0.171826920583 * x[22] + 0.138368426838 * x[23]
												 + 0.071909684799 * x[24] - 6.567495145681 * x[25]
												 + 0.039293511274 * x[26] + 0.006041152866 * x[27]),
	
			(1.997224587333 * x[8] + 13.482210983798 * x[9]
												 + 2.488520358003 * x[10] + 0.076750797248 * x[11]
												 + 0.804972334222 * x[12] + 0.023466195202 * x[13]
												 - 0.022740687251 * x[14] + 0.018646161041 * x[15]
												 + 0.436604617107 * x[16] - 0.414374632569 * x[17]
												 - 6.563020897889 * x[18] - 1.423460802051 * x[19]
												 - 0.224998971426 * x[20] - 0.259852011779 * x[21]
												 - 0.008437464875 * x[22] - 0.003945344110 * x[23]
												 + 0.102235829031 * x[24] + 0.191829027845 * x[25]
												 - 6.312428841540 * x[26] - 0.038075090345 * x[27]),
	
			(1.761524247668 * x[8] - 3.415298165208 * x[9]
												 - 1.836225244248 * x[10] - 0.015605131825 * x[11]
												 + 10.486845595600 * x[12] - 0.031379180918 * x[13]
												 + 0.001266746410 * x[14] + 0.525873993847 * x[15]
												 + 9.808565668907 * x[16] - 0.367529750255 * x[17]
												 + 1.370405524130 * x[18] - 12.076970057329 * x[19]
												 + 0.004883176776 * x[20] - 0.015765473705 * x[21]
												 - 0.000399777933 * x[22] - 0.000497333312 * x[23]
												 + 0.199818976539 * x[24] - 0.002648145523 * x[25]
												 - 0.101212258081 * x[26] - 5.199268943788 * x[27]),
			x[7]

			)

	return dx