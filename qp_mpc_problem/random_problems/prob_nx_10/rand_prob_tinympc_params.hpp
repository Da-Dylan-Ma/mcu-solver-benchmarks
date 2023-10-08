#include "types.hpp"

#pragma once

const PROGMEM tinytype Adyn_data[10*10] = {
  -0.013073761504971462,	0.2246110094471291,	0.2544408073658474,	0.13233480195824593,	-0.05119190975443441,	-0.2853635217317816,	0.005852846280463135,	0.07011807025248935,	-0.2968322415804054,	-0.013342503116044284,	
  -0.061372489552688154,	0.2556656045343441,	-0.030632565028865355,	0.11552939765512761,	-0.05896912724333504,	-0.07512087891577876,	-0.27025013918333696,	0.12774698682523422,	0.2908888109041684,	-0.020948991811370842,	
  -0.0037584741172127212,	-0.16552794408937865,	-0.08875096809930894,	0.04237298440223152,	0.21786326884394444,	-0.22975411646799354,	-0.0777467299409002,	-0.1137542144340093,	-0.05818066723226998,	-0.40988281522310976,	
  0.12242331773993005,	0.12471677936292301,	-0.1391130349990805,	-0.265821477005358,	-0.06658710699508395,	0.14922853420923898,	0.2972460153275471,	0.03912968497346173,	-0.25106147488514546,	-0.005822319071533367,	
  -0.14332453646315435,	-0.2013761743702378,	-0.003491287843568522,	-0.4031924893388929,	0.14221711686705377,	0.01480509677146434,	-0.24551101231525008,	0.20304991085071084,	-0.07610204893228215,	0.11618405552812039,	
  0.3210840769778933,	0.03371740843856074,	-0.07351063257254406,	0.12192570383079772,	-0.09990897508378217,	0.16135562872560227,	-0.0966134567993955,	-0.30496644443426796,	-0.26972984451576165,	-0.08068945894468343,	
  -0.2453297827460566,	0.07817106155392896,	0.21751900328259138,	0.16456053729640224,	0.039794460418855364,	-0.34110129159981173,	0.063771323499338,	-0.4229502226809963,	0.1311149585415761,	-0.2636762989898568,	
  0.14232115802608258,	-0.234146837739637,	-0.0841343426124534,	-0.3162995888947142,	0.0583712712361423,	-0.0007534977634417858,	0.2791294857680359,	0.04530726363395114,	0.1648116997197439,	0.1254814841374427,	
  0.1957264057858731,	0.02768687137406536,	0.04362125841822115,	-0.08965321413277916,	0.3543881498570771,	-0.10454970854675331,	-0.14243354594504237,	-0.21015125400397933,	-0.1790609777910215,	-0.13920921968532368,	
  0.17549826346412822,	0.03576707299363247,	0.13962148612526296,	-0.10341080386062358,	0.3449722760724923,	-0.07652298009339971,	-0.1685188116906205,	-0.08983973705155053,	0.2409963870089963,	-0.012089801465592988,	
};

const PROGMEM tinytype Bdyn_data[10*4] = {
  0.5847337443977123,	0.9156896162787989,	-0.6632152796182489,	-0.7287620922223179,	
  -0.06009044137962216,	0.7220150513443002,	0.5341385478526204,	0.1163562164376648,	
  -0.6864334687845264,	-0.2174748617885558,	0.4386915533094997,	-0.25317463011527597,	
  -0.921761299446676,	0.16776792214640257,	0.4402691593858987,	0.42954298328491336,	
  -0.6702683229305435,	-0.3280168211921326,	-0.18165541540616048,	-0.1590371745457484,	
  -0.7716830209906447,	-0.22893693859729036,	0.013264373087658488,	0.42112134423274306,	
  0.13889537768820004,	-0.19013887920220185,	0.8551947255090679,	0.19616713326347623,	
  0.949581767410826,	0.7347527127002798,	0.34725220325022876,	0.7986269908494112,	
  0.9824793562366523,	-0.5582454660717111,	0.38211303980760825,	0.8375585557182579,	
  -0.9651996799670686,	0.5989779930651686,	-0.8211941002179848,	0.8331085948904269,	
};

const PROGMEM tinytype Q_data[10] = {6.964691855978616,2.8613933495037944,2.268514535642031,5.513147690828912,7.1946896978556305,4.23106460124461,9.807641983846155,6.848297385848633,4.809319014843609,3.9211751819415053};

const PROGMEM tinytype Qf_data[10] = {62.682226703807544,25.752540145534148,20.416630820778277,49.618329217460214,64.75220728070067,38.07958141120149,88.2687778546154,61.63467647263769,43.283871133592484,35.29057663747355};

const PROGMEM tinytype R_data[4] = {0.1,0.1,0.1,0.1};

const PROGMEM tinytype umin[4] = {
  -3.0,	
  -3.0,	
  -3.0,	
  -3.0,	
};

const PROGMEM tinytype umax[4] = {
  3.0,	
  3.0,	
  3.0,	
  3.0,	
};

const PROGMEM tinytype xmin[10] = {
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
  -10000.0,	
};

const PROGMEM tinytype xmax[10] = {
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
  10000.0,	
};

const PROGMEM tinytype rho_value = 0.1;

const PROGMEM tinytype Kinf_data[4*10] = {
  -0.008452453586541038,	-0.051965163640795724,	0.05860774083414708,	0.03320744129071543,	0.03230669142947221,	-0.07626921391712352,	0.03411505699565515,	-0.023301382827595026,	0.01625446470763075,	0.026953436878575154,	
  0.02112026968770759,	0.142959732090482,	0.04624491863339418,	0.04360555118407825,	-0.13342882984021026,	-0.07934832996459597,	0.13980950366900735,	0.04570061653920564,	0.06899555395385716,	0.030147953229894234,	
  -0.15748046868266077,	0.1046979161826944,	-0.013037994217837268,	0.08392238951002158,	-0.09778157831169393,	-0.13274984692408023,	0.13482980351402127,	-0.13219121487168078,	0.05748429130581329,	-0.17704501952069607,	
  0.23452896353554112,	-0.09028998289651041,	-0.041091960815047304,	-0.16354086414344338,	0.17582796478397486,	0.09216821837701118,	0.020865757314786054,	-0.1401316239888512,	0.07041756221738185,	0.03420878918524333,	
};

const PROGMEM tinytype Pinf_data[10*10] = {
  7.997267518385562,	0.3332702706481152,	-0.5606554240100279,	0.23057030119764346,	-0.4095059594817465,	0.6567037282106823,	0.2825995091316401,	0.13116522606357348,	-1.026567813617536,	-0.02067997854847359,	
  0.33327027064811526,	4.523152906945522,	0.5565005670289022,	1.3742781160235882,	-0.21892349386605892,	-0.34556574333878887,	-0.1496239847864868,	-0.9142552851171374,	-0.7792229800927039,	-0.6010242665212405,	
  -0.5606554240100282,	0.5565005670289025,	3.5616819172737193,	0.8452744821911785,	0.3786756396834304,	-1.2833295744255397,	-0.3333556728793163,	-1.3691196648900703,	0.1069803736535712,	-0.7449148660654147,	
  0.23057030119764352,	1.374278116023588,	0.8452744821911783,	8.403286203060027,	-0.5095831554917835,	-0.6818575201069557,	-0.17637318584451014,	-2.0733625491701893,	0.28587575452982583,	-1.101259441774158,	
  -0.4095059594817468,	-0.21892349386605883,	0.37867563968343043,	-0.5095831554917835,	8.412027761745414,	-0.904302287537876,	-0.6577798213024726,	0.10608765164830804,	0.20443414810502922,	-0.4709773693281963,	
  0.6567037282106827,	-0.3455657433387893,	-1.28332957442554,	-0.6818575201069558,	-0.9043022875378757,	6.028073238385279,	0.5962710714181839,	1.3626538160531576,	-0.2148254274917998,	1.056198703226572,	
  0.28259950913164034,	-0.14962398478648692,	-0.3333556728793162,	-0.17637318584451017,	-0.6577798213024726,	0.596271071418184,	11.381418863694089,	0.001413915869429714,	-0.5293272753819517,	0.268651444773744,	
  0.13116522606357336,	-0.9142552851171375,	-1.3691196648900699,	-2.0733625491701893,	0.10608765164830844,	1.3626538160531572,	0.0014139158694297185,	9.40829951880905,	0.1326400202706865,	1.2339567054410323,	
  -1.0265678136175367,	-0.7792229800927044,	0.10698037365357105,	0.28587575452982583,	0.20443414810502897,	-0.21482542749179973,	-0.5293272753819518,	0.13264002027068647,	7.4036268525283715,	0.20331028154779315,	
  -0.02067997854847355,	-0.6010242665212407,	-0.7449148660654147,	-1.101259441774158,	-0.4709773693281962,	1.056198703226572,	0.26865144477374386,	1.2339567054410325,	0.20331028154779315,	5.075759598679111,	
};

const PROGMEM tinytype Quu_inv_data[4*4] = {
  0.031095226314116136,	-0.008088683240843095,	-0.002362725641254285,	-0.006461157088362438,	
  -0.0080886832408431,	0.05494430326237461,	0.020040987475059648,	-0.001324608340014836,	
  -0.0023627256412542884,	0.020040987475059654,	0.0580499865220362,	-0.012404046396119655,	
  -0.0064611570883624375,	-0.0013246083400148417,	-0.01240404639611966,	0.04411185143124605,	
};

const PROGMEM tinytype AmBKt_data[10*10] = {
  0.039001426765101745,	-0.020302067807155967,	0.12349474159564308,	0.07968239228064633,	-0.13337050087464133,	0.22272039924912812,	-0.15147039986180705,	0.0022135576992556583,	0.07956466236530325,	-0.17002074583704285,	
  0.12772739285397355,	0.10390652448836732,	-0.2388976481469717,	0.04552141046289816,	-0.18465410088908463,	0.06297982412782424,	0.040745807231163694,	-0.25409028149186774,	0.19416490282148063,	0.0611792521193472,	
  0.13923163150101342,	-0.048754926930511894,	-0.04314733001727812,	-0.06945811136722947,	0.042057163922223875,	-0.00021922137939056618,	0.23738253202130605,	-0.13642114292025762,	0.051255290044405336,	0.19201731207503453,	
  0.009464368747183469,	0.06024379610177814,	-0.0035696464659980692,	-0.20922836469619616,	-0.37739535949824116,	0.22529161449951937,	0.1285506454547674,	-0.2784061531556237,	0.006970972179027715,	0.1076861129864962,	
  0.11538319027947785,	0.07108005305201372,	0.29843337920143603,	-0.046898426837020196,	0.13030499762944087,	-0.1782731352152195,	0.05906782437671252,	0.01926453085028483,	0.13825885304428226,	0.2292941731125501,	
  -0.1889809200643069,	0.03776922418727309,	-0.2177932334860901,	0.11109408095331688,	-0.07179988384839076,	0.047281047055019024,	-0.25014845912825623,	0.1024614078078721,	-0.10038345812469959,	-0.28840962352680183,	
  -0.03749013070879856,	-0.443590372351265,	-0.07778997608818142,	0.23691226076891414,	-0.14897390625556342,	-0.04885533538491368,	-0.03378279049991477,	0.08052513425855676,	-0.16689938328109583,	-0.1259957880940369,	
  -0.14789825395840034,	0.18026387257038928,	-0.09729693089521886,	0.12837654545228772,	0.15612291543035267,	-0.25171931522876045,	-0.2704858556305404,	0.19167176775181144,	0.006134463124628287,	-0.13151367326823638,	
  -0.2800731776142531,	0.20315162343951426,	-0.03940819285547333,	-0.303209909460218,	-0.02093415089829217,	-0.2718077507317077,	0.07900215178966707,	0.02248317333765945,	-0.23745852763914035,	0.20389865726615025,	
  -0.1491981482585206,	0.04948954046096768,	-0.29849569992109526,	0.07721777301651253,	0.11741840885588463,	-0.06504562932474753,	-0.1169903819828636,	0.11189491222589193,	-0.10986111026723323,	-0.17802027513143875,	
};

const PROGMEM tinytype coeff_d2p_data[10*4] = {
  3.175916734460271e-11,	-4.161477065722785e-11,	1.3816794930399112e-12,	-1.7345534730761614e-11,	
  4.987930034283661e-11,	-6.826723908393006e-11,	-5.004816006071167e-13,	-1.4271049619818399e-11,	
  -1.2858545825333856e-11,	1.9381014154662424e-11,	1.0411012330013847e-12,	-1.7448941597164591e-12,	
  4.0857357427870333e-11,	-8.573920046162087e-11,	1.811274047414102e-11,	-2.060070170006867e-11,	
  -4.523732343580944e-11,	8.6824152034648e-11,	-4.922971752474581e-13,	-2.3486851352672033e-11,	
  3.8202916524676667e-11,	-6.540558372680749e-11,	-3.3296351786837874e-12,	1.8541449625653073e-11,	
  -1.5141444521804548e-11,	4.180990623159353e-11,	-3.5532098097146303e-12,	-1.7779405551976435e-11,	
  -1.5870759567659931e-12,	1.2853235836773891e-11,	-1.0906095471163724e-11,	1.652780690086786e-11,	
  -3.063843406411748e-11,	1.0032759345524056e-11,	1.318728459764884e-11,	2.0353278090690097e-11,	
  1.4729465043494816e-11,	-3.1185225408958406e-11,	-7.777750665738381e-12,	3.624485000325306e-11,	
};

