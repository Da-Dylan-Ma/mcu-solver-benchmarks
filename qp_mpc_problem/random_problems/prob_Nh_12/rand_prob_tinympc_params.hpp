#include "types.hpp"

#pragma once

const PROGMEM tinytype Adyn_data[10*10] = {
  0.3174763498576604,	-0.025408851238763533,	0.16346347031636216,	0.028007142833935593,	-0.12509099624391867,	0.09138594386083856,	-0.420724387954458,	0.04795929801330711,	-0.2730526297701459,	-0.0749809980675494,	
  0.06960878910026941,	-0.04242219753808094,	-0.3287179369447449,	0.2074096414229801,	-0.10411117117169953,	0.1379804508052085,	-0.26162806707833197,	0.016635713526822617,	0.0015544128195374904,	0.19114865053962893,	
  -0.3186944769074917,	-0.11470386582711972,	0.09467338606872015,	0.3068073058887873,	0.08868904717908528,	0.23945067271398854,	-0.01467962313003123,	0.19323115540126445,	-0.031618775889305846,	0.29459636211352397,	
  0.006186315023952666,	0.33834790424935735,	0.20094221925498823,	0.0353697284114278,	-0.14826336374049387,	-0.052096198335791224,	0.3363336933009938,	0.11895971091014915,	-0.11830929867937796,	0.09720142086878421,	
  0.012611739718915533,	0.22368967296705963,	0.2852285220955373,	0.12739581106635325,	-0.0017586314095066492,	0.29382948954698523,	0.14446857628903137,	-0.16227295209041992,	-0.11788241008450606,	0.13519521204932822,	
  0.255802157556296,	-0.03880533610537748,	0.5863800138486357,	-0.013112568559571417,	-0.24186774610753717,	0.18524894091944955,	-0.11243771451358478,	0.18963534260133078,	0.2266823299122907,	0.0799212122005919,	
  0.036493935143713505,	-0.1911074169206161,	0.21471363956883763,	0.1683066456907564,	-0.20364821390529436,	0.0834598111265319,	0.12975267477303049,	-0.17114179760593298,	0.09651841790937743,	0.0015156764026617708,	
  0.11366561958115226,	0.07385076968959797,	-0.280347310641782,	-0.04572192872401932,	0.27790083371021607,	0.030375530007739033,	0.07967087002653835,	0.19177601804119296,	-0.28265760023153486,	-0.016598454759533705,	
  -0.227512450374456,	-0.08366550715793959,	-0.08039656490694312,	0.09340301038636532,	-0.14198676889733705,	-0.3472956525397828,	0.09955176841295943,	-0.03192097998310104,	0.3891201275077618,	0.4395245846613375,	
  -0.32229145169105416,	0.12511736209582155,	0.22788746960475523,	-0.2634168916728832,	0.1164653626204203,	0.042953379097887996,	0.011351682045023946,	0.2896839714604237,	-0.13015942432945682,	0.11436388180322224,	
};

const PROGMEM tinytype Bdyn_data[10*4] = {
  -0.6476619582311063,	0.2585011982644909,	0.4617004833577003,	0.01700322733222004,	
  -0.22637905665505165,	0.28128713194716815,	0.9240537557908699,	-0.7776225888170152,	
  0.31441180670076796,	0.8476900018373994,	0.5458381587035994,	-0.5870052813589601,	
  -0.7251337499353925,	-0.13447186452273407,	-0.8034431042380106,	0.39869153844848815,	
  -0.5886509380176825,	-0.8142252444332378,	-0.7890789574184924,	0.7651036142974372,	
  0.5893512688172211,	0.48994751612098053,	-0.6075647372870316,	-0.3918951630521794,	
  -0.28527438369822566,	-0.6454365843427163,	-0.02569506597464133,	0.8690600773749577,	
  -0.26461070537876963,	-0.3233225916443261,	0.5037519691827408,	0.7786526138522463,	
  0.4843082345867016,	-0.4885646366089951,	0.9877435036170645,	0.6404228151153355,	
  0.3474545030739715,	0.9274723262809903,	-0.44269864219709576,	0.3131195722959339,	
};

const PROGMEM tinytype Q_data[10] = {6.244677783350655,8.563252583226546,8.438756710514145,5.419871874291245,3.6634454776868317,6.709488609683071,0.8676909940626487,7.318028998574291,2.4631624166225077,8.292315109173988};

const PROGMEM tinytype Qf_data[10] = {68.6914556168572,94.19577841549201,92.8263238156556,59.6185906172037,40.297900254555145,73.80437470651378,9.544600934689136,80.4983189843172,27.094786582847583,91.21546620091387};

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
  -0.22231698573119263,	-0.16051678406311415,	0.014020008384832083,	-0.0827054582032448,	0.041553071225099,	-0.15180171120510563,	0.14220146002671624,	-0.021272167685638322,	0.34906863523325066,	0.06723821606768218,	
  -0.15411555977945182,	0.08578602444359475,	0.22313945314575778,	-0.04621883243729519,	0.05237833367359852,	0.12303771910809631,	-0.07764101052625984,	0.3338651053641106,	-0.21684931530566948,	0.1445027683997183,	
  -0.04722011070580811,	-0.11159294302507852,	-0.32428862922934626,	0.07687490717675009,	0.09411979572769101,	-0.0658481791899941,	-0.11475042041419334,	0.01686648366499923,	-0.044324267566786285,	0.07384534191721505,	
  -0.19057016225948395,	0.09421622894326553,	0.017357947684513646,	-0.14956442418239285,	0.15861584431701067,	-0.0866737571722505,	0.1737098862174581,	0.17289945695201564,	-0.14235540842977698,	0.08197519212165695,	
};

const PROGMEM tinytype Pinf_data[10*10] = {
  8.74269780755106,	-0.26657922916958987,	0.9716997952674612,	-0.5998141181976304,	-0.47198682913492573,	0.26257331805591116,	-1.4570107283184606,	0.36810430652692916,	-0.23345773554925436,	-0.9210212041454243,	
  -0.2665792291695898,	8.99042294193351,	-0.3153572365866002,	0.0035139989421602764,	0.07165856393730904,	-0.10431423674104516,	0.33776602494666186,	0.1894994855684259,	0.040528866257946626,	0.34482248097404916,	
  0.9716997952674615,	-0.3153572365866,	10.52355062021785,	0.7104965598773243,	-0.9955223204108148,	0.6400565035887756,	-0.6750616090983422,	0.11670927534268241,	0.39580023698275274,	0.5225255330601948,	
  -0.5998141181976303,	0.003513998942160192,	0.7104965598773239,	6.643824397554063,	-0.2289854125368821,	0.6800360806085191,	0.660512793137007,	0.15424526709352193,	0.28055907210006414,	1.047241525714819,	
  -0.47198682913492596,	0.07165856393730904,	-0.995522320410815,	-0.22898541253688215,	4.939483225244152,	0.18032735386128076,	0.2727118461969935,	0.000901730468796696,	-0.7431192781908074,	-0.5401976966535381,	
  0.2625733180559115,	-0.10431423674104517,	0.6400565035887756,	0.6800360806085186,	0.18032735386128068,	8.147465502392231,	-0.23056051060717436,	0.2064613689930094,	-0.2766624239759405,	0.4426152079579008,	
  -1.4570107283184606,	0.3377660249466618,	-0.6750616090983426,	0.6605127931370067,	0.2727118461969938,	-0.23056051060717433,	2.5334487921200144,	0.19506774818983083,	0.2399047921812733,	0.8288062641998394,	
  0.3681043065269294,	0.18949948556842627,	0.11670927534268241,	0.15424526709352138,	0.0009017304687966418,	0.20646136899300924,	0.1950677481898308,	7.880228948100715,	-0.05467522992612894,	0.22100952059720164,	
  -0.23345773554925459,	0.0405288662579463,	0.39580023698275296,	0.2805590721000641,	-0.7431192781908073,	-0.27666242397594054,	0.23990479218127359,	-0.05467522992612883,	3.20942432138564,	0.6176415055951708,	
  -0.9210212041454247,	0.3448224809740493,	0.5225255330601947,	1.047241525714819,	-0.5401976966535379,	0.44261520795790116,	0.8288062641998395,	0.22100952059720222,	0.6176415055951705,	10.00810676054556,	
};

const PROGMEM tinytype Quu_inv_data[4*4] = {
  0.09212320793388805,	-0.03168600030295621,	0.007136359779158838,	0.006304615030296932,	
  -0.031686000302956194,	0.06253626651517324,	0.0007395405361745734,	0.030584472257818643,	
  0.007136359779158832,	0.0007395405361745745,	0.03661693038867899,	0.01836587769205959,	
  0.00630461503029693,	0.030584472257818646,	0.018365877692059593,	0.06952978764505038,	
};

const PROGMEM tinytype AmBKt_data[10*10] = {
  0.23837100813386214,	-0.04192613889009411,	-0.20424432604517587,	-0.1377074023434733,	-0.1351946176043488,	0.3589606919189317,	0.03800436965415496,	0.17718430168720284,	-0.02645107857483936,	-0.06334152833897566,	
  -0.1016247202721756,	0.07349220551303934,	-0.020738338437706658,	0.10626577953169472,	0.038909643754809875,	-0.01711226805359024,	-0.2662762562149725,	0.04196632184529759,	0.08587309296475072,	0.022422488218372935,	
  0.2642909561151321,	-0.0751522994572883,	0.08831056168678411,	-0.026353350981461854,	0.2059972269974163,	0.2785668432160232,	0.3393178382101161,	-0.054646215822273414,	0.33102903652002386,	-0.1329367621305527,	
  -0.04656051651135037,	0.014346399202550292,	0.2422337558978338,	0.0905767612250146,	0.21617127849609258,	0.04636770870232809,	0.24683734759690257,	-0.004817427086357365,	0.13072883718623632,	-0.11094985419590664,	
  -0.1578706512218941,	-0.07306626461311161,	0.07295784353695765,	-0.09870731133537403,	0.018259778996518317,	-0.17267508470315945,	-0.293415577915228,	0.13491164557437993,	-0.33106848423778157,	0.04544908636834784,	
  -0.006859777745083631,	0.062454581352457864,	0.1679455743251584,	-0.1639768028401832,	0.3190066790627889,	0.14045738927083895,	0.19320044549050502,	0.1306479438487133,	-0.09311599246874636,	-0.020428162074561364,	
  -0.2585289217492056,	0.03351907085574418,	0.1710298460864038,	0.2675561440376125,	-0.05849487554048477,	-0.15984655000108697,	-0.03370608677594594,	0.014741899221861157,	-0.005154013785250905,	-0.07123878752225,	
  -0.0628495179939644,	0.03677316807048453,	0.009191788283623104,	0.09304771690867819,	-0.021930448607776376,	0.11660168411546369,	-0.11154807892122062,	0.15096816742213817,	0.014107473251114354,	-0.05924699861703603,	
  0.03196669106299471,	0.0718323878210476,	-0.01691857964941239,	0.12679588768890282,	-0.015025726617367577,	0.044484812262225115,	0.17871276359129518,	-0.12722873711858543,	0.24906706485230193,	-0.025371183070798442,	
  -0.1042757750491913,	0.16123190075276825,	0.15877458636497854,	0.19203739260220642,	0.28798314256498675,	0.046487018842603915,	0.0446204844407273,	-0.053115427421015576,	0.3521204654137592,	-0.035997462406563346,	
};

const PROGMEM tinytype coeff_d2p_data[10*4] = {
  4.730143360331951e-11,	-6.665280480322444e-11,	2.3262641812848983e-11,	7.990392375534228e-11,	
  3.4851288521764445e-12,	-4.903837752534557e-12,	1.7158982568155068e-12,	5.887047793695643e-12,	
  1.7746889461461857e-11,	-2.5043155049697674e-11,	8.72127370321607e-12,	2.9975983934643624e-11,	
  -5.629149923969123e-12,	7.903034129896902e-12,	-2.773923452048521e-12,	-9.515666032910985e-12,	
  2.3698831091989447e-11,	-3.336975314127688e-11,	1.1662053267524897e-11,	4.00379382425875e-11,	
  3.408129334303567e-11,	-4.80259859658716e-11,	1.6763467350355832e-11,	5.757152046759195e-11,	
  -3.391256372942131e-11,	4.776230626279965e-11,	-1.668286689349152e-11,	-5.7290186483704986e-11,	
  1.581885056972654e-11,	-2.2303880964358314e-11,	7.777619260435076e-12,	2.6721763690673583e-11,	
  -3.0959165786548226e-11,	4.360670158298774e-11,	-1.5230691707834865e-11,	-5.230002195211192e-11,	
  -1.590610802792991e-11,	2.2392979831531434e-11,	-7.82629343276664e-12,	-2.687255279409939e-11,	
};

