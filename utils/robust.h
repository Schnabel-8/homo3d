#include "homogenization/Framework.cuh"
#include "utils/tools.h"
#include "utils/output.h"
using namespace homo;
using namespace culib;

enum json_mode{json_off=0,json_on};

#define ROBUST_INIT     setPathPrefix(config.outprefix);\
	                    Homogenization hom(config);\
	                    for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];\
	                    TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);\
	                    initDensity(rho, config);\
	                    rho.value().toVdb(getPath("initRho"));\
                        int ne = config.reso[0] * config.reso[1] * config.reso[2]

#define ROBUST_OC_INIT     AbortErr();\
	                      OCOptimizer oc(0.001, config.designStep, config.dampRatio)

#define ROBUST_OUT      hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);\
	                    hom.grid->array2matlab("objlist", objlist.data(), objlist.size());\
	                    rho.value().toVdb(getPath("rho"))

#define ROBUST_BULK_LOOP(beta_value,cycle_value,origin_obj,erode_obj,origin_filter,erode_filter)    float beta=beta_value;\
int cycle=cycle_value;\
bool erode_flag=true;\
bool origin_flag=true;\
\
auto rhop = origin_filter;\
auto Ch = genCH(hom, rhop);\
auto objective = origin_obj;\
auto rhop1=erode_filter;\
\
std::vector<double> objlist;\
ConvergeChecker criteria(config.finthres);\
struct timeval t1,t2,t3;\
double time_eq,time_mma;\
float val=0,val1=0;\
homo::Tensor<float> objGrad;\
homo::Tensor<float> objGrad1;\
homo::Tensor<float> dfdx;\
\
for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {\
    gettimeofday(&t1,NULL);\
	if((iter%cycle==0)&&beta<=16){\
		beta*=2;\
		rhop1= erode_filter;\
		}\
        \
		if((iter%3)==0){\
			erode_flag=true;\
			origin_flag=true;\
		}\
		\
		erode_flag=true;\
        if(origin_flag){\
		AbortErr();\
		val = objective.eval();\
		objlist.emplace_back(val);\
		objective.backward(1);\
		symmetrizeField(rho.diff(), config.sym);\
        objGrad = rho.diff();\
        }\
        \
        auto Ch1=genCH(hom,rhop1);\
		auto objective1 = erode_obj;\
        \
        if(erode_flag){\
            val1 = objective1.eval();\
		    objective1.backward(1);\
		    symmetrizeField(rho.diff(), config.sym);\
		    objGrad1 = rho.diff();\
        }\
        \
        if(val>=val1){\
			dfdx=objGrad;\
			erode_flag=false;\
		}\
		else{\
			dfdx=objGrad1;\
			origin_flag=false;\
		}\
        \
		dfdx=objGrad1;\
        gettimeofday(&t2,NULL);\
        time_eq = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;\
        float vol_ratio = rho.value().sum() / ne;\
		oc.filterSens(dfdx, rho.value(), float(config.filterRadius));\
		oc.update(dfdx, rho.value(), config.volRatio);\
		symmetrizeField(rho.value(), config.sym);\
        \
		logIter(iter, config, rho, Ch, val);\
        \
        gettimeofday(&t3,NULL);\
		time_mma = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0;\
        \
		orgv=val;\
		erdv=val1;\
		printf("\033[32m\n * Iter %d   origin = %.4e\033[0m    erode = %.4e\033[0m\n", iter, orgv, erdv);\
		JSON_OUTPUT\
	}

#define ROBUST_BULK(beta_value,cycle_value,origin_obj,erode_obj,origin_filter,erode_filter)   JSON_INIT;\
CONFIG;\
ROBUST_INIT;\
ROBUST_OC_INIT;\
ROBUST_BULK_LOOP(beta_value,cycle_value,origin_obj,erode_obj,origin_filter,erode_filter);\
ROBUST_OUT;
                                                            

