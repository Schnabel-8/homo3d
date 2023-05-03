#ifndef ROBUST_OUTPUT_H
#define ROBUST_OUTPUT_H
#include<csignal>
#include<fstream>
#include"nlohmann/json.hpp"
#include<sys/time.h>

//json

#define CONFIG configvec.push_back(string("funtion = ")+string(__FUNCTION__));\
			   configvec.push_back(string("resolution= ")+to_string(config.reso[0]))

#define JSON_INIT  using std::vector;\
					using std::string;\
					using std::to_string;\
					using nlohmann::json;\
					vector<float> vec_origin,vec_volfrac,vec_erode,vec_dilate,vec_time_eq,vec_time_mma;\
					json js;\
				    std::ofstream o(getPath("debug.json"));\
					float orgv,erdv,dltv;\
				    vector<string> configvec
					  
#define JSON_OUTPUT vec_origin.push_back(orgv);\
					 vec_erode.push_back(erdv);\
					 vec_dilate.push_back(dltv);\
					 vec_volfrac.push_back(vol_ratio);\
					 vec_time_eq.push_back(time_eq);\
					 vec_time_mma.push_back(time_mma)
					 
#define JSON_WRITE	\
					 js["origin"]=vec_origin;\
					 js["volfrac"]=vec_volfrac;\
					 js["erode"]=vec_erode;\
					 js["dilate"]=vec_dilate;\
					 js["time_eq"]=vec_time_eq;\
					 js["time_mma"]=vec_time_mma;\
					 js["config"]=configvec

					// o.seekp(0,std::ios::beg);\
					// o<<std::setw(4)<<js<<std::endl

#define JSON_ROBUST_RESULT	{Tensor<float> rhoorg(config.reso[0], config.reso[1], config.reso[2]);\
							rhoorg.copy(rho.value());\
							float vol_ratio=robust_result_filter(rhoorg,config.filterRadius,0.5,16)/ne;\
							js["org_vol_ratio"]=vol_ratio;\
      						rhoorg.toVdb(getPath("rho_origin.vdb"));}\
							\
							{Tensor<float> rhoerd(config.reso[0], config.reso[1], config.reso[2]);\
							rhoerd.copy(rho.value());\
							float vol_ratio=robust_result_filter(rhoerd,config.filterRadius,0.7,16)/ne;\
							js["erd_vol_ratio"]=vol_ratio;\
      						rhoerd.toVdb(getPath("rho_erode.vdb"));}\
							\
							{Tensor<float> rhodlt(config.reso[0], config.reso[1], config.reso[2]);\
							rhodlt.copy(rho.value());\
							float vol_ratio=robust_result_filter(rhodlt,config.filterRadius,0.4,16)/ne;\
							js["dlt_vol_ratio"]=vol_ratio;\
      						rhodlt.toVdb(getPath("rho_dilate.vdb"));}\
							o.seekp(0,std::ios::beg);\
							o<<std::setw(4)<<js<<std::endl


// sig_handler: type ctrl-\ to quit the running process and save current results
// remember to involve quit_flag in the loop condition
// for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {...}
#define SIG_HANDLER int quit_flag=0;\
					void sig_handler(int signo){\
						quit_flag=1;\
					}\
		
#define SIG_SET signal(SIGQUIT,sig_handler);

SIG_HANDLER

// time: statistics the running time of solving equations and solving optimization problem

#define ROBUST_TIME_INIT \
struct timeval t1,t2,t3;\
double time_eq,time_mma

#define ROBUST_TIME1 \
gettimeofday(&t1,NULL)

#define ROBUST_TIME2 \
gettimeofday(&t2,NULL);\
time_eq = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0

#define ROBUST_TIME3 \
gettimeofday(&t3,NULL);\
time_mma = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0
#endif