#pragma once
#include "utils/robust.h"
#include "utils/output.h"
#include "utils/tensor2js.h"
#include <string>
using std::string;

/*void robust_result(cfg::HomoConfig config,Tensor<float> rhoold,int mode){
	enum ResultMode{Origin=0,Erode,Dilate};
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	rho.value().copy(rhoold);
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	float val=0;
	if(mode==Origin){
		float vol_ratio=robust_result_filter(rho.value(),config.filterRadius,0.5,16)/ne;
		printf("origin vol_ratio:   %.4e\033[0m   val:   %.4e\033[0m\n",vol_ratio,val);
        rho.value().toVdb(getPath("rho_origin.vdb"));
	}
	if(mode==Erode){
		float vol_ratio=robust_result_filter(rho.value(),config.filterRadius,0.7,16)/ne;
		printf("erode vol_ratio:   %.4e\033[0m\n",vol_ratio);
        rho.value().toVdb(getPath("rho_erode.vdb"));
	}

	if(mode==Dilate){
		float vol_ratio=robust_result_filter(rho.value(),config.filterRadius,0.3,16)/ne;		
		printf("dilate vol_ratio:   %.4e\033[0m		val:   %.4e\033[0m\n",vol_ratio,val);
        rho.value().toVdb(getPath("rho_dilate1.vdb"));
	}


		
}


void example_opti_bulk(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=1;
	int cycle=50;
	
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);

	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(1, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);

	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if(((iter%50)==0)&&(beta<16)){
			beta*=2;
			rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
		}

		elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
		AbortErr();
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 900.f; // bulk modulus


		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		
		// check convergence
		//if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		auto vol_ones=vol_diff.flatten();
		float* dgdx[1] = { vol_ones.data()};
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m   vol = %.4e\033[0m\n", iter, val,vol_ratio);
		// design variables
		auto rhoArray = rho.value().flatten();
		// mma update
		mma.update(iter, rhoArray.data(), objGrad.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho456.vdb"));
	//Ch.writeTo(getPath("C"));

	robust_result(config,rho.value(),0);
}

void robust_bulk(cfg::HomoConfig config) {
	
	// set output prefix
	setPathPrefix(config.outprefix);
	JSON_INIT;
	CONFIG;
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=4;
	int cycle=50;
	float voldlt=config.volRatio;
	
	// define material interpolation term
	auto rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	auto rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
	auto rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);

	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(4, ne, -1, -1, 1000, 1,1);
	mma.setBound(0.001, 1);

	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	ROBUST_TIME_INIT;
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if(((iter%50)==0)&&(beta<16)){
			beta*=2;
			rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
			rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
			rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);
		}
		if((iter%20==0)){
			float dltvol=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.2,beta)/ne;
			float orgvol=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
			voldlt=config.volRatio*dltvol/orgvol;
		}

		ROBUST_TIME1;


		elastic_tensor_t<float, decltype(rhoporg)> Ch(hom, rhoporg);
		AbortErr();
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 900.f; // bulk modulus
		AbortErr();
		float val = objective.eval();
		objlist.emplace_back(val);
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhoperd)> Ch1(hom, rhoperd);
		AbortErr();
		auto objective1 = -(Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2) +
		(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 9.f; // bulk modulus
		AbortErr();
		float val1 = objective1.eval();
		objlist.emplace_back(val);
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad1=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhopdlt)> Ch2(hom, rhopdlt);
		AbortErr();
		auto objective2 = -(Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2) +
		(Ch2(0, 1) + Ch2(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
		float val2 = objective2.eval();
		objlist.emplace_back(val);
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad2=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.2,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - voldlt) * vol_scale;
		gval.proxy<float>()[1] = val;
		gval.proxy<float>()[2] = val1;
		gval.proxy<float>()[3] = val2;

		auto vol_ones=vol_diff.flatten();
		float* dgdx[4] = { vol_ones.data(),objGrad.data(),objGrad1.data(),objGrad2.data()};
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m   vol = %.4e\033[0m\n", iter, val,vol_ratio);
		// design variables
		auto rhoArray = rho.value().flatten();

		ROBUST_TIME2;

		// mma update
		mma.update(iter, rhoArray.data(), zeros.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
		ROBUST_TIME3;
		orgv=val;
		erdv=val1;
		dltv=val2;
		JSON_OUTPUT;
	}
	//rhop.value().toMatlab("rhofinal");
	//hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	//hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho_raw.vdb"));
	//Ch.writeTo(getPath("C"));
	JSON_WRITE;
	JSON_ROBUST_RESULT;
}


void example_opti_npr(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=1;
	int cycle=50;
	
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);

	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(1, ne, 1, 0, 10000, 1);
	mma.setBound(0.001, 1);

	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if(((iter%50)==0)&&(beta<16)){
			beta*=2;
			rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
		}

		float beta1=0.6;

		elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
		AbortErr();
		auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * beta1 + 1).log() - (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f) * 1e-3f;


		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		
		// check convergence
		//if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		auto vol_ones=vol_diff.flatten();
		float* dgdx[1] = { vol_ones.data()};
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m   vol = %.4e\033[0m	pr = %.4e\033[0m\n", iter, val,vol_ratio,Ch.pr());
		// design variables
		auto rhoArray = rho.value().flatten();
		// mma update
		mma.update(iter, rhoArray.data(), objGrad.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho123.vdb"));
	//Ch.writeTo(getPath("C"));

	robust_result(config,rho.value(),0);
}

void robust_npr(cfg::HomoConfig config) {
	
	// set output prefix
	setPathPrefix(config.outprefix);
	JSON_INIT;
	CONFIG;
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=2;
	int cycle=50;
	float voldlt=config.volRatio;
	
	// define material interpolation term
	auto rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	auto rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
	auto rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);

	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(4, ne, -1, -1, 1000, 1,1);
	mma.setBound(0.001, 1);

	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	ROBUST_TIME_INIT;
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if(((iter%50)==0)&&(beta<16)){
			beta*=2;
			rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
			rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
			rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);
		}
		if((iter%20==0)){
			float dltvol=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.4,beta)/ne;
			float orgvol=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
			voldlt=config.volRatio*dltvol/orgvol;
		}

		ROBUST_TIME1;


		float beta1=0.6;

		elastic_tensor_t<float, decltype(rhoporg)> Ch(hom, rhoporg);
		AbortErr();
		auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * beta1 + 1).log() - (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f) * 1e-3f;
		AbortErr();
		float npr=Ch.pr();
		float val = objective.eval();
		objlist.emplace_back(val);
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhoperd)> Ch1(hom, rhoperd);
		AbortErr();
		auto objective1 = ((Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2))
			/ (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)) * beta1 + 1).log() - (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)).pow(0.5f) * 1e-3f;
		AbortErr();
		float npr1=Ch1.pr();
		float val1 = objective1.eval();
		objlist.emplace_back(val);
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad1=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhopdlt)> Ch2(hom, rhopdlt);
		AbortErr();
		auto objective2 = ((Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2))
			/ (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)) * beta1 + 1).log() - (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)).pow(0.5f) * 1e-3f;
		AbortErr();
		float npr2=Ch2.pr();
		float val2 = objective2.eval();
		objlist.emplace_back(val);
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad2=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.4,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - voldlt) * vol_scale;
		gval.proxy<float>()[1] = val;
		gval.proxy<float>()[2] = val1;
		gval.proxy<float>()[3] = val2;

		auto vol_ones=vol_diff.flatten();
		float* dgdx[4] = { vol_ones.data(),objGrad.data(),objGrad1.data(),objGrad2.data()};
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m   vol = %.4e\033[0m	pr = %.4e\033[0m	pr1 = %.4e\033[0m	pr2 = %.4e\033[0m\n", iter, val,vol_ratio,npr,npr1,npr2);
		// design variables
		auto rhoArray = rho.value().flatten();

		ROBUST_TIME2;

		// mma update
		mma.update(iter, rhoArray.data(), zeros.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
		ROBUST_TIME3;
		orgv=val;
		erdv=val1;
		dltv=val2;
		JSON_OUTPUT;
	}
	//rhop.value().toMatlab("rhofinal");
	//hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	//hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho_raw.vdb"));
	//Ch.writeTo(getPath("C"));
	JSON_WRITE;
	JSON_ROBUST_RESULT;
}

void robust_npr2(cfg::HomoConfig config) {
	
	// set output prefix
	setPathPrefix(config.outprefix);
	JSON_INIT;
	CONFIG;
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> ndf(config.reso[0], config.reso[1], config.reso[2]);
	ndf.reset(0);
	Tensor<float> df1(config.reso[0], config.reso[1], config.reso[2]);
	df1.reset(0);
	Tensor<float> df2(config.reso[0], config.reso[1], config.reso[2]);
	df2.reset(0);
	Tensor<float> df3(config.reso[0], config.reso[1], config.reso[2]);
	df3.reset(0);
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	//initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=1;
	int cycle=50;
	float voldlt=config.volRatio;
	
	// define material interpolation term
	auto rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	auto rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
	auto rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);

	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(1, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);

	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	ROBUST_TIME_INIT;
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if(((iter%50)==0)&&(beta<16)){
			beta*=2;
			rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
			rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
			rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);
		}
		if((iter%20==0)){
			float dltvol=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.3,beta)/ne;
			float orgvol=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
			voldlt=config.volRatio*dltvol/orgvol;
		}

		ROBUST_TIME1;


		float beta1=0.6;

		elastic_tensor_t<float, decltype(rhoporg)> Ch(hom, rhoporg);
		AbortErr();
		auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * beta1 + 1).log() - (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f) * 1e-3f;
		AbortErr();
		float val = objective.eval();
		objlist.emplace_back(val);
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		df1.copy(rho.diff());
		float pr=Ch.pr();

		elastic_tensor_t<float, decltype(rhoperd)> Ch1(hom, rhoperd);
		AbortErr();
		auto objective1 = ((Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2))
			/ (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)) * beta1 + 1).log() - (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)).pow(0.5f) * 1e-3f;
		AbortErr();
		float val1 = objective1.eval();
		objlist.emplace_back(val);
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		df2.copy(rho.diff());
		float pr1=Ch1.pr();

		elastic_tensor_t<float, decltype(rhopdlt)> Ch2(hom, rhopdlt);
		AbortErr();
		auto objective2 = ((Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2))
			/ (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)) * beta1 + 1).log() - (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)).pow(0.5f) * 1e-3f;
		AbortErr();
		float val2 = objective2.eval();
		objlist.emplace_back(val);
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		df3.copy(rho.diff());
		float pr2=Ch2.pr();

		mmadiff(ndf,df1,df2,df3,val,val1,val2,0.7);

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.3,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - voldlt) * vol_scale;

		auto vol_ones=vol_diff.flatten();
		float* dgdx[1] = { vol_ones.data()};
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m   vol = %.4e\033[0m\n   pr = %.4e\033[0m\n   pr1 = %.4e\033[0m\n	pr2 = %.4e\033[0m\n", iter, val,vol_ratio,pr,pr1,pr2);
		// design variables
		auto rhoArray = rho.value().flatten();
		auto df_flt=ndf.flatten();
		ROBUST_TIME2;

		// mma update
		mma.update(iter, rhoArray.data(), df_flt.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
		ROBUST_TIME3;
		orgv=val;
		erdv=val1;
		dltv=val2;
		JSON_OUTPUT;
	}
	//rhop.value().toMatlab("rhofinal");
	//hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	//hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho_raw.vdb"));
	//Ch.writeTo(getPath("C"));
	JSON_WRITE;
	JSON_ROBUST_RESULT;
}


// usage of MMA optimizer
void example_opti_shear_isotropy(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density

	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
		
	float beta=1;
	int cycle=50;
	
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	// create elastic tensor expression
	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(2, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter; iter++) {
		// define objective expression

		if(((iter%50)==0)&&(beta<16)){
			beta*=2;
			rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
		}
		auto Ch = genCH(hom, rhop);
		auto objective = -(Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 3.f; // shear modulus
		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		//if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		auto objGrad = rho.diff().flatten();
		float aniScale = 1000.f;
		auto constrain = ((Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) * 2.f /
			(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) - Ch(0, 1) - Ch(0, 2) - Ch(1, 2)) - 1.f).pow(2) * aniScale;
		float anistroy_constrain = constrain.eval();
		constrain.backward(1);
		float zener_ratio = sqrt(anistroy_constrain / aniScale) + 1;
		symmetrizeField(rho.diff(), config.sym);
		auto gGrad = rho.diff().flatten();
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;

		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		symmetrizeField(vol_diff, config.sym);

		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;
		gval.proxy<float>()[1] = anistroy_constrain * aniScale - 0.1f;
		// constrain derivative
		auto vol_ones = vol_diff.flatten();

		float* dgdx[2] = { vol_ones.data(), gGrad.data() };
		// design variables
		auto rhoArray = rho.value().flatten();
		printf("zener ratio = %4.2e ; obj = %4.2e ; vol = %4.2e\n", zener_ratio, val, vol_ratio);
		printf("constrain   = %4.2e ;       %4.2e\n", float(gval.proxy<float>()[0]), float(gval.proxy<float>()[1]));
		// mma update
		mma.update(iter, rhoArray.data(), objGrad.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	//Ch.writeTo(getPath("C"));

	robust_result(config,rho.value(),0);
}


// usage of MMA optimizer
void example_opti_shear_isotropy2(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(2, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter; iter++) {
		// define objective expression
		auto objective = -(Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 3.f; // shear modulus
		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		auto objGrad = rho.diff().flatten();
		float aniScale = 1000.f;
		auto constrain = ((Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) * 2.f /
			(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) - Ch(0, 1) - Ch(0, 2) - Ch(1, 2)) - 1.f).pow(2) * aniScale;
		float anistroy_constrain = constrain.eval();
		constrain.backward(1);
		float zener_ratio = sqrt(anistroy_constrain / aniScale) + 1;
		symmetrizeField(rho.diff(), config.sym);
		auto gGrad = rho.diff().flatten();
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = rho.value().sum() / ne;
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;
		gval.proxy<float>()[1] = anistroy_constrain * aniScale - 0.1f;
		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[2] = { vol_ones.data(), gGrad.data() };
		// design variables
		auto rhoArray = rho.value().flatten();
		printf("zener ratio = %4.2e ; obj = %4.2e ; vol = %4.2e\n", zener_ratio, val, vol_ratio);
		printf("constrain   = %4.2e ;       %4.2e\n", float(gval.proxy<float>()[0]), float(gval.proxy<float>()[1]));
		// mma update
		mma.update(iter, rhoArray.data(), objGrad.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));

}*/

void mytest(cfg::HomoConfig config){
	matlabjs matjs(string("mytest"));

	TensorVar<float> x(10,10,10);
	initDensity(x,config);
	auto y=x.org(16);
	y.eval();
	auto ones=Tensor<float>(x.getDim());
	ones.reset(1);
	y.backward(ones);
	matjs.tensor2js("y",y.value().view());
	matjs.tensor2js("x",x.value().view());
	matjs.tensor2js("xdiff",x.diff().view());
}

void runCustom(cfg::HomoConfig config) {
	SIG_SET
	//example_opti_bulk(config);
	//robust_bulk(config);
	//robust_npr(config);
	//robust_npr2(config);
	//example_opti_shear_isotropy(config);
	//example_opti_npr(config);
	//example_opti_shear_isotropy(config);
	//example_opti_shear_isotropy2(config);
	mytest(config);
#if 0
	Tensor<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	rho.fromVdb("./rho123.vdb",1);
	//robust_result(config,rho,0);
	//robust_result(config,rho,1);
	robust_result(config,rho,2);
#endif
}


