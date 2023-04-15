#pragma once
#include "utils/robust.h"
#include "utils/output.h"
#include <string>


void robust_result(cfg::HomoConfig config,int mode,const std::string &name){
	enum ResultMode{Origin=0,Erode,Dilate};
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	rho.value().fromVdb(getPath(name),1);

	float val=0;
	if(mode==Origin){
		rho.value().proj(16.f,0.5f,1.f,0.f);
		printf("origin vol_ratio:   %.4e\033[0m   val:   %.4e\033[0m\n",rho.value().sum()/ne,val);
        rho.value().toVdb(getPath("rho_origin.vdb"));
	}
	else if(mode==Erode){
		rho.value().proj(16.f,0.7f,1.f,0.f);
		printf("erode vol_ratio:   %.4e\033[0m\n",rho.value().sum()/ne);
        rho.value().toVdb(getPath("rho_erode.vdb"));
	}

	else{
		rho.value().proj(16.f,0.3f,1.f,0.f);
		printf("dilate vol_ratio:   %.4e\033[0m\n",rho.value().sum()/ne);
        rho.value().toVdb(getPath("rho_dilate.vdb"));
	}


		
}

/*void example_opti_bulk3(cfg::HomoConfig config) {
	JSON_INIT;
	CONFIG;
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);

	//#########################################################################
	//##########################################################################
	//can be replaced with const tensor
	//#######################################################################
	//########################################################
	Tensor<float> obj(config.reso[0], config.reso[1], config.reso[2]);
	obj.reset();
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// define material interpolation term
	float beta=1;
	float val=0,val1=0,val2=0;
	
	int cycle=50;

	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
	auto rhop1=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta);
	auto rhop2=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
	AbortErr();

	MMAOptimizer mma(4, ne, -1, -1, 1000, 1,1);
	mma.setBound(0.001, 1);

	float vol_ratio=config.volRatio;
	float volRatioDilate=vol_ratio;
	float volRatioOrigin=vol_ratio;

	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	ROBUST_TIME_INIT;
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if((iter%cycle==0)&&beta<16){
			beta*=2;
			rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
			rhop1=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta);
			rhop2=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
		}

		ROBUST_TIME1;
		

		auto Ch=genCH(hom, rhop);
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
		AbortErr();
		val = objective.eval();
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		auto Ch2=genCH(hom, rhop2);
		auto objective2 = -(Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2) +
		(Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2)) * 2) / 9.f; // bulk modulus
		AbortErr();
		val2 = objective2.eval();
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad2=rho.diff().flatten();

		auto Ch1=genCH(hom, rhop1);
		auto objective1 = -(Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2) +
		(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 9.f; // bulk modulus
		AbortErr();
		val1 = objective1.eval();
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad1=rho.diff().flatten();

		printf("\033[32m\n * Iter %d   origin = %.4e\033[0m    erode = %.4e\033[0m    dilate = %.4e\033[0m \n", iter, val,val1,val2);

		ROBUST_TIME2;


#if 1

		
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		
		if((iter%20)==0){
			volRatioOrigin=robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		}

		vol_ratio =robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.3,beta)/ne;
		
		if((iter%20)==0){
			volRatioDilate=vol_ratio*volRatioOrigin/config.volRatio;
		}	
		gval.proxy<float>()[0] = (vol_ratio - volRatioDilate) * vol_scale;
		gval.proxy<float>()[1] = val;
		gval.proxy<float>()[2] = val1;
		gval.proxy<float>()[3] = val2;

		// constrain derivative
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);
		float* dgdx[4] = { vol_diff.flatten().data(),objGrad.data(),objGrad1.data(),objGrad2.data()};
		// design variables
		auto rhoArray = rho.value().flatten();
		mma.update(iter, rhoArray.data(), obj.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
#endif


		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
		ROBUST_TIME3;
		float orgv=val;
		float erdv=val1;
		float dltv=val2;
		JSON_OUTPUT;
	}
	

	//rhop.value().toMatlab("rhofinal");
	//hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	//hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho.vdb"));
	//Ch.writeTo(getPath("C"));

	printf("\033[32m\n *  erode = %.4e\033[0m\n", val);
	robust_result(config,0,"rho.vdb");
	robust_result(config,1,"rho.vdb");
	robust_result(config,2,"rho.vdb");
	
}*/



void example_opti_bulk(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	//#########################################################################
	//##########################################################################
	//can be replaced with const tensor
	//#######################################################################
	//########################################################
	Tensor<float> obj(config.reso[0], config.reso[1], config.reso[2]);
	obj.reset();
	Tensor<float> vol_diff(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> xTilde_diff(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> dc(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> rhoArray(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// define material interpolation term
	float beta=1;
	int cycle=50;
#if 1
	auto rhop = rho;
	//auto rhop = rho.org(8);
#else
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
#endif
	// create elastic tensor expression
	//auto Ch = genCH(hom, rhop);
	//elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	AbortErr();
	// create a oc optimizer
#if 0
	OCOptimizer oc(0.001, config.designStep, config.dampRatio);
#else
	MMAOptimizer mma(2, ne, -1, -1, 1000, 1,1);
	mma.setBound(0.001, 1);
#endif
	// define objective expression
#if 1
	//auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
	//	(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
#else
	auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // shear modulus
#endif
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		if(((iter%cycle)==0)&&beta<=16){
			beta*=2;
		}

		// use rhoArray to save raw rho
		rhoArray.copy(rho.value());
		// now rho contains xOrigin
		robust_filter_proj(xTilde_diff,rho.value(),config.filterRadius,0.5,beta);

		elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);

		// check convergence
		if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);

		// dc(xOrigin)
		dc.copy(rho.diff());
		// dc(xOrigin).*projdiff(xTilde)
		dc.multiply(xTilde_diff);
		// (H*...)./Hs
		robust_filter(dc,rho.value(),config.filterRadius,1);
#if 0
		// filtering the sensitivity
		//oc.filterSens(rho.diff(), rho.value(), config.filterRadius);
		// update density
		oc.update(rho.diff(), rho.value(), config.volRatio,beta);
#else
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio =rho.value().sum()/ne;	

		vol_diff.copy(xTilde_diff);

		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);

		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m    vol_ratio = %.4e\033[0m\n", iter, val,vol_ratio);

		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;
		gval.proxy<float>()[1] = val;

		// constrain derivative
		float* dgdx[2] = { vol_diff.flatten().data(),dc.flatten().data()};
		// design variables
		mma.update(iter, rhoArray.data(), obj.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
#endif
		
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rhod.vdb"));
	robust_result(config,0,"rhod.vdb");
	//Ch.writeTo(getPath("C"));
}



void runCustom(cfg::HomoConfig config) {
	SIG_SET
	//example_opti_bulk3(config);
	example_opti_bulk(config);
}


