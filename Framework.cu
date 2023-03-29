#pragma once
#include "utils/robust.h"

void erode_bulk(cfg::HomoConfig config) {
	ROBUST_BULK(0.5,30,\
	-(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f,\
	-(Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2) +(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 9.f,\
	(rho.pow(2)),\
	(rho.pow(2)).erd(beta)\
	)
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
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// define material interpolation term
	float beta=0.5;
#if 1
	auto rhop = rho.pow(2).erd(beta);
#else
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(2).erd(beta);
#endif
	// create elastic tensor expression
	//auto Ch = genCH(hom, rhop);
	//elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	AbortErr();
	// create a oc optimizer
	OCOptimizer oc(0.001, config.designStep, config.dampRatio);
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
	for (int iter = 0; iter < config.max_iter; iter++) {
		if((iter%30==0)&&beta<=16){
			beta*=2;
			rhop=  rho.pow(2).erd(beta);
		}
		auto Ch=genCH(hom, rhop);
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
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
#if 1
		// filtering the sensitivity
		oc.filterSens(rho.diff(), rho.value(), config.filterRadius);
#endif
		//rho.diff().toMatlab("senscustom");
		// update density
		oc.update(rho.diff(), rho.value(), config.volRatio);
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	//Ch.writeTo(getPath("C"));
}

#if 0
void example_opti_npr(cfg::HomoConfig config) {
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
#if 1
	auto rhop = rho.pow(3);
#else
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
#endif
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
	AbortErr();
	// create a oc optimizer
	OCOptimizer oc(0.001, config.designStep, config.dampRatio);
	// define objective expression
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter; iter++) {
		// abort when cuda error occurs
		AbortErr();
		float beta = 0.8f;
		auto objective = Ch(0, 1) + Ch(0, 2) + Ch(1, 2) -
			(Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * powf(beta, iter);
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
#if 1
		// filtering the sensitivity
		oc.filterSens(rho.diff(), rho.value(), config.filterRadius);
#endif
		// update density
		oc.update(rho.diff(), rho.value(), config.volRatio);
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
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

}

void mma_bulk(cfg::HomoConfig config) {
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
	
	auto rhop=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);

	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(1, ne, 1, 0, 1000, 1,1);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		// define objective expression
		auto objective =-(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus

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

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = rho.value().sum() / ne;
		printf("vol: %f\n", vol_ratio);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[1] = { vol_ones.data()};//,objGrad2.data()};
		// design variables
		auto rhoArray = rho.value().flatten();
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

	freeMem();
}


void robust_bulk(cfg::HomoConfig config) {
	JSON_INIT;
	CONFIG;
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
	double beta=1;
    int cycle=30;
	bool erode_flag=true;
	bool origin_flag=true;
	//Erode
	auto rhop1 = (rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(2)).erd(beta);
	auto rhop=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(2);
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
/*auto Ch1=genCH(hom,rhop1);
		auto objective1 = -(Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2) +
		(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 9.f; // bulk modulus*/
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(1, ne, 1, 0, 1000, 1,1);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	struct timeval t1,t2,t3;
    double time_eq,time_mma;
	float val=0,val1=0;
	homo::Tensor<float> objGrad;
	homo::Tensor<float> objGrad1;
	float *dfdx=nullptr;
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		gettimeofday(&t1,NULL);
		if((iter%cycle)==0&&beta<=16){
			beta*=2;
			rhop1=(rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(2)).erd(beta);
		}
		auto objective =-(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
		if(origin_flag){
		// define objective expression
		
		// abort when cuda error occurs
		AbortErr();
		val = objective.eval();
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
		objGrad = rho.diff().flatten();
		}

		auto Ch1=genCH(hom,rhop1);
		auto objective1 = -(Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2) +
		(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 9.f; // bulk modulus
		
		if(erode_flag){    
		
 		val1 = objective1.eval();
        // compute derivative
		objective1.backward(1);
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		objGrad1 = rho.diff().flatten();
		}
		

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = rho.value().sum() / ne;
		printf("vol: %f\n", vol_ratio);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[1] = { vol_ones.data()};
		// design variables
		auto rhoArray = rho.value().flatten();

		gettimeofday(&t2,NULL);
		time_eq = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

		if(val>=val1){
			dfdx=objGrad.data();
			erode_flag=false;
		}
		else{
			dfdx=objGrad1.data();
			origin_flag=false;
		}

		if((iter%3)==0){
			erode_flag=true;
			origin_flag=true;
		}
		//dfdx=objGrad.data();
		// mma update
		mma.update(iter, rhoArray.data(), dfdx, gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		// output temp results
		logIter(iter, config, rho, Ch, val);

		gettimeofday(&t3,NULL);
		time_mma = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0;

		orgv=val;
		erdv=val1;
		JSON_OUTPUT;
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	//Ch.writeTo(getPath("C"));

	freeMem();
}



// usage of MMA optimizer
void robust_shear(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);

	Tensor<float> obj(config.reso[0],config.reso[1],config.reso[2]);
	obj.reset();
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	float beta=-10.f;
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	//Erode
	auto rhop1 = ((rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3)*(-beta)+beta).exp()).conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	//Dilate
	auto rhop2 = ((((rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3))*beta).exp()+1.f).pow(-1)*2.f-1.f).conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
	auto Ch1 = genCH(hom, rhop1);
	auto Ch2 = genCH(hom, rhop2);
	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(5, ne, -1, -1, 1000, 1,2);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		// define objective expression
		auto objective = -(Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 3.f; // shear modulus
		auto objective1 = -(Ch1(3, 3) + Ch1(4, 4) + Ch1(5, 5)) / 3.f; // shear modulus
		auto objective2 = -(Ch2(3, 3) + Ch2(4, 4) + Ch2(5, 5)) / 3.f; // shear modulus
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

		float val1 = objective1.eval();
        // compute derivative
		objective1.backward(1);
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		auto objGrad1 = rho.diff().flatten();

		float val2 = objective2.eval();
        // compute derivative
		objective2.backward(1);
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		auto objGrad2 = rho.diff().flatten();

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
		gval.proxy<float>()[2] = val;
		gval.proxy<float>()[3] = val1;
		gval.proxy<float>()[4] = val2;
		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[5] = { vol_ones.data(), gGrad.data() ,objGrad.data(),objGrad1.data(),objGrad2.data()};
		// design variables
		auto rhoArray = rho.value().flatten();
		printf("zener ratio = %4.2e ; obj = %4.2e ; vol = %4.2e\n", zener_ratio, val, vol_ratio);
		printf("constrain   = %4.2e ;       %4.2e\n", float(gval.proxy<float>()[0]), float(gval.proxy<float>()[1]));
		// mma update
		mma.update(iter, rhoArray.data(), obj.data(), gval.data<float>(), dgdx);
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

}

void robust_npr(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);

	Tensor<float> obj(config.reso[0],config.reso[1],config.reso[2]);
	obj.reset();
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	float beta=-10.f;
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	//Erode
	auto rhop1 = (rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3)*(-beta)+beta).exp();
	//Dilate
	//auto rhop2 = (((rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3))*beta).exp()+1.f).pow(-1)*2.f-1.f;
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
	auto Ch1 = genCH(hom, rhop1);
	//auto Ch2 = genCH(hom, rhop2);
	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(4, ne, -1, -1, 1000, 1,2);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		// define objective expression
		auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * 0.6f + 1).log();
		auto objective1 = ((Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2))
			/ (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)) * 0.6f + 1).log();
		//auto objective2 = ((Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2))
		//	/ (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)) * 0.6f + 1).log();
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

		float val1 = objective1.eval();
        // compute derivative
		objective1.backward(1);
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		auto objGrad1 = rho.diff().flatten();
#if 0
		float val2 = objective2.eval();
        // compute derivative
		objective2.backward(1);
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// objective derivative
		auto objGrad2 = rho.diff().flatten();
#endif

		auto constrain = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * 1e-4;
		float constrain_val = constrain.eval()+4;
		constrain.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto gGrad = rho.diff().flatten();
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = rho.value().sum() / ne;
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;
		gval.proxy<float>()[1] = constrain_val;
		gval.proxy<float>()[2] = val;
		gval.proxy<float>()[3] = val1;
//		gval.proxy<float>()[4] = val2;
		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[5] = { vol_ones.data(), gGrad.data() ,objGrad.data(),objGrad1.data()};//,objGrad2.data()};
		// design variables
		auto rhoArray = rho.value().flatten();
		printf("constrain   = %4.2e ;       %4.2e      vol   =:       %4.2e\n", float(gval.proxy<float>()[0]), float(gval.proxy<float>()[1]),vol_ratio);
		// mma update
		mma.update(iter, rhoArray.data(), obj.data(), gval.data<float>(), dgdx);
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
}

void example2(cfg::HomoConfig config) {
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
	float beta=-10.f;
	// define material interpolation term
	//auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	//Erode
	//auto rhop = (rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3)*(-beta)+beta).exp();
	//Dilate
	auto rhop = (((rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3))*beta).exp()+1.f).pow(-1)*2.f-1.f;
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(1, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		// define objective expression
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
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
		
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = rho.value().sum() / ne;
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;
		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[1] = { vol_ones.data() };
		// design variables
		auto rhoArray = rho.value().flatten();
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

}

#endif


void runCustom(cfg::HomoConfig config) {
	SIG_SET
	example_opti_bulk(config);
	//example_opti_npr(config);
	//example_opti_shear_isotropy(config);
	//robust_bulk(config);
	//robust_npr(config);
	//robust_shear(config);
	//example2(config);
	//mma_bulk(config);
	//erode_bulk(config);
}


