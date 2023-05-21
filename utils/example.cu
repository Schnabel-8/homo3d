

void example_opti_bulk(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// define density expression
	float vol_scale = 1000.f;
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> ones(config.reso[0], config.reso[1], config.reso[2]);
	ones.reset(vol_scale/ne);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=1;
	int cycle=50;
	
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);

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
		/*float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);*/

		auto proj_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
		proj_rho.eval();
		float vol_ratio=proj_rho.value().sum()/ne;
		proj_rho.backward(ones);
		auto vol_diff=rho.diff().flatten();
		//symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		float* dgdx[1] = { vol_diff.data()};
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

void example_opti_npr(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// define density expression
	float vol_scale = 1000.f;
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> ones(config.reso[0], config.reso[1], config.reso[2]);
	ones.reset(vol_scale/ne);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=1;
	int cycle=50;
	
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);

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
		float beta1=0.6;
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
		/*float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);*/

		auto proj_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
		proj_rho.eval();
		float vol_ratio=proj_rho.value().sum()/ne;
		proj_rho.backward(ones);
		auto vol_diff=rho.diff().flatten();
		//symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		float* dgdx[1] = { vol_diff.data()};
		float pr=Ch.pr();
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m    pr = %.4e\033[0m  vol = %.4e\033[0m\n", iter, val,pr,vol_ratio);
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


void example_opti_shear(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// define density expression
	float vol_scale = 1000.f;
	TensorVar<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	Tensor<float> zeros(config.reso[0], config.reso[1], config.reso[2]);
	zeros.reset(0);
	Tensor<float> ones(config.reso[0], config.reso[1], config.reso[2]);
	ones.reset(vol_scale/ne);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	
	float beta=1;
	int cycle=50;
	
	// define material interpolation term
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);

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
		auto objective = -(Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 3.; // Shear modulus


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
		/*float vol_scale = 1000.f;
		float vol_ratio = robust_filter_proj(vol_diff,rho.value(),config.filterRadius,0.5,beta)/ne;
		robust_filter(vol_diff,rho.value(),config.filterRadius,vol_scale/ne);*/

		auto proj_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
		proj_rho.eval();
		float vol_ratio=proj_rho.value().sum()/ne;
		proj_rho.backward(ones);
		auto vol_diff=rho.diff().flatten();
		//symmetrizeField(vol_diff, config.sym);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;

		float* dgdx[1] = { vol_diff.data()};
		float pr=Ch.pr();
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m    pr = %.4e\033[0m  vol = %.4e\033[0m\n", iter, val,pr,vol_ratio);
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