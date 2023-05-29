void robust_bulk(cfg::HomoConfig config) {
	
	// set output prefix
	setPathPrefix(config.outprefix);
	JSON_INIT;
	CONFIG;
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
	float voldlt=config.volRatio;
	
	// define material interpolation term
	auto rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	auto rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
	auto rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);

	
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
            auto dlt_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
            dlt_rho.eval();
			float dltvol=dlt_rho.value().sum()/ne;
			auto org_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
            org_rho.eval();
			float orgvol=org_rho.value().sum()/ne;
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
		(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 900.f; // bulk modulus
		AbortErr();
		float val1 = objective1.eval();
		objlist.emplace_back(val);
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad1=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhopdlt)> Ch2(hom, rhopdlt);
		AbortErr();
		auto objective2 = -(Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2) +
		(Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2)) * 2) / 900.f; // bulk modulus
		float val2 = objective2.eval();
		objlist.emplace_back(val);
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad2=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();

		auto proj_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
		proj_rho.eval();
		float vol_ratio=proj_rho.value().sum()/ne;
		proj_rho.backward(ones);
		auto vol_diff=rho.diff().flatten();

		gval.proxy<float>()[0] = (vol_ratio - voldlt) * vol_scale;
		gval.proxy<float>()[1] = val;
		gval.proxy<float>()[2] = val1;
		gval.proxy<float>()[3] = val2;

		float* dgdx[4] = { vol_diff.data(),objGrad.data(),objGrad1.data(),objGrad2.data()};
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



void robust_shear(cfg::HomoConfig config) {
	
	// set output prefix
	setPathPrefix(config.outprefix);
	JSON_INIT;
	CONFIG;
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
	float voldlt=config.volRatio;
	
	// define material interpolation term
	auto rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	auto rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
	auto rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);

	
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
            auto dlt_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
            dlt_rho.eval();
			float dltvol=dlt_rho.value().sum()/ne;
			auto org_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
            org_rho.eval();
			float orgvol=org_rho.value().sum()/ne;
			voldlt=config.volRatio*dltvol/orgvol;
		}

		ROBUST_TIME1;


		elastic_tensor_t<float, decltype(rhoporg)> Ch(hom, rhoporg);
		AbortErr();
		auto objective = -(Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 300.; // Shear modulus
		AbortErr();
		float val = objective.eval();
		objlist.emplace_back(val);
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhoperd)> Ch1(hom, rhoperd);
		AbortErr();
		auto objective1 = -(Ch1(3, 3) + Ch1(4, 4) + Ch1(5, 5)) / 300.; // Shear modulus
		AbortErr();
		float val1 = objective1.eval();
		objlist.emplace_back(val);
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad1=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhopdlt)> Ch2(hom, rhopdlt);
		AbortErr();
		auto objective2 = -(Ch2(3, 3) + Ch2(4, 4) + Ch2(5, 5)) / 300.; // Shear modulus
		float val2 = objective2.eval();
		objlist.emplace_back(val);
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad2=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();

		auto proj_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
		proj_rho.eval();
		float vol_ratio=proj_rho.value().sum()/ne;
		proj_rho.backward(ones);
		auto vol_diff=rho.diff().flatten();

		gval.proxy<float>()[0] = (vol_ratio - voldlt) * vol_scale;
		gval.proxy<float>()[1] = val;
		gval.proxy<float>()[2] = val1;
		gval.proxy<float>()[3] = val2;

		float* dgdx[4] = { vol_diff.data(),objGrad.data(),objGrad1.data(),objGrad2.data()};
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




void robust_npr(cfg::HomoConfig config) {
	
	// set output prefix
	setPathPrefix(config.outprefix);
	JSON_INIT;
	CONFIG;
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
	float voldlt=config.volRatio;
	
	// define material interpolation term
	auto rhoporg = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta).pow(3);
	auto rhoperd = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).erd(beta).pow(3);
	auto rhopdlt = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta).pow(3);

	
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
            auto dlt_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
            dlt_rho.eval();
			float dltvol=dlt_rho.value().sum()/ne;
			auto org_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).org(beta);
            org_rho.eval();
			float orgvol=org_rho.value().sum()/ne;
			voldlt=config.volRatio*dltvol/orgvol;
		}

		ROBUST_TIME1;


		elastic_tensor_t<float, decltype(rhoporg)> Ch(hom, rhoporg);
		AbortErr();
		float beta1=0.6;
		auto objective = 1000*(((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * beta1 + 1).log() - (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f) * 1e-3f);
		AbortErr();
		float val = objective.eval();
		objlist.emplace_back(val);
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhoperd)> Ch1(hom, rhoperd);
		AbortErr();
		auto objective1 =  1000*(((Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2))
			/ (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)) * beta1 + 1).log() - (Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2)).pow(0.5f) * 1e-3f);
		AbortErr();
		float val1 = objective1.eval();
		objlist.emplace_back(val);
		objective1.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad1=rho.diff().flatten();

		elastic_tensor_t<float, decltype(rhopdlt)> Ch2(hom, rhopdlt);
		AbortErr();
		auto objective2 =  1000*(((Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2))
			/ (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)) * beta1 + 1).log() - (Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2)).pow(0.5f) * 1e-3f);
		float val2 = objective2.eval();
		objlist.emplace_back(val);
		objective2.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objGrad2=rho.diff().flatten();

		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();

		auto proj_rho=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).dlt(beta);
		proj_rho.eval();
		float vol_ratio=proj_rho.value().sum()/ne;
		proj_rho.backward(ones);
		auto vol_diff=rho.diff().flatten();

		gval.proxy<float>()[0] = (vol_ratio - voldlt) * vol_scale;
		gval.proxy<float>()[1] = val;
		gval.proxy<float>()[2] = val1;
		gval.proxy<float>()[3] = val2;

		float* dgdx[4] = { vol_diff.data(),objGrad.data(),objGrad1.data(),objGrad2.data()};
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m   vol = %.4e\033[0m	pr = %.4e\033[0m	pr1= = %.4e\033[0m	pr2= = %.4e\033[0m\n", iter, val,vol_ratio,Ch.pr(),Ch1.pr(),Ch2.pr());
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
