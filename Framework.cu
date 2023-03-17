#include "homogenization/Framework.cuh"
#include<csignal>
#include<fstream>
#include"nlohmann/json.hpp"
#include<sys/time.h>
using namespace homo;
using namespace culib;
using nlohmann::json;

#define CONFIG config.push_back(string("funtion = ")+string(__FUNCTION__));\
			   config.push_back(string("resolution= ")+to_string(config.reso[0]));

# define JSON_INIT  using std::vector;\
					using std::string;\
					using std::to_string;\
					vector<float> vec_obj,vec_volfrac,vec_constrain1,vec_constrain2,vec_time_eq,vec_time_mma;\
					json js;\
				    vector<string> config;

# define JSON_EMPLACE vec_obj.push_back(val);\
					  vec_constrain1.push_back(val1);\
					  vec_volfrac.push_back(vol_ratio);\
					  vec_time_eq.push_back(time_eq);\
					  vec_time_mma.push_back(time_mma);
					  
# define JSON_OUTPUT js["obj"]=vec_obj;\
					 js["volfrac"]=vec_volfrac;\
					 js["constrain1"]=vec_constrain1;\
					 js["constrain2"]=vec_constrain2;\
					 js["time_eq"]=vec_time_eq;\
					 js["time_mma"]=vec_time_mma;\
					 std::ofstream o("debug.json");\
					 o<<std::setw(4)<<js<<std::endl;
					 

int quit_flag=0;

template<typename CH>
void logIter(int iter, cfg::HomoConfig config, TensorVar<>& rho, CH& Ch, double obj) {
	/// fixed log 
	if (iter % 5 == 0) {
		rho.value().toVdb(getPath("rho"));
		//rho.diff().toVdb(getPath("sens"));
		Ch.writeTo(getPath("C"));
	}
	Ch.domain_.logger() << "finished iteration " << iter << std::endl;

	/// optional log
	char namebuf[100];
	if (config.logrho != 0 && iter % config.logrho == 0) {
		sprintf_s(namebuf, "rho_%04d", iter);
		rho.value().toVdb(getPath(namebuf));
	}

	if (config.logc != 0 && iter % config.logc == 0) {
		sprintf_s(namebuf, "Clog");
		//Ch.writeTo(getPath(namebuf));
		auto ch = Ch.data();
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		} else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		for (int i = 0; i < 36; i++) { ofs << ch[i] << " "; }
		ofs << std::endl;
		ofs.close();
	}

	if (config.logsens != 0 && iter % config.logsens == 0) {
		sprintf_s(namebuf, "sens_%04d", iter);
		//rho.diff().graft(sens.data());
		rho.diff().toVdb(getPath(namebuf));
	}

	if (config.logobj != 0 && iter % config.logobj == 0) {
		sprintf_s(namebuf, "objlog");
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		}
		else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		ofs << "obj = " << obj << std::endl;
		ofs.close();
	}
}

void initDensity(var_tsexp_t<>& rho, cfg::HomoConfig config) {
	int resox = rho.value().length(0);
	int resoy = rho.value().length(1);
	int resoz = rho.value().length(2);
	constexpr float pi = 3.1415926;
	if (config.winit == cfg::InitWay::random || config.winit == cfg::InitWay::randcenter) {
		randTri(rho.value(), config);
	} else if (config.winit == cfg::InitWay::manual) {
		rho.value().fromVdb(config.inputrho, false);
	} else if (config.winit == cfg::InitWay::interp) {
		rho.value().fromVdb(config.inputrho, true);
	} else if (config.winit == cfg::InitWay::rep_randcenter) {
		randTri(rho.value(), config);
	} else if (config.winit == cfg::InitWay::noise) {
		rho.value().rand(0.f, 1.f);
		symmetrizeField(rho.value(), config.sym);
		rho.value().proj(20.f, 0.5f);
		auto view = rho.value().view();
		auto ker = [=] __device__(int id) { return  view(id); };
		float s = config.volRatio / (sequence_sum(ker, view.size(), 0.f) / view.size());
		rho.value().mapInplace([=] __device__(int x, int y, int z, float val) {
			float newval = val * s;
			if (newval < 0.001f) newval = 0.001;
			if (newval >= 1.f) newval = 1.f;
			return newval;
		});
	} else if (config.winit == cfg::InitWay::P) {
		rho.rvalue().setValue([=]__device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy , float(k) / resoz };
			float val = cosf(2 * pi * p[0]) + cosf(2 * pi * p[1]) + cosf(2 * pi * p[2]);
			auto newval = tanproj(-val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::G) {
		rho.rvalue().setValue([=]__device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy, float(k) / resoz };
			float s[3], c[3];
			for (int i = 0; i < 3; i++) {
				s[i] = sin(2 * pi * p[i]);
				c[i] = cos(2 * pi * p[i]);
			}
			float val = s[0] * c[1] + s[2] * c[0] + s[1] * c[2];
			auto newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::D) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy, float(k) / resoz };
			float x = p[0], y = p[1], z = p[2];
			float val = cos(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z) - sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z);
			float newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::IWP) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = { float(i) / resox, float(j) / resoy, float(k) / resoz };
			float x = p[0], y = p[1], z = p[2];
			float val = 2 * (cos(2 * pi * x) * cos(2 * pi * y) + cos(2 * pi * y) * cos(2 * pi * z) + cos(2 * pi * z) * cos(2 * pi * x)) -
				(cos(2 * 2 * pi * x) + cos(2 * 2 * pi * y) + cos(2 * 2 * pi * z));
			float newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	}

	symmetrizeField(rho.value(), config.sym);
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
#if 1
	auto rhop = rho.pow(3);
#else
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
#endif
	// create elastic tensor expression
	//auto Ch = genCH(hom, rhop);
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	AbortErr();
	// create a oc optimizer
	OCOptimizer oc(0.001, config.designStep, config.dampRatio);
	// define objective expression
#if 1
	auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
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
	Ch.writeTo(getPath("C"));
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
	JSON_INIT
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
	// define material interpolation term
	float beta=-10.f;
	//Erode
	auto rhop1 = (rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3)*(-beta)+beta).exp();
	//Dilate
	//auto rhop2 = (((rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3))*beta).exp()+1.f).pow(-1)*2.f-1.f;
	auto rhop=rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	// create elastic tensor expression
	auto Ch = genCH(hom, rhop);
	auto Ch1 = genCH(hom, rhop1);
	//auto Ch2 = genCH(hom, rhop2);
	AbortErr();
	// create a oc optimizer
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	MMAOptimizer mma(3, ne, -1, -1, 1000, 1,1);
	mma.setBound(0.001, 1);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	struct timeval t1,t2,t3;
    double time_eq,time_mma;
	for (int iter = 0; iter < config.max_iter&&!quit_flag; iter++) {
		gettimeofday(&t1,NULL);
		// define objective expression
		auto objective =-(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
		auto objective1 = -(Ch1(0, 0) + Ch1(1, 1) + Ch1(2, 2) +
		(Ch1(0, 1) + Ch1(0, 2) + Ch1(1, 2)) * 2) / 9.f; // bulk modulus
		//auto objective2 = -(Ch2(0, 0) + Ch2(1, 1) + Ch2(2, 2) +
		//(Ch2(0, 1) + Ch2(0, 2) + Ch2(1, 2)) * 2) / 9.f; // bulk modulus
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
		// constrain value
		auto gval = getTempPool().getUnifiedBlock<float>();
		float vol_scale = 1000.f;
		float vol_ratio = rho.value().sum() / ne;
		printf("vol: %f\n", vol_ratio);
		gval.proxy<float>()[0] = (vol_ratio - config.volRatio) * vol_scale;
		gval.proxy<float>()[1] = val; 
		gval.proxy<float>()[2] = val1;
		//gval.proxy<float>()[3] = val2;
		// constrain derivative
		auto vol_ones = rho.diff().flatten();
		vol_ones.reset(vol_scale / ne);
		float* dgdx[3] = { vol_ones.data(),objGrad.data(),objGrad1.data()};//,objGrad2.data()};
		// design variables
		auto rhoArray = rho.value().flatten();

		gettimeofday(&t2,NULL);
		time_eq = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
		// mma update
		mma.update(iter, rhoArray.data(), obj.data(), gval.data<float>(), dgdx);
		//update variable
		rho.value().graft(rhoArray.data());
		// output temp results
		logIter(iter, config, rho, Ch, val);

		gettimeofday(&t3,NULL);
		time_mma = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0;

		JSON_EMPLACE
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));

	JSON_OUTPUT
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

void sig_handler(int signo){
	quit_flag=1;
}

void runCustom(cfg::HomoConfig config) {
	signal(SIGQUIT,sig_handler);
	//example_opti_bulk(config);
	//example_opti_npr(config);
	//example_opti_shear_isotropy(config);
	robust_bulk(config);
	//robust_npr(config);
	//robust_shear(config);
	//example2(config);
	//mma_bulk(config);
}


