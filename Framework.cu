#pragma once
#include "utils/robust.h"
#include "utils/output.h"
#include "utils/tensor2js.h"
#include <string>
using std::string;
#include "utils/example.cu"
#include "utils/robust_example.cu"

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
	//example_opti_npr(config);
	//example_opti_shear(config);
	robust_bulk(config);
	//robust_npr(config);
	//robust_npr2(config);
	//example_opti_shear_isotropy(config);
	//example_opti_npr(config);
	//example_opti_shear_isotropy(config);
	//example_opti_shear_isotropy2(config);
	//mytest(config);
#if 0
	Tensor<float> rho(config.reso[0], config.reso[1], config.reso[2]);
	rho.fromVdb("./rho123.vdb",1);
	//robust_result(config,rho,0);
	//robust_result(config,rho,1);
	robust_result(config,rho,2);
#endif
}


