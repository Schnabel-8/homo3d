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
	auto y=x.erd(2);
	y.eval();
	auto ones=Tensor<float>(x.getDim());
	ones.reset(2);
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
	if(config.obj==cfg::Objective::rbulk){
		robust_bulk(config);
	}
	else if(config.obj==cfg::Objective::rshear){
		robust_shear(config);
	}
	else if(config.obj==cfg::Objective::rnpr){
		robust_npr(config);
		//example_opti_npr(config);
	}
	else if(config.obj==cfg::Objective::test){
		//robust_npr(config);
		mytest(config);
	}
	//
	//robust_shear(config);
	//robust_npr(config);
	//mytest(config);
}


