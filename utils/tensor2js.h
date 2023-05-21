#pragma once

#ifndef TENSOR2JS_H
#define TENSOR2JS_H

#include<fstream>
#include<string>
#include"nlohmann/json.hpp"

class matlabjs{
    public:
        matlabjs(std::string str):name(str){};
        ~matlabjs(){
            std::ofstream o(getPath(name));
            o<<std::setw(4)<<js<<std::endl;
        }
        auto& operator[] (std::string str){
            return js[str];
        }

    //interfaces:
    void tensor2js(const std::string& tname, const TensorView<float>& tf);
    void tensor2js(const std::string& tname, const TensorView<double>& tf);
    private:
        nlohmann::json js;
        std::string name;
};

void matlabjs::tensor2js(const std::string& tname, const TensorView<double>& tf) {
        std::vector<double> vec(tf.size());
		cudaMemcpy2D(&vec[0], tf.size(0) * sizeof(double),
			tf.data(), tf.getPitchT() * sizeof(double), tf.size(0) * sizeof(double), tf.size(1) * tf.size(2), cudaMemcpyDeviceToHost);
		cuda_error_check;
		(*this)[tname]=vec;
	}

void matlabjs::tensor2js(const std::string& tname, const TensorView<float>& tf) {
        std::vector<float> vec(tf.size());
		cudaMemcpy2D(&vec[0], tf.size(0) * sizeof(float),
			tf.data(), tf.getPitchT() * sizeof(float), tf.size(0) * sizeof(float), tf.size(1) * tf.size(2), cudaMemcpyDeviceToHost);
		cuda_error_check;
		(*this)[tname]=vec;
	}

#endif