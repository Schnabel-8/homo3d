#include "OCOptimizer.h"
#include "cuda_runtime.h"
#include "culib/lib.cuh"
#include "AutoDiff/TensorExpression.h"


using namespace homo;
using namespace culib;

template<typename T>
__global__ void update_kernel(int ne,
	const T* sens, T g,
	const T* rhoold, T* rhonew,
	T minRho, T stepLimit, T damp) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	
	T rho = rhoold[tid];

	T B = -sens[tid] / g;
	if (B < 0) B = 0.01f;
	T newrho = powf(B, damp) * rho;
	//if (tid == 0) {
	//	printf("sens = %.4e  g = %.4e  damp = %.4e  rho =%.4e  newrho = %.4e\n",
	//		sens[tid], g, damp, rho, newrho);
	//}

	if (newrho - rho < -stepLimit) newrho = rho - stepLimit;
	if (newrho - rho > stepLimit) newrho = rho + stepLimit;
	if (newrho < minRho) newrho = minRho;
	if (newrho > 1) newrho = 1;
	rhonew[tid] = newrho;
}


//_Performance_
void OCOptimizer::update(const float* sens, float* rho, float volratio) {
	float* newrho;
	cudaMalloc(&newrho, sizeof(float) * ne);
	float maxSens = abs(parallel_maxabs(sens, ne));
	printf("max sens = %f\n", maxSens);
	float minSens = 0;
	for (int itn = 0; itn < 20; itn++) {
		float gSens = (maxSens + minSens) / 2;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, ne, 256);
		update_kernel << <grid_size, block_size >> > (ne, sens, gSens, rho, newrho,
			minRho, step_limit, damp);
		cudaDeviceSynchronize();
		cuda_error_check;
		float curVol = parallel_sum(newrho, ne) / ne;
		printf("[OC] : g = %.4e   vol = %4.2f%% (Goal %4.2f%%)       \r", gSens, curVol * 100, volratio * 100);
		if (curVol < volratio - 0.0001) {
			maxSens = gSens;
		}
		else if (curVol > volratio + 0.0001) {
			minSens = gSens;
		}
		else {
			break;
		}
	}
	printf("\n");
	cudaMemcpy(rho, newrho, sizeof(float) * ne, cudaMemcpyDeviceToDevice);
	cudaFree(newrho);
}

__device__ bool is_bounded(int p[3], int reso[3]) {
	return p[0] >= 0 && p[0] < reso[0] &&
		p[1] >= 0 && p[1] < reso[1] &&
		p[2] >= 0 && p[2] < reso[2];
}

template<typename Kernel>
__global__ void filterSens_kernel(
	int ne, devArray_t<int, 3> reso, size_t pitchT,
	const float* sens, const float* rho, const float* weightSum, float* newsens, Kernel wfunc) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	int ereso[3] = { reso[0],reso[1],reso[2] };
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (is_bounded(neighpos, ereso)) {
			int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			//w /= weightSum[neighid];
			sum += sens[neighid] * rho[neighid] * w;
			wsum += w;
		}
	}
	int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	sum /= wsum * rho[eid];
	newsens[eid] = sum;
}

template<typename Kernel>
__global__ void weightSum_kernel(int ne, devArray_t<int, 3> reso, size_t pitchT,
	float* weightSum, Kernel wfunc
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	int ereso[3] = { reso[0],reso[1],reso[2] };
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (is_bounded(neighpos, ereso)) {
			int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			wsum += w;
		}
	}
	int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	weightSum[eid] = wsum;
}

void OCOptimizer::filterSens(float* sens, const float* rho, size_t pitchT, int reso[3], float radius)
{
	static float* filterWeightSum = nullptr;
	if (!filterWeightSum) {
		cudaMalloc(&filterWeightSum, sizeof(float) * reso[1] * reso[2] * pitchT);
		init_array(filterWeightSum, float(0), reso[1] * reso[2] * pitchT);
	}
	float* newsens;
	cudaMalloc(&newsens, sizeof(float) * reso[1] * reso[2] * pitchT);
	radial_convker_t<float, Linear> convker(radius, 0, true, false);
	devArray_t<int, 3> ereso{ reso[0],reso[1],reso[2] };
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, ne, 256);
	filterSens_kernel << <grid_size, block_size >> > (ne, ereso, pitchT, sens, rho, filterWeightSum, newsens, convker);
	cudaDeviceSynchronize();
	cuda_error_check;
	cudaMemcpy(sens, newsens, sizeof(float) * reso[1] * reso[2] * pitchT, cudaMemcpyDeviceToDevice);
	cudaFree(newsens);
}

template<typename Kernel>
__global__ void filterSens_Tensor_kernel(
	TensorView<float> sens, TensorView<float> rho, TensorView<float> newsens, Kernel wfunc) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rho.size();
	int reso[3] = { rho.size(0),rho.size(1),rho.size(2) };
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, reso)) {
			//int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			//w /= weightSum[neighid];
			sum += sens(neighpos[0], neighpos[1], neighpos[2]) * rho(neighpos[0], neighpos[1], neighpos[2]) * w;
			wsum += w;
		} 
	}
	//int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	float tmp=0.001;
	if(rho(epos[0], epos[1], epos[2])>tmp){
		tmp=rho(epos[0], epos[1], epos[2]);
	}
	sum /= wsum * tmp;
	newsens(epos[0], epos[1], epos[2]) = sum;
}

void OCOptimizer::filterSens(Tensor<float> sens, Tensor<float> rho, float radius /*= 2*/) {
	Tensor<float> newsens(rho.getDim());
	newsens.reset(0);
	radial_convker_t<float, Linear> convker(radius, 0, true, false);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, rho.size(), 256);
	filterSens_Tensor_kernel << <grid_size, block_size >> > (sens.view(), rho.view(), newsens.view(), convker);
	cudaDeviceSynchronize();
	cuda_error_check;
	sens.copy(newsens);
}

template<typename T>
__global__ void update_Tensor_kernel(TensorView<T> sens, T g,
	TensorView<T> rhoold, TensorView<T> rhonew,TensorView<T> rhophys,
	T minRho, T stepLimit, T damp,T beta) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rhoold.size();
	if (tid >= ne) return;

	T rho = rhoold(tid);

	T B = -sens(tid) / g;
	if (B < 0) B = 0.01f;
	T newrho = powf(B, damp) * rho;

	if (newrho - rho < -stepLimit) newrho = rho - stepLimit;
	if (newrho - rho > stepLimit) newrho = rho + stepLimit;
	if (newrho < minRho) newrho = minRho;
	if (newrho > 1) newrho = 1;
	rhonew(tid) = newrho;
	T newrho2=newrho*2;
	T ret=0;
	if(newrho<=0.5){
					ret=(exp(double(-beta*(1-newrho2)))-(1-newrho2)*exp(double(-beta)))/2;
				}
				else{
					ret=(1-exp(double(-beta*(newrho2-1)))+(newrho2-1)*exp(double(-beta)))/2+0.5;
				}
	rhophys(tid)=ret;
}

float OCOptimizer::update(Tensor<float> sens, Tensor<float> rho, float volratio,float beta) {
	Tensor<float> newrho(rho.getDim());
	newrho.reset(0);
	Tensor<float> physrho(rho.getDim());
	physrho.reset(0);
	float maxSens = abs(sens.maxabs());
	printf("max sens = %f\n", maxSens);
	float minSens = 0;
	for (int itn = 0; itn < 20; itn++) {
		float gSens = (maxSens + minSens) / 2;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, rho.size(), 256);
		update_Tensor_kernel << <grid_size, block_size >> > (sens.view(), gSens, rho.view(), newrho.view(),physrho.view(),
			minRho, step_limit, damp,beta);
		cudaDeviceSynchronize();
		cuda_error_check;
		//float curVol = parallel_sum(newrho, ne) / ne;
		float curVol = physrho.sum() / physrho.size();
		printf("[OC] : g = %.4e   vol = %4.2f%% (Goal %4.2f%%)       \r", gSens, curVol * 100, volratio * 100);
		if (curVol < volratio - 0.0001) {
			maxSens = gSens;
		}
		else if (curVol > volratio + 0.0001) {
			minSens = gSens;
		}
		else {
			break;
		}
	}
	printf("\n");
	rho.copy(newrho);
	return physrho.sum() / physrho.size();
}

template<typename T>
__global__ void update_Tensor_kernel(TensorView<T> sens, T g,
	TensorView<T> rhoold, TensorView<T> rhonew,
	T minRho, T stepLimit, T damp) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rhoold.size();
	if (tid >= ne) return;

	T rho = rhoold(tid);

	T B = -sens(tid) / g;
	if (B < 0) B = 0.01f;
	T newrho = powf(B, damp) * rho;

	if (newrho - rho < -stepLimit) newrho = rho - stepLimit;
	if (newrho - rho > stepLimit) newrho = rho + stepLimit;
	if (newrho < minRho) newrho = minRho;
	if (newrho > 1) newrho = 1;
	rhonew(tid) = newrho;
}

void OCOptimizer::update(Tensor<float> sens, Tensor<float> rho, float volratio) {
	Tensor<float> newrho(rho.getDim());
	newrho.reset(0);
	float maxSens = abs(sens.maxabs());
	printf("max sens = %f\n", maxSens);
	float minSens = 0;
	for (int itn = 0; itn < 20; itn++) {
		float gSens = (maxSens + minSens) / 2;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, rho.size(), 256);
		update_Tensor_kernel << <grid_size, block_size >> > (sens.view(), gSens, rho.view(), newrho.view(),
			minRho, step_limit, damp);
		cudaDeviceSynchronize();
		cuda_error_check;
		//float curVol = parallel_sum(newrho, ne) / ne;
		float curVol = newrho.sum() / newrho.size();
		printf("[OC] : g = %.4e   vol = %4.2f%% (Goal %4.2f%%)       \r", gSens, curVol * 100, volratio * 100);
		if (curVol < volratio - 0.0001) {
			maxSens = gSens;
		}
		else if (curVol > volratio + 0.0001) {
			minSens = gSens;
		}
		else {
			break;
		}
	}
	printf("\n");
	rho.copy(newrho);
}


template<typename Kernel>
__global__ void robust_filter_proj_kernel(
	TensorView<float> sens, TensorView<float> rho, TensorView<float> newsens,TensorView<float> newsens1, Kernel wfunc,float eta,float beta) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rho.size();
	int reso[3] = { rho.size(0),rho.size(1),rho.size(2) };
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, reso)) {
			//int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			//w /= weightSum[neighid];
			sum += rho(neighpos[0], neighpos[1], neighpos[2])* w;
			wsum += w;
		} 
	}
	//int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	sum /= wsum ;//xTilde


	//xDilate
	double rho1=sum;

	double tmp1=tanh(beta*eta);
	double tmp2=tanh(beta*(rho1-eta));
	double tmp3=tanh(beta*(1-eta));
	double ret=(tmp1+tmp2)/(tmp1+tmp3);
	if(ret<0.0001){
		ret=0.0001;
	}
	//double ret=1-exp(-beta*rho1)+rho1*exp(-beta);
	float proj=ret;
	newsens1(epos[0], epos[1], epos[2]) = proj;

	tmp1=tanh(beta*eta);
	tmp2=1/cosh(beta*(rho1-eta));
	tmp3=tanh(beta*(1-eta));
	ret=beta*pow(double(tmp2),double(2))/(tmp1+tmp3);
	//ret=beta*exp(-beta*rho1)+exp(-beta);

	float projdiff=ret;
	newsens(epos[0], epos[1], epos[2]) = projdiff;
}

float homo::robust_filter_proj(Tensor<float> sens, Tensor<float> rho, float radius /*= 2*/,float eta,float beta) {
	Tensor<float> newsens(rho.getDim());//projdiff
	newsens.reset(0);
	Tensor<float> newsens1(rho.getDim());//proj
	newsens1.reset(0);
	radial_convker_t<float,Spline4> convker(radius, 0, false, false);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, rho.size(), 256);
	robust_filter_proj_kernel << <grid_size, block_size >> > (sens.view(), rho.view(), newsens.view(),newsens1.view(), convker,eta,beta);
	cudaDeviceSynchronize();
	cuda_error_check;
	sens.copy(newsens);
	return newsens1.sum();
}


template<typename Kernel>
__global__ void robust_filter_kernel(
	TensorView<float> sens, TensorView<float> rho, TensorView<float> newsens, Kernel wfunc,float scale) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rho.size();
	int reso[3] = { rho.size(0),rho.size(1),rho.size(2) };
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, reso)) {
			//int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			//w /= weightSum[neighid];
			sum += sens(neighpos[0], neighpos[1], neighpos[2])* w;
			wsum += w;

		} 
	}
	//int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	sum /= wsum;
	newsens(epos[0], epos[1], epos[2]) = sum*scale;

}

void homo::robust_filter(Tensor<float> sens, Tensor<float> rho, float radius /*= 2*/,float scale) {
	Tensor<float> newsens(rho.getDim());
	newsens.reset(0);
	radial_convker_t<float, Spline4> convker(radius, 0, false, false);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, rho.size(), 256);
	robust_filter_kernel << <grid_size, block_size >> > (sens.view(), rho.view(), newsens.view(),convker,scale);
	cudaDeviceSynchronize();
	cuda_error_check;
	sens.copy(newsens);
}



template<typename Kernel>
__global__ void robust_result_filter_kernel(
	TensorView<float> rho, TensorView<float> newsens,Kernel wfunc,float eta,float beta) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rho.size();
	int reso[3] = { rho.size(0),rho.size(1),rho.size(2) };
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, reso)) {
			sum += rho(neighpos[0], neighpos[1], neighpos[2])* w;
			wsum += w;
		} 
	}
	//int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	sum /= wsum ;//xTilde


	//xDilate
	double rho1=sum;

	double tmp1=tanh(beta*eta);
	double tmp2=tanh(beta*(rho1-eta));
	double tmp3=tanh(beta*(1-eta));
	double ret=(tmp1+tmp2)/(tmp1+tmp3);
	float proj=ret;
	if(proj>0.5){
		proj=1;
	}
	else{
		proj=0.0001;
	}
	newsens(epos[0], epos[1], epos[2]) = proj;

}

float homo::robust_result_filter(Tensor<float> rho, float radius /*= 2*/,float eta,float beta) {
	Tensor<float> newsens(rho.getDim());
	newsens.reset(0);
	radial_convker_t<float,Spline4> convker(radius, 0, false, false);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, rho.size(), 256);
	robust_result_filter_kernel << <grid_size, block_size >> > (rho.view(), newsens.view(), convker,eta,beta);
	cudaDeviceSynchronize();
	cuda_error_check;
	rho.copy(newsens);
	return newsens.sum();
}


__global__ void mmadiff_kernel(
	TensorView<float> rho, TensorView<float> newsens,TensorView<float> rho1, TensorView<float> rho2, TensorView<float> rho3,float w1, float w2,float w3) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rho.size();
	int reso[3] = { rho.size(0),rho.size(1),rho.size(2) };
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	
	newsens(epos[0], epos[1], epos[2]) = w1*rho1(epos[0], epos[1], epos[2])+w2*rho2(epos[0], epos[1], epos[2])+w3*rho3(epos[0], epos[1], epos[2]);

}

void homo::mmadiff(Tensor<float> rho,Tensor<float> rho1,Tensor<float> rho2,Tensor<float> rho3,float val,float val1,float val2,float off) {
	Tensor<float> newsens(rho.getDim());
	newsens.reset(0);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, rho.size(), 256);
	double w1,w2,w3;
	double c;
	c=pow(pow((val+off),8)+pow((val+off),8)+pow((val+off),8),0.125);
	w1=pow((val+off)/c,7);
	w2=pow((val1+off)/c,7);
	w3=pow((val2+off)/c,7);
	mmadiff_kernel << <grid_size, block_size >> > (rho.view(), newsens.view(), rho1.view(),rho2.view(),rho3.view(),w1,w2,w3);
	cudaDeviceSynchronize();
	cuda_error_check;
	rho.copy(newsens);
}