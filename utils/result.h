#ifndef ROBUST_RESULT_H
#define ROBUST_RESULT_H
#include "utils/robust.h"
#include<string>

using std::string;


void robust_result_eroe(char* path){
    robust_result_erode(string(path));
}

#endif