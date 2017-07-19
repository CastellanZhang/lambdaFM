#ifndef FTRL_PREDICTOR_H_
#define FTRL_PREDICTOR_H_

#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"


class ftrl_predictor : public pc_task
{
public:
    ftrl_predictor(double _factor_num, ifstream& _fModel, ofstream& _fPredict);
    virtual void run_task(vector<vector<fm_sample> >& dataBuffer);
private:
    ftrl_model* pModel;
    ofstream& fPredict;
    mutex outMtx;
};


ftrl_predictor::ftrl_predictor(double _factor_num, ifstream& _fModel, ofstream& _fPredict):fPredict(_fPredict)
{
    pModel = new ftrl_model(_factor_num);
    if(!pModel->loadModel(_fModel))
    {
        cout << "load model error!" << endl;
        exit(-1);
    }
}

void ftrl_predictor::run_task(vector<vector<fm_sample> >& dataBuffer)
{
    vector<string> outputVec;
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        for(int j = 0; j < dataBuffer[i].size(); ++j)
        {
            fm_sample& sample = dataBuffer[i][j];
            double score = pModel->getScore(sample.x, pModel->muMap);
            outputVec.push_back(to_string(sample.y) + " " + sample.qid + " " + to_string(score));
        }
    }
    outMtx.lock();
    for(int i = 0; i < outputVec.size(); ++i)
    {
        fPredict << outputVec[i] << endl;
    }
    outMtx.unlock();
}


#endif /*FTRL_PREDICTOR_H_*/
