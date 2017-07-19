#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include <unordered_set>
#include <unordered_map>
#include<algorithm>
#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"

using namespace std;

struct trainer_option
{
    trainer_option() : k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), 
               w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0), 
               v_alpha(0.05), v_beta(1.0), v_l1(0.1), v_l2(5.0), 
               threads_num(1), b_init(false), force_v_sparse(false), 
               rank("pairwise"), fast_mode(false) {}
    string model_path, init_m_path;
    double init_mean, init_stdev;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    int threads_num, factor_num;
    bool k1, b_init, force_v_sparse;
    string rank;
    bool fast_mode;
    
    void parse_option(const vector<string>& args) 
    {
        int argc = args.size();
        if(0 == argc) throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            if(args[i].compare("-m") == 0) 
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_path = args[++i];
            }
            else if(args[i].compare("-dim") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                vector<string> strVec;
                string tmpStr = args[++i];
                utils::splitString(tmpStr, ',', &strVec);
                if(strVec.size() != 2)
                    throw invalid_argument("invalid command\n");
                k1 = 0 == stoi(strVec[0]) ? false : true;
                factor_num = stoi(strVec[1]);
            }
            else if(args[i].compare("-init_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_stdev = stod(args[++i]);
            }
            else if(args[i].compare("-w_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-w_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_beta = stod(args[++i]);
            }
            else if(args[i].compare("-w_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-w_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-v_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-v_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_beta = stod(args[++i]);
            }
            else if(args[i].compare("-v_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-v_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                threads_num = stoi(args[++i]);
            }
            else if(args[i].compare("-im") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_m_path = args[++i];
                b_init = true; //if im field exits , that means b_init = true !
            }
            else if(args[i].compare("-fvs") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                int fvs = stoi(args[++i]);
                force_v_sparse = (1 == fvs) ? true : false;
            }
            else if(args[i].compare("-rank") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                rank = args[++i];
                if(rank != "pairwise" && rank != "ndcg")
                    throw invalid_argument("invalid command\n");
            }
            else if(args[i].compare("-fast_mode") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                int f_m = stoi(args[++i]);
                fast_mode = (1 == f_m) ? true : false;
            }
            else
            {
                throw invalid_argument("invalid command\n");
                break;
            }
        }
    }

};


struct variable_NDCG
{
    vector<double> vecGain;
    vector<double> vecDCG;
    vector<double> vecIdealDCG;
    double DCG;
    double idealDCG;
    void init(int size)
    {
        vecGain.resize(size);
        vecDCG.resize(size);
        vecIdealDCG.resize(size);
    }
};


class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option& opt);
    virtual void run_task(vector<vector<fm_sample> >& dataBuffer);
    bool loadModel(ifstream& in);
    void outputModel(ofstream& out);
private:
    void trainGroup(vector<fm_sample>& sampleGrp);
    void trainPair(fm_sample& sample1, fm_sample& sample2, const double deltaNDCG);
    void trainGroupFastMode(vector<fm_sample>& sampleGrp);
    void trainPairFastMode(fm_sample& sample1, fm_sample& sample2, double& pred1, double& pred2,
        map<string, vector<double> >& grad1, map<string, vector<double> >& grad2, const double deltaNDCG);
    //NDCG
    inline double discount(int i);
    inline double gain(int y);
    void calcVariables4NDCG(vector<fm_sample>& sampleGrp, variable_NDCG& v_NDCG);
    inline double calcDeltaNDCG(int i, int j, variable_NDCG& v_NDCG);
private:
    ftrl_model* pModel;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    bool k1;
    bool force_v_sparse;
    string rank;
    bool fast_mode;
    vector<double> vecDiscount;//cache for NDCG
    static const int cacheSize = 5000;//cache size for NDCG
};


ftrl_trainer::ftrl_trainer(const trainer_option& opt)
{
    w_alpha = opt.w_alpha;
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    v_alpha = opt.v_alpha;
    v_beta = opt.v_beta;
    v_l1 = opt.v_l1;
    v_l2 = opt.v_l2;
    k1 = opt.k1;
    force_v_sparse = opt.force_v_sparse;
    rank = opt.rank;
    fast_mode = opt.fast_mode;
    pModel = new ftrl_model(opt.factor_num, opt.init_mean, opt.init_stdev);
    vecDiscount.reserve(cacheSize);
    for(int i = 0; i < cacheSize; ++i)
    {
        vecDiscount.push_back(1.0/log2(i + 2.0));// discount(i) = 1/log2(i+1), i from 1 to ... 
    }
}


void ftrl_trainer::run_task(vector<vector<fm_sample> >& dataBuffer)
{
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        if(fast_mode)
        {
            trainGroupFastMode(dataBuffer[i]);
        }
        else
        {
            trainGroup(dataBuffer[i]);
        }
    }
}


void ftrl_trainer::trainGroup(vector<fm_sample>& sampleGrp)
{
    int size = sampleGrp.size();
    variable_NDCG v_NDCG;
    if("ndcg" == rank)
    {
        v_NDCG.init(size);
        calcVariables4NDCG(sampleGrp, v_NDCG);
    }
    for(int i = 0; i < size - 1; ++i)
    {
        for(int j = i + 1; j < size; ++j)
        {
            if(sampleGrp[i].y == sampleGrp[j].y)
            {
                continue;
            }
            double deltaNDCG = 1.0;
            if("ndcg" == rank)
            {
                deltaNDCG = calcDeltaNDCG(i, j, v_NDCG);
            }
            if(sampleGrp[i].y > sampleGrp[j].y)
            {
                trainPair(sampleGrp[i], sampleGrp[j], deltaNDCG);
            }
            else if(sampleGrp[i].y < sampleGrp[j].y)
            {
                trainPair(sampleGrp[j], sampleGrp[i], deltaNDCG);
            }
        }
    }
}


void ftrl_trainer::trainPair(fm_sample& sample1, fm_sample& sample2, const double deltaNDCG)
{
    unordered_map<string, ftrl_model_unit*> theta;
    for(unordered_map<string, double>::iterator iter = sample1.x.begin(); iter != sample1.x.end(); ++iter)
    {
        const string& index = iter->first;
        if(theta.find(index) == theta.end())
        {
            theta[index] = pModel->getOrInitModelUnit(index);
        }
    }
    for(unordered_map<string, double>::iterator iter = sample2.x.begin(); iter != sample2.x.end(); ++iter)
    {
        const string& index = iter->first;
        if(theta.find(index) == theta.end())
        {
            theta[index] = pModel->getOrInitModelUnit(index);
        }
    }
    //update w via FTRL
    if(k1)
    {
        for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
        {
            ftrl_model_unit& mu = *(iter->second);
            mu.mtx.lock();
            if(fabs(mu.w_zi) <= w_l1)
            {
                mu.wi = 0.0;
            }
            else
            {
                if(force_v_sparse && mu.w_ni > 0 && 0.0 == mu.wi)
                {
                    mu.reinit_vi(pModel->init_mean, pModel->init_stdev);
                }
                mu.wi = (-1) *
                    (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                    (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);
            }
            mu.mtx.unlock();
        }
    }
    //update v via FTRL
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
    {
        ftrl_model_unit& mu = *(iter->second);
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
            double& vif = mu.vi[f];
            double& v_nif = mu.v_ni[f];
            double& v_zif = mu.v_zi[f];
            if(v_nif > 0)
            {
                if(force_v_sparse && 0.0 == mu.wi)
                {
                    vif = 0.0;
                }
                else if(fabs(v_zif) <= v_l1)
                {
                    vif = 0.0;
                }
                else
                {
                    vif = (-1) *
                        (1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
                        (v_zif - utils::sgn(v_zif) * v_l1);
                }
            }
            mu.mtx.unlock();
        }
    }
    vector<double> sum1(pModel->factor_num);
    double p1 = pModel->predict(sample1.x, theta, sum1);
    vector<double> sum2(pModel->factor_num);
    double p2 = pModel->predict(sample2.x, theta, sum2);
    double lambda12 = -1 / (1 + exp(p1 - p2)) * deltaNDCG;
    //update w_n, w_z
    if(k1)
    {
        for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
        {
            ftrl_model_unit& mu = *(iter->second);
            const string& index = iter->first;
            unordered_map<string, double>::iterator iter_x1 = sample1.x.find(index);
            double xi1 = (iter_x1 == sample1.x.end()) ? 0 : iter_x1->second;
            unordered_map<string, double>::iterator iter_x2 = sample2.x.find(index);
            double xi2 = (iter_x2 == sample2.x.end()) ? 0 : iter_x2->second;
            double xi = xi1 - xi2;
            if(xi != 0)
            {
                mu.mtx.lock();
                double w_gi = lambda12 * xi;
                double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
                mu.w_zi += w_gi - w_si * mu.wi;
                mu.w_ni += w_gi * w_gi;
                mu.mtx.unlock();
            }
        }
    }
    //update v_n, v_z
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
    {
        ftrl_model_unit& mu = *(iter->second);
        const string& index = iter->first;
        unordered_map<string, double>::iterator iter_x1 = sample1.x.find(index);
        double xi1 = (iter_x1 == sample1.x.end()) ? 0 : iter_x1->second;
        unordered_map<string, double>::iterator iter_x2 = sample2.x.find(index);
        double xi2 = (iter_x2 == sample2.x.end()) ? 0 : iter_x2->second;
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
            double& vif = mu.vi[f];
            double& v_nif = mu.v_ni[f];
            double& v_zif = mu.v_zi[f];
            double v_gif = lambda12 * ((sum1[f] * xi1 - vif * xi1 * xi1) - (sum2[f] * xi2 - vif * xi2 * xi2));
            double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
            v_zif += v_gif - v_sif * vif;
            v_nif += v_gif * v_gif;
            //有的特征在整个训练集中只出现一次，这里还需要对vif做一次处理
            if(force_v_sparse && v_nif > 0 && 0.0 == mu.wi)
            {
                vif = 0.0;
            }
            mu.mtx.unlock();
        }
    }
}


bool ftrl_trainer::loadModel(ifstream& in)
{
    return pModel->loadModel(in);
}


void ftrl_trainer::outputModel(ofstream& out)
{
    return pModel->outputModel(out);
}


void ftrl_trainer::trainGroupFastMode(vector<fm_sample>& sampleGrp)
{
    int size = sampleGrp.size();
    variable_NDCG v_NDCG;
    if("ndcg" == rank)
    {
        v_NDCG.init(size);
        calcVariables4NDCG(sampleGrp, v_NDCG);
    }
    int factor_num = pModel->factor_num;
    vector<double> vecPred(size);
    vector<map<string, vector<double> > > vecGradPred2v(size);
    unordered_map<string, ftrl_model_unit*> theta;
    for(int i = 0; i < size; ++i)
    {
        for(unordered_map<string, double>::iterator iter = sampleGrp[i].x.begin(); iter != sampleGrp[i].x.end(); ++iter)
        {
            const string& index = iter->first;
            if(theta.find(index) == theta.end())
            {
                theta[index] = pModel->getOrInitModelUnit(index);
            }
        }
        vector<double> sum(factor_num);
        vecPred[i] = pModel->predict(sampleGrp[i].x, theta, sum);
        for(unordered_map<string, double>::iterator iter = sampleGrp[i].x.begin(); iter != sampleGrp[i].x.end(); ++iter)
        {
            const string& index = iter->first;
            const double& xi = iter->second;
            vector<double>& vi = theta[index]->vi;
            vecGradPred2v[i][index] = vector<double>(factor_num);
            for(int f = 0; f < factor_num; ++f)
            {
                vecGradPred2v[i][index][f] = sum[f] * xi - vi[f] * xi * xi;
            }
        }
    }
    for(int i = 0; i < size - 1; ++i)
    {
        for(int j = i + 1; j < size; ++j)
        {
            if(sampleGrp[i].y == sampleGrp[j].y)
            {
                continue;
            }
            double deltaNDCG = 1.0;
            if("ndcg" == rank)
            {
                deltaNDCG = calcDeltaNDCG(i, j, v_NDCG);
            }
            if(sampleGrp[i].y > sampleGrp[j].y)
            {
                trainPairFastMode(sampleGrp[i], sampleGrp[j], vecPred[i], vecPred[j], vecGradPred2v[i], vecGradPred2v[j], deltaNDCG);
            }
            else if(sampleGrp[i].y < sampleGrp[j].y)
            {
                trainPairFastMode(sampleGrp[j], sampleGrp[i], vecPred[j], vecPred[i], vecGradPred2v[j], vecGradPred2v[i], deltaNDCG);
            }
        }
    }
}



void ftrl_trainer::trainPairFastMode(fm_sample& sample1, fm_sample& sample2, double& pred1, double& pred2,
    map<string, vector<double> >& grad1, map<string, vector<double> >& grad2, const double deltaNDCG)
{
    unordered_map<string, ftrl_model_unit*> theta;
    for(unordered_map<string, double>::iterator iter = sample1.x.begin(); iter != sample1.x.end(); ++iter)
    {
        const string& index = iter->first;
        if(theta.find(index) == theta.end())
        {
            theta[index] = pModel->getOrInitModelUnit(index);
        }
    }
    for(unordered_map<string, double>::iterator iter = sample2.x.begin(); iter != sample2.x.end(); ++iter)
    {
        const string& index = iter->first;
        if(theta.find(index) == theta.end())
        {
            theta[index] = pModel->getOrInitModelUnit(index);
        }
    }
    //update w via FTRL
    if(k1)
    {
        for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
        {
            ftrl_model_unit& mu = *(iter->second);
            mu.mtx.lock();
            if(fabs(mu.w_zi) <= w_l1)
            {
                mu.wi = 0.0;
            }
            else
            {
                if(force_v_sparse && mu.w_ni > 0 && 0.0 == mu.wi)
                {
                    mu.reinit_vi(pModel->init_mean, pModel->init_stdev);
                }
                mu.wi = (-1) *
                    (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                    (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);
            }
            mu.mtx.unlock();
        }
    }
    //update v via FTRL
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
    {
        ftrl_model_unit& mu = *(iter->second);
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
            double& vif = mu.vi[f];
            double& v_nif = mu.v_ni[f];
            double& v_zif = mu.v_zi[f];
            if(v_nif > 0)
            {
                if(force_v_sparse && 0.0 == mu.wi)
                {
                    vif = 0.0;
                }
                else if(fabs(v_zif) <= v_l1)
                {
                    vif = 0.0;
                }
                else
                {
                    vif = (-1) *
                        (1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
                        (v_zif - utils::sgn(v_zif) * v_l1);
                }
            }
            mu.mtx.unlock();
        }
    }
    double lambda12 = -1 / (1 + exp(pred1 - pred2)) * deltaNDCG;
    //update w_n, w_z
    if(k1)
    {
        for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
        {
            ftrl_model_unit& mu = *(iter->second);
            const string& index = iter->first;
            unordered_map<string, double>::iterator iter_x1 = sample1.x.find(index);
            double xi1 = (iter_x1 == sample1.x.end()) ? 0 : iter_x1->second;
            unordered_map<string, double>::iterator iter_x2 = sample2.x.find(index);
            double xi2 = (iter_x2 == sample2.x.end()) ? 0 : iter_x2->second;
            double xi = xi1 - xi2;
            if(xi != 0)
            {
                mu.mtx.lock();
                double w_gi = lambda12 * xi;
                double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
                mu.w_zi += w_gi - w_si * mu.wi;
                mu.w_ni += w_gi * w_gi;
                mu.mtx.unlock();
            }
        }
    }
    //update v_n, v_z
    vector<double> vecZero(pModel->factor_num, 0);
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = theta.begin(); iter != theta.end(); ++iter)
    {
        ftrl_model_unit& mu = *(iter->second);
        const string& index = iter->first;
        unordered_map<string, double>::iterator iter_x1 = sample1.x.find(index);
        double xi1 = (iter_x1 == sample1.x.end()) ? 0 : iter_x1->second;
        unordered_map<string, double>::iterator iter_x2 = sample2.x.find(index);
        double xi2 = (iter_x2 == sample2.x.end()) ? 0 : iter_x2->second;
        vector<double>* pGrad1Index = &vecZero;
        map<string, vector<double> >::iterator gradIter1 = grad1.find(index);
        if(gradIter1 != grad1.end())
        {
            pGrad1Index = &(gradIter1->second);
        }
        vector<double>* pGrad2Index = &vecZero;
        map<string, vector<double> >::iterator gradIter2 = grad2.find(index);
        if(gradIter2 != grad2.end())
        {
            pGrad2Index = &(gradIter2->second);
        }
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
            double& vif = mu.vi[f];
            double& v_nif = mu.v_ni[f];
            double& v_zif = mu.v_zi[f];
            double v_gif = lambda12 * ((*pGrad1Index)[f] - (*pGrad2Index)[f]);
            double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
            v_zif += v_gif - v_sif * vif;
            v_nif += v_gif * v_gif;
            //有的特征在整个训练集中只出现一次，这里还需要对vif做一次处理
            if(force_v_sparse && v_nif > 0 && 0.0 == mu.wi)
            {
                vif = 0.0;
            }
            mu.mtx.unlock();
        }
    }
}


inline double ftrl_trainer::discount(int i)
{
    if(i < cacheSize) return vecDiscount[i];
    return 1.0/log2(i + 2.0);
}


inline double ftrl_trainer::gain(int y)
{
    return (1<<y) - 1;// gain(y) = 2^y-1
}


void ftrl_trainer::calcVariables4NDCG(vector<fm_sample>& sampleGrp, variable_NDCG& v_NDCG)
{
    int size = sampleGrp.size();
    v_NDCG.DCG = 0;
    for(int i = 0; i < size; ++i)
    {
        v_NDCG.vecGain[i] = gain(sampleGrp[i].y);
        v_NDCG.vecDCG[i] = v_NDCG.vecGain[i] * discount(i);
        v_NDCG.DCG += v_NDCG.vecDCG[i];
    }
    v_NDCG.vecIdealDCG.assign(v_NDCG.vecGain.begin(), v_NDCG.vecGain.end());
    v_NDCG.idealDCG = 0;
    sort(v_NDCG.vecIdealDCG.begin(), v_NDCG.vecIdealDCG.end(),greater<double>());
    for(int i = 0; i < size; ++i)
    {
        v_NDCG.vecIdealDCG[i] *= discount(i);
        v_NDCG.idealDCG += v_NDCG.vecIdealDCG[i];
    }
}


inline double ftrl_trainer::calcDeltaNDCG(int i, int j, variable_NDCG& v_NDCG)
{
    double deltaDCG = v_NDCG.vecGain[j] * discount(i) + v_NDCG.vecGain[i] * discount(j) - v_NDCG.vecDCG[i] - v_NDCG.vecDCG[j];
    return fabs(deltaDCG/v_NDCG.idealDCG);
}


#endif /*FTRL_TRAINER_H_*/
