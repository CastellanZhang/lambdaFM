#ifndef TEST_TASK_H
#define TEST_TASK_H

#include <iostream>
#include "pc_task.h"
using namespace std;

class test_task : public pc_task
{
public:
    test_task(){}
    virtual void run_task(vector<vector<fm_sample> >& dataBuffer)
    {
        cout << "==========\n";
        for(int i = 0; i < dataBuffer.size(); ++i)
        {
            for(int j = 0; j < dataBuffer[i].size(); ++j)
            {
                cout << dataBuffer[i][j].y << " " << dataBuffer[i][j].qid << " ";
                for(int k = 0; k < dataBuffer[i][j].x.size(); ++k)
                {
                    cout << dataBuffer[i][j].x[k].first << ":" << dataBuffer[i][j].x[k].second << " ";
                }
                cout << endl;
            }
        }
        cout << "**********\n";
    }
};


#endif //TEST_TASK_H
