#ifndef PC_TASK_H
#define PC_TASK_H

#include <vector>
#include "../Sample/fm_sample.h"

using std::vector;

class pc_task
{
public:
    pc_task(){}
    virtual void run_task(vector<vector<fm_sample> >& dataBuffer) = 0;
};


#endif //PC_TASK_H
