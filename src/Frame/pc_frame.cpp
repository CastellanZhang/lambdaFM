#include "pc_frame.h"

bool pc_frame::init(pc_task& task, int t_num, int buf_size, int log_num)
{
    pTask = &task;
    threadNum = t_num;
    bufSize = buf_size;
    logNum = log_num;
    sem_init(&semPro, 0, 1);
    sem_init(&semCon, 0, 0);
    threadVec.clear();
    threadVec.push_back(thread(&pc_frame::proThread, this));
    for(int i = 0; i < threadNum; ++i)
    {
        threadVec.push_back(thread(&pc_frame::conThread, this));
    }
    return true;
}


void pc_frame::run()
{
    for(int i = 0; i < threadVec.size(); ++i)
    {
        threadVec[i].join();
    }
}

void pc_frame::proThread()
{
    string line;
    int line_num = 0;
    int i = 0;
    bool finished_flag = false;
    bool first_line = true;
    string last_qid = "";
    vector<fm_sample> sampleGrp;
    while(true)
    {
        sem_wait(&semPro);
        bufMtx.lock();
        for(i = 0; i < bufSize;)
        {
            if(!getline(cin, line))
            {
                if(!sampleGrp.empty())
                {
                    buffer.push(sampleGrp);
                    sampleGrp.clear();
                }
                finished_flag = true;
                break;
            }
            line_num++;
            fm_sample sample(line);
            if(!first_line && last_qid != sample.qid)
            {
                buffer.push(sampleGrp);
                i += sampleGrp.size();
                sampleGrp.clear();
                last_qid = sample.qid;
            }
            sampleGrp.push_back(sample);
            if(first_line)
            {
                first_line = false;
                last_qid = sample.qid;
            }
            if(line_num%logNum == 0)
            {
                cout << line_num << " lines have finished" << endl;
            }
        }
        bufMtx.unlock();
        sem_post(&semCon);
        if(finished_flag)
        {
            break;
        }
    }
}


void pc_frame::conThread()
{
    bool finished_flag = false;
    vector<vector<fm_sample> > input_vec;
    while(true)
    {
        input_vec.clear();
        sem_wait(&semCon);
        bufMtx.lock();
        for(int i = 0; i < bufSize;)
        {
            if(buffer.empty())
            {
                finished_flag = true;
                break;
            }
            input_vec.push_back(buffer.front());
            i += buffer.front().size();
            buffer.pop();
        }
        bufMtx.unlock();
        sem_post(&semPro);
        pTask->run_task(input_vec);
        if(finished_flag)
            break;
    }
    sem_post(&semCon);
}

