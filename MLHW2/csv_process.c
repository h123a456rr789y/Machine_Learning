#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define nodeNum 1400
#define K 3
#define square(x) ((x)*(x))

//srand(time(NULL)); // random

struct Node{
    double x, y;
    int Cluster=-1;
};

void extract(char* line,  struct Node node[], char *target[], int numofline){
    int num=0;
    char* tok;
    for(tok = strtok(line, ",\n"); tok && *tok; tok = strtok(NULL, ",\n"))
    {
        num++;
        if(num == 1){
            node[numofline].x = strtod(tok, NULL);
        }
        else if(num == 2){
            node[numofline].y = strtod(tok, NULL);
        }
        else if(num == 3){
            target[numofline] = strdup(tok);
        }
    }
    //return NULL;
}
void charToInt(char *target[], int targetInInt[]){
    for(int i=1; i<=1321; i++){
        if(!strcmp(target[i], "FF")){
            targetInInt[i] = 1;
        }
        else if(!strcmp(target[i], "CH")){
            targetInInt[i] = 2;
        }
        else if(!strcmp(target[i], "CU")){
            targetInInt[i] = 3;
        }
    }
}
void genNRandNumber(int arr[], int totalNum, int maximumNum){ // totalNum>1
    srand(time(NULL)); // random
    memset(arr, 0, sizeof(arr));
    int totalDistinctNum = 0;
    while(totalDistinctNum < totalNum){
        bool dup=false;
        int r = rand()%maximumNum+1;
        for(int i=0; i<totalNum; i++){
            if(arr[i] == r){
                dup=true;
            }
        }
        if(!dup){
            arr[totalDistinctNum]=r;
            totalDistinctNum++;
        }
    }
}
double distance(double x1, double y1, double x2, double y2){
    return square(x1-x2)+square(y1-y2);
}
int main()
{
    srand(time(NULL)); // random
    char *target[nodeNum]; // FF; CH; CU

    struct Node feature_node[nodeNum];   
    struct Node center_node[K];

    FILE* stream = fopen("processed_data_noah.csv", "r"); // read file

    char line[10000]; // read each line
    int numofline=0; // argument for extract()

    while(fgets(line, 10000, stream)) // scan each line
    {
        if(numofline==0){ // the first line is "x,y,pitch_type"
            numofline++;
            continue;
        }
        char* tmp = strdup(line);
        extract(tmp, feature_node, target, numofline);
        free(tmp);
        numofline++;
    }

    int targetInInt[2000]; // 1 for "FF"; 2 for "CH"; 3 for "CU"
    
    charToInt(target, targetInInt);
    
    // for(int i=1; i<=1321; i++){
    //     printf("%d : %f %f %d\n", i, node[i].x, node[i].y, targetInInt[i]);
    // }

    int randseed[K];

    genNRandNumber(randseed, K, 1321);

    for(int i=0; i<K; i++){
        center_node[i].x = feature_node[randseed[i]].x;
        center_node[i].y = feature_node[randseed[i]].y;
    }
    int iterTimes=0;
    while(iterTimes<10){
        for(int i=1; i<1321; i++){
            double minDist=1000000;
            int CenterWithMinDist = -1;
            for(int j=0; j<K; j++){
                double dis = distance(feature_node[i].x, feature_node[i].y, center_node[j].x, center_node[j].y);
                if(minDist>dis){
                    minDist=dis;
                    CenterWithMinDist=j;
                }
            }
            feature_node[i].Cluster=CenterWithMinDist;
        }
        double xSum[K]=0, ySum[K]=0, xMean[K]=0, yMean[K]=0;
        int numofdots[K]=0;
        for(int i=1; i<=1321; i++){
            xSum[feature_node[i].Cluster] += feature_node[i].x - center_node[feature_node[i].Cluster].x;
            ySum[feature_node[i].Cluster] += feature_node[i].y - center_node[feature_node[i].Cluster].y;
            numofdots[feature_node[i].Cluster]++;
        }
        for(int i=0; i<K; i++){
            xMean[i]=xSum[i]/numofdots[i];
            yMean[i]=ySum[i]/numofdots[i];
            center_node[i].x=xMean[i];
            center_node[i].y=yMean[i];
        }
        iterTimes++;
    }

    return 0;
}
