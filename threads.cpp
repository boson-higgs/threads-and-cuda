// ***********************************************************************
//
// Threads programming example for Linux (10/2016)
// For the propper testing is necessary to have at least 2 cores CPU
//
// ***********************************************************************
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <pthread.h>
 
#define TYPE int
//#define LENGTH_LIMIT 10000000
#define LENGTH_LIMIT 100
#define THREAD_NUM 8
#define SME_R 0
 
class task_part

{
public:
         int id;
         int *from;
         int *length;
         TYPE *data;
         int smer;
 
        void setdata(int inid, int *infrom, int *inlength, TYPE *indata,int insmer, int totlength)
         {
             id = inid;
             smer = insmer;
 
             from = infrom;
             length = inlength;
             data = indata;
         }
    //task_part( int myide, int *first, int *num, TYPE *ptr, int asc ) :
      // id( myide ), from( first ), length( num ), data( ptr ), smer( asc ) {}
 
    //TYPE get_result() { return max; }
 
    // function search_max search the largest number in part of array
    // from the left (included) up to the right element
    TYPE max()
    {
        int start = from[id-1];
        TYPE max_elem = data[start];
        for ( int i = 1; i < length[id-1]; i++ )
            if ( max_elem < data[ start + i ] )
                max_elem = data[ start + i ];
        return max_elem;
    }
    
    void generator()
    {
        srand( ( int ) time( NULL ) );
        int start = from[id-1];
        int mylength = length[id-1];
        printf( "Random numbers generetion started..." );
        for ( int i = 0; i < mylength; i++ )
        {
                data[ start + i ] = rand() % ( mylength * 10);
                if ( !( i % LENGTH_LIMIT ) )
                {
                    printf( ".");
                    fflush( stdout );
                }
        }
        printf( "%d ", data[mylength] );
        return;
    }
 
    /*void insertionsort()
    {
        int start = from[id-1];
        int mylength = length[id-1];
        printf(\"Sorting the array by insertion sort...\");
        if(smer == 1)
        {
        for(int i = 0; i < mylength - 1; i++)
        {
            int j = i+1;
            TYPE tmp = data[start + j];
            while(j>0 && tmp > data[start + j-1])
            {
                data[start + j] = data[start + j-1];
                j--;
            }
            data[start + j] = tmp;
        }
        }
        else if(smer == 0)
        {
             for(int i = 0; i < mylength - 1; i++)
        {
            int j = i+1;
            TYPE tmp = data[start + j];
            while(j>0 && tmp < data[start + j-1])
            {
                data[start + j] = data[start + j-1];
                j--;
            }
            data[start + j] = tmp;
        }
        }
             
        return;
    }*/
            
               
   void sort()
    {
        int start = from[id-1];
        int mylength = length[id-1];
        printf( "Sorting the array by bubble sort...");
               
        for(int i = 0; i < mylength - 1; i++)
        {

            for(int j = 0; j < mylength - i - 1; j++)
            {
                if(data[start + j + 1] < data[start + j])
                {

                    int tmp = data[start + j + 1];

                    data[start + j + 1] = data[start + j];

                    data[start + j] = tmp;
                         
                }

            }

        }
    }
               
    void merge()
    {
        int start = from[id-1];
        int mylength = length[id-1];
 
        int from2 = from[id];
        int length2 = length[id];
 
        TYPE tmpdata[mylength+length2];
        int i = 0;
        int j = 0;
        int k = 0;
        if (smer == 1)
        {
        while(i < mylength && j < length2)
        {
            if(data[start + i] > data[from2 + j])
            {
                tmpdata[k] = data[start + i];
                i++;
                k++;
            }
            else
            {
                tmpdata[k] = data[from2 + j];
                j++;
                k++;
            }
        }
        }
        if(smer == 0)
        {
            while(i < mylength && j < length2)
        {
            if(data[start + i] < data[from2 + j])
            {
                tmpdata[k] = data[start + i];
                i++;
                k++;
            }
            else
            {
                tmpdata[k] = data[from2 + j];
                j++;
                k++;
            }
        }
        }
        while(i < mylength)
        {
            tmpdata[k++] = data[start + i++];
        }
        while(j < length2)
        {
            tmpdata[k++] = data[from2 + j++];
        }
        for(int i = 0; i < (mylength+length2); i++)
        {
            data[start + i] = tmpdata[i];
        }
    }
 
};
 
// Thread will search the largest element in array
// from element arg->from with length of arg->length.
// Result will be stored to arg->max.
/*void *my_thread( void *void_arg )
{
    task_part *ptr_task = ( task_part * ) void_arg;
    int id = ptr_task->id;
    printf( \"Thread %d started from %d with length %d...\\n\",
        ptr_task->id, ptr_task->from[id-1], ptr_task->length[id-1] );
 
    //ptr_task->max = ptr_task->search_max();
 
    printf( \"Found maximum in thread %d is %d\\n\", ptr_task->id, ptr_task->max );
 
    return NULL;
}*/
 
void *gene( void *void_arg )
{
    task_part *ptr_task = ( task_part * ) void_arg;
 
    printf( "Thread %d started from %d with length %d.../n",
        ptr_task->id, ptr_task->from[ptr_task->id-1], ptr_task->length[ptr_task->id-1] );
 
    ptr_task->generator();
 
 
    return NULL;
}
 
void *sorting(void *void_arg)
{
    task_part *ptr_task = ( task_part * ) void_arg;
 
    printf( "Thread %d started from %d with length %d.../n",
        ptr_task->id, ptr_task->from[ptr_task->id-1], ptr_task->length[ptr_task->id-1] );
 
    ptr_task->sort();
 
 
    return NULL;
}
void *merge(void *void_arg)
{
    task_part *ptr_task = ( task_part * ) void_arg;
 
    printf( "Thread %d started from %d with length %d.../n",
        ptr_task->id, ptr_task->from[ptr_task->id-1], ptr_task->length[ptr_task->id-1] );
 
    ptr_task->merge();
 
 
    return NULL;
}
int timeval_diff_to_ms( timeval *t_before, timeval *t_after )
{
    timeval l_res;
    timersub( t_after, t_before, &l_res );
    return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}
void com_mer(int threads, int *from, int *length, int total,int smer, TYPE *data )
{
    if(THREAD_NUM == 1) return;
    printf( "Threads: %d /n",threads/2);
    int help;
        pthread_t pt[threads];
        task_part merg[threads];
        for(int i = 0; i < threads/2; i++) merg[i].setdata(((i*2+1)),from,length,data,SME_R,total);
        for(int i = 0; i < threads/2; i++){ pthread_create(&pt[i], NULL, merge, &merg[i]); }
        for(int i = 0; i < threads/2; i++){ pthread_join(pt[i], NULL); }
        for(int i = 0; i < threads/2; i++){length[i] = length[i*2]+length[(i*2)+1];}
        for(int i = 0; i < threads/2; i++){from[i] = from[i*2];}
        if(threads%2 == 1)
        {
            from[(threads/2) +1] = from[threads-1];
            length[(threads/2) +1] = length[threads-1];
            threads = (threads/2) +1;
            help = 1;
        }
        else
        {
            threads = (threads/2);
            help = 0;
        }
        if((threads) == 1 && help == 1) return;
        else com_mer(threads,from,length,total,SME_R,data);
}
 
void porc(int threads, int *from, int *length, int total )
{
    int start = 0;
    for(int i = 0; i<threads; i++)
    {
        from[i] = start;
        length[i] = (total-start)/(threads-i);
        start += (total-start)/(threads-i);
    }
}
 
 
 
// Time interval between two measurements
 
 
 
int main( int na, char **arg )
{
    // The number of elements must be used as program argument
    if ( na != 2 )
    {
        printf( "Specify number of elements, at least %d./ n ", LENGTH_LIMIT );
        return 0;
    }
    //printf( "aaa/n");
    int my_length = atoi( arg[ 1 ] );
    if ( my_length < LENGTH_LIMIT )
    {
        printf( "The number of elements must be at least %d./n", LENGTH_LIMIT );
        return 0;
    }
 
    int threads = THREAD_NUM;
    // array allocation
    TYPE *my_array = new TYPE [ my_length ];
    TYPE *from = new TYPE [threads];
    TYPE *length = new TYPE [threads];
    if ( !my_array )
    {
        printf( "Not enought memory for array! /n " );
        return 1;
    }
    timeval l_time_before, l_time_after;
    porc(THREAD_NUM, from, length, my_length);                                      //Priprava threadu
 
    /*pthread_t pt[THREAD_NUM];
    task_part gen1( 1, 0, my_length / 2, my_array );
    task_part gen2( 2, my_length / 2, my_length - my_length / 2, my_array );
 
    pthread_create( &pt1, NULL, gene, &gen1 );
    pthread_create( &pt2, NULL, gene, &gen2);
 
    pthread_join( pt1, NULL );
    pthread_join( pt2, NULL );
    */
    task_part gen[threads];
    pthread_t pt[threads];
    //task_part gen[threads];
    int i;
    // GENERACE CISEL
 
    for(int i = 0; i < threads; i++) gen[i].setdata((i+1),from,length,my_array,SME_R,my_length);
 
    for(i = 0; i < threads; i++)
    {
        pthread_create(&pt[i], NULL, gene, &gen[i]);
    }
 
 
 
    for(i = 0; i < threads; i++)
    {
        pthread_join(pt[i],NULL);
    }
 
 
 
    /*printf( \"Generated array: \\n\");
    for(int i = 0; i < my_length; i++)
    {
        printf(\"%d \\n\", my_array[i]);
    }*/
 
    //SERAZENI
    gettimeofday( &l_time_before, NULL );
 
     for(i = 0; i < threads; i++)
    {
        pthread_create(&pt[i], NULL,sorting, &gen[i]);
    }
    for(i = 0; i < threads; i++)
    {
        pthread_join(pt[i],NULL);
    }
    gettimeofday( &l_time_after, NULL );
    printf( "Thread sort time: %d [ms] /n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
    
    /*printf( "Sorted unmerged array: /n");
    for(int i = 0; i < my_length; i++)
    {
        printf("%d ", my_array[i]);
    }*/
 
    //MERGING
    gettimeofday( &l_time_before, NULL );
    com_mer(threads,from,length,my_length,SME_R,my_array);
    gettimeofday( &l_time_after, NULL );
    printf( "Sorted merged array: /n");
    for(int i = 0; i< my_length; i++)
    {
        printf("%d ", my_array[i]);
    }
    printf( "The search time: %d [ms] /n", timeval_diff_to_ms( &l_time_before, &l_time_after ) );
 
}
