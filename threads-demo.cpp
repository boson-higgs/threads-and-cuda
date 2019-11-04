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

class task_part
{
public:
    int id;                 // user identification
    int from, length;       // data range
    TYPE *data;             // array
    TYPE max;               // result 

    task_part( int myid, int first, int num, TYPE *ptr ) :
        id( myid ), from( first ), length( num ), data( ptr ) {}

    TYPE get_result() { return max; }

    // function search_max search the largest number in part of array
    // from the left (included) up to the right element
    TYPE search_max() 
    {
        TYPE max_elem = data[ from ];
        for ( int i = 1; i < length; i++ )
            if ( max_elem < data[ from + i ] )
                max_elem = data[ from + i ];
        return max_elem;
    }
};

// Thread will search the largest element in array 
// from element arg->from with length of arg->length.
// Result will be stored to arg->max.
void *my_thread( void *void_arg )
{
    task_part *ptr_task = ( task_part * ) void_arg;

    printf( "Thread %d started from %d with length %d...\n",
        ptr_task->id, ptr_task->from, ptr_task->length );

    ptr_task->max = ptr_task->search_max();

    printf( "Found maximum in thread %d is %d\n", ptr_task->id, ptr_task->max );

    return NULL;
}

// Time interval between two measurements
int timeval_to_ms( timeval *before, timeval *after )
{
    timeval res;
    timersub( after, before, &res );
    return 1000 * res.tv_sec + res.tv_usec / 1000;
}

#define LENGTH_LIMIT 10000000

int main( int na, char **arg )
{
    // The number of elements must be used as program argument
    if ( na != 2 ) 
    { 
        printf( "Specify number of elements, at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
    }
    int my_length = atoi( arg[ 1 ] );
    if ( my_length < LENGTH_LIMIT ) 
    { 
        printf( "The number of elements must be at least %d.\n", LENGTH_LIMIT ); 
        return 0; 
    }

    // array allocation
    TYPE *my_array = new TYPE [ my_length ];
    if ( !my_array ) 
    {
        printf( "Not enought memory for array!\n" );
        return 1;
    }

    // Initialization of random number generator
    srand( ( int ) time( NULL ) );

    printf( "Random numbers generetion started..." );
    for ( int i = 0; i < my_length; i++ )
    {
            my_array[ i ] = rand() % ( my_length * 10 );
            if ( !( i % LENGTH_LIMIT ) ) 
            {
                printf( "." ); 
                fflush( stdout );
            }
    }

    printf( "\nMaximum number search using two threads...\n" );
    pthread_t pt1, pt2;
    task_part tp1( 1, 0, my_length / 2, my_array );
    task_part tp2( 2, my_length / 2, my_length - my_length / 2, my_array );
    timeval time_before, time_after;

    // Time recording before searching
    gettimeofday( &time_before, NULL );


    // Threads starting
    pthread_create( &pt1, NULL, my_thread, &tp1 );
    pthread_create( &pt2, NULL, my_thread, &tp2 );

    // Waiting for threads completion 
    pthread_join( pt1, NULL );
    pthread_join( pt2, NULL );

    // Time recording after searching
    gettimeofday( &time_after, NULL );

    printf( "The found maximum: %d\n", MAX( tp1.get_result(), tp2.get_result() ) );
    printf( "The search time: %d [ms]\n", timeval_to_ms( &time_before, &time_after ) );

    printf( "\nMaximum number search using one thread...\n" );

    gettimeofday( &time_before, NULL );

    // Searching in single thread
    task_part single( 333, 0, my_length, my_array );
    TYPE res = single.search_max();

    gettimeofday( &time_after, NULL );

    printf( "The found maximum: %d\n", res );
    printf( "The search time: %d [ms]\n", timeval_to_ms( &time_before, &time_after ) );
}
