
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

char **Make_grid(int size, int value_flag, char value) {        //functions that creates an array of contiguous memory
    char **temp, *arrDt;
    temp = malloc(size * sizeof(char *));
    arrDt = malloc(size * size * sizeof(char));
    for (int i = 0; i < size; ++i) {
        temp[i] = &(arrDt[i * size]);           //start of each line
    }
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < size; ++k) {
            if (value_flag == 1) {          //if flag is true we fill all the grid with value
                temp[j][k] = value;
            } else {
                temp[j][k] = rand() % 2;        //else we fill it randomly
            }
        }
    }
    return temp;
}

void Free_grid(char **temp) {
    free(temp[0]);
    free(temp);
}

int main(int argc, char *argv[]) {
    if (argc != 7) {  //mpirun -np (processes) mpi_prog -g (generations) -sz (N for N*N grid) -sc (sanity 0 or 1)
        printf("Invalid number of arguments given\n");
        return 0;
    }
    int ndims = 2;
    int dims[ndims];
    int period[ndims], coords[ndims];
    int block_side_sz[ndims];
    int arr_size, gens;
    int block_ar_len, block_proc;
    int comm_sz, rank, cart_rank;
    int zero_count, same_count;
    char **current_arr, **block;
    int sanity;
    MPI_Comm comm2;
    srand(time(NULL));

    if (strcmp(argv[1], "-g") == 0) {       //get generations
        gens = atoi(argv[2]);
    }

    if (strcmp(argv[3], "-sz") == 0) {      //get size of the grid
        arr_size = atoi(argv[4]);
    }
    if (strcmp(argv[5], "-sc") == 0) {      //value of sanity
        sanity = atoi(argv[6]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    current_arr = Make_grid(arr_size, 0, 0);

    MPI_Pcontrol(0);
    
    if (rank == 0) {
        printf("Printing initial grid\n");
        for (int i = 0; i < arr_size; ++i) {
            for (int j = 0; j < arr_size; ++j) {
                if (current_arr[i][j] == 0)
                    printf("0");
                else if (current_arr[i][j] == 1)
                    printf("1");
                else
                    printf("#");
            }
            printf("\n");
        }
    }
    //allocate array to blocks
    block_proc = (int)sqrt((float)comm_sz);     
    block_ar_len = (int)(arr_size / block_proc);       
    block = Make_grid(block_ar_len, 1, 0);

    //find where each block begins
    int num[comm_sz], begin[comm_sz], block_num;
    if (rank == 0) {
        block_num = 0;
        for (int i = 0; i < block_proc; ++i) {
            for (int j = 0; j < block_proc; ++j) {
                begin[block_proc * i + j] = block_num;
                block_num++;
            }
            block_num += (block_ar_len - 1) * block_proc;
        }
        for (int k = 0; k < comm_sz; ++k) {
            num[k] = 1;
        }
    }

    //creating datatype
    dims[0] = dims[1] = arr_size;
    block_side_sz[0] = block_side_sz[1] = block_ar_len;
    int start[2];
    start[0] = start[1] = 0;
    MPI_Datatype temp_type, block_type;
    MPI_Type_create_subarray(ndims, dims, block_side_sz, start, MPI_ORDER_C, MPI_CHAR, &temp_type);
    MPI_Type_create_resized(temp_type, 0, block_ar_len * sizeof(char), &block_type);
    MPI_Type_commit(&block_type);

    MPI_Scatterv(&(current_arr[0][0]), num, begin, block_type, &(block[0][0]), (arr_size * arr_size) / comm_sz,
                 MPI_CHAR, 0, MPI_COMM_WORLD);      //we scatter the array to each block

    int dims2[ndims];
    dims2[0] = dims2[1] = block_proc;
    period[0] = period[1] = 1;

    //make 2D Cart
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims2, period, 1, &comm2);   
    MPI_Cart_coords(comm2, rank, ndims, coords);


    //Finding neighbours
    int north, south, east, west, ne, nw, se, sw;
    int neighbour_coords[ndims];

    neighbour_coords[0] = coords[0] - 1;
    neighbour_coords[1] = coords[1];
    MPI_Cart_rank(comm2, neighbour_coords, &north);

    neighbour_coords[0] = coords[0] + 1;
    neighbour_coords[1] = coords[1];
    MPI_Cart_rank(comm2, neighbour_coords, &south);

    neighbour_coords[0] = coords[0];
    neighbour_coords[1] = coords[1] + 1;
    MPI_Cart_rank(comm2, neighbour_coords, &east);

    neighbour_coords[0] = coords[0];
    neighbour_coords[1] = coords[1] - 1;
    MPI_Cart_rank(comm2, neighbour_coords, &west);

    neighbour_coords[0] = coords[0] - 1;
    neighbour_coords[1] = coords[1] + 1;
    MPI_Cart_rank(comm2, neighbour_coords, &ne);

    neighbour_coords[0] = coords[0] - 1;
    neighbour_coords[1] = coords[1] - 1;
    MPI_Cart_rank(comm2, neighbour_coords, &nw);

    neighbour_coords[0] = coords[0] + 1;
    neighbour_coords[1] = coords[1] + 1;
    MPI_Cart_rank(comm2, neighbour_coords, &se);

    neighbour_coords[0] = coords[0] + 1;
    neighbour_coords[1] = coords[1] - 1;
    MPI_Cart_rank(comm2, neighbour_coords, &sw);


    char *send_line = malloc(block_ar_len * sizeof(char));
    char *send_side = malloc(block_ar_len * sizeof(char));
    char **recv_array = malloc(8 * sizeof(char *));             //array in which each process receives the halo points and the corners
    for (int l = 0; l < 8; ++l) {
        recv_array[l] = malloc(block_ar_len * sizeof(char));
    }

    char **next = Make_grid(block_ar_len, 1, 0);

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();       //start time
    MPI_Pcontrol(1);
    
    //evolve and communication
    //before:block, after: next
    //we receive into the recv_array
    //we send lines from send_line, for corners we send only one element
    //we send columns from send_side, for corners we send only one element
    MPI_Request req[8];
    MPI_Request reqs[8];
    for (int m = 0; m < gens; ++m) {
        MPI_Irecv(recv_array[0], block_ar_len, MPI_CHAR, north, 0, comm2, &req[0]);
        MPI_Irecv(recv_array[1], block_ar_len, MPI_CHAR, south, 0, comm2, &req[1]);
        MPI_Irecv(recv_array[2], block_ar_len, MPI_CHAR, east, 0, comm2, &req[2]);
        MPI_Irecv(recv_array[3], block_ar_len, MPI_CHAR, west, 0, comm2, &req[3]);
        MPI_Irecv(recv_array[4], 1, MPI_CHAR, ne, 0, comm2, &req[4]);
        MPI_Irecv(recv_array[5], 1, MPI_CHAR, nw, 0, comm2, &req[5]);
        MPI_Irecv(recv_array[6], 1, MPI_CHAR, se, 0, comm2, &req[6]);
        MPI_Irecv(recv_array[7], 1, MPI_CHAR, sw, 0, comm2, &req[7]);

        send_line = &(block[block_ar_len - 1][0]);
        MPI_Isend(send_line, block_ar_len, MPI_CHAR, south, 0, comm2, &reqs[0]);
        send_line = &(block[0][0]);
        MPI_Isend(send_line, block_ar_len, MPI_CHAR, north, 0, comm2, &reqs[1]);
        for (int i = 0; i < block_ar_len; ++i) {
            send_side[i] = block[i][0];
        }
        MPI_Isend(send_side, block_ar_len, MPI_CHAR, west, 0, comm2, &reqs[2]);
        for (int i = 0; i < block_ar_len; ++i) {
            send_side[i] = block[i][block_ar_len - 1];
        }
        MPI_Isend(send_side, block_ar_len, MPI_CHAR, east, 0, comm2, &reqs[3]);
        send_side = &(block[block_ar_len - 1][0]);
        MPI_Isend(send_side, 1, MPI_CHAR, sw, 0, comm2, &reqs[4]);
        send_side = &(block[block_ar_len - 1][block_ar_len - 1]);
        MPI_Isend(send_side, 1, MPI_CHAR, se, 0, comm2, &reqs[5]);
        send_side = &(block[0][0]);
        MPI_Isend(send_side, 1, MPI_CHAR, nw, 0, comm2, &reqs[6]);
        send_side = &(block[0][block_ar_len - 1]);
        MPI_Isend(send_side, 1, MPI_CHAR, ne, 0, comm2, &reqs[7]);

        //Calculate inner 

        int zeros = 0;
        int same = 0;

        for (int k = 1; k < block_ar_len - 1; ++k) {
            int n = k - 1;
            int s = k + 1;
            for (int l = 1; l < block_ar_len - 1; ++l) {
                int neighbours = 0;
                int w = l - 1;
                int e = l + 1;

                if (block[n][l] == 1)
                    neighbours++;
                if (block[s][l] == 1)
                    neighbours++;
                if (block[k][w] == 1)
                    neighbours++;
                if (block[k][e] == 1)
                    neighbours++;
                if (block[n][w] == 1)
                    neighbours++;
                if (block[n][e] == 1)
                    neighbours++;
                if (block[s][w] == 1)
                    neighbours++;
                if (block[s][e] == 1)
                    neighbours++;


                if (block[k][l] == 1 && (neighbours == 0 || neighbours == 1))
                    next[k][l] = 0;
                if (block[k][l] == 1 && (neighbours == 2 || neighbours == 3))
                    next[k][l] = 1;
                if (block[k][l] == 1 && neighbours >= 4)
                    next[k][l] = 0;
                if (block[k][l] == 0 && neighbours == 3)
                    next[k][l] = 1;
                if (block[k][l] == 0 && neighbours != 3)
                    next[k][l] = 0;
                if (next[k][l] == 1)
                    zeros = 1;
                if (next[k][l] != block[k][l])
                    same = 1;
            }
        }

        //Wait
        MPI_Waitall(8, req, MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
    

        //Calculate outer
        //in recv_array: north=0, south=1, east=2, west=3, ne=4, nw=5, se=6, sw=7
        //upper line

        for (int j = 0; j < block_ar_len; ++j) {
            int neighbours = 0;
            if (j == 0) {  //left corner
                if (block[0][j + 1] == 1)
                    neighbours++;
                if (block[1][j + 1] == 1)
                    neighbours++;
                if (block[1][j] == 1)
                    neighbours++;
                if (recv_array[3][1] == 1)
                    neighbours++;
                if (recv_array[3][0] == 1)
                    neighbours++;
                if (recv_array[5][0] == 1)
                    neighbours++;
                if (recv_array[0][0] == 1)
                    neighbours++;
                if (recv_array[0][1] == 1)
                    neighbours++;
            } else if (j == block_ar_len - 1) {  //right corner
                if (block[0][j - 1] == 1)
                    neighbours++;
                if (block[1][j - 1] == 1)
                    neighbours++;
                if (block[1][j] == 1)
                    neighbours++;
                if (recv_array[2][1] == 1)
                    neighbours++;
                if (recv_array[2][0] == 1)
                    neighbours++;
                if (recv_array[4][0] == 1)
                    neighbours++;
                if (recv_array[0][j] == 1)
                    neighbours++;
                if (recv_array[0][j - 1] == 1)
                    neighbours++;
            } else {
                if (block[0][j + 1] == 1)
                    neighbours++;
                if (block[1][j + 1] == 1)
                    neighbours++;
                if (block[1][j] == 1)
                    neighbours++;
                if (block[1][j - 1] == 1)
                    neighbours++;
                if (block[0][j - 1] == 1)
                    neighbours++;
                if (recv_array[0][j - 1] == 1)
                    neighbours++;
                if (recv_array[0][j] == 1)
                    neighbours++;
                if (recv_array[0][j + 1] == 1)
                    neighbours++;
            }


            if (block[0][j] == 1 && (neighbours == 0 || neighbours == 1))
                next[0][j] = 0;
            if (block[0][j] == 1 && (neighbours == 2 || neighbours == 3))
                next[0][j] = 1;
            if (block[0][j] == 1 && neighbours >= 4)
                next[0][j] = 0;
            if (block[0][j] == 0 && neighbours == 3)
                next[0][j] = 1;
            if (block[0][j] == 0 && neighbours != 3)
                next[0][j] = 0;
            if (next[0][j] == 1)
                zeros = 1;
            if (next[0][j] != block[0][j])
                same = 1;
        }

        //in recv_array: north=0, south=1, east=2, west=3, ne=4, nw=5, se=6, sw=7
        //bottom line
        for (int j = 0; j < block_ar_len; ++j) {
            int neighbours = 0;
            if (j == 0) {  //left corner
                if (block[block_ar_len - 1][j + 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j + 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j] == 1)
                    neighbours++;
                if (recv_array[3][block_ar_len - 2] == 1)
                    neighbours++;
                if (recv_array[3][block_ar_len - 1] == 1)
                    neighbours++;
                if (recv_array[7][0] == 1)
                    neighbours++;
                if (recv_array[1][0] == 1)
                    neighbours++;
                if (recv_array[1][1] == 1)
                    neighbours++;
            } else if (j == block_ar_len - 1) {  //right corner
                if (block[block_ar_len - 1][j - 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j - 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j] == 1)
                    neighbours++;
                if (recv_array[2][block_ar_len - 2] == 1)
                    neighbours++;
                if (recv_array[2][block_ar_len - 1] == 1)
                    neighbours++;
                if (recv_array[6][0] == 1)
                    neighbours++;
                if (recv_array[1][j] == 1)
                    neighbours++;
                if (recv_array[1][j - 1] == 1)
                    neighbours++;
            } else {
                if (block[block_ar_len - 1][j + 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j + 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j] == 1)
                    neighbours++;
                if (block[block_ar_len - 2][j - 1] == 1)
                    neighbours++;
                if (block[block_ar_len - 1][j - 1] == 1)
                    neighbours++;
                if (recv_array[1][j - 1] == 1)
                    neighbours++;
                if (recv_array[1][j] == 1)
                    neighbours++;
                if (recv_array[1][j + 1] == 1)
                    neighbours++;
            }


            if (block[block_ar_len - 1][j] == 1 && (neighbours == 0 || neighbours == 1))
                next[block_ar_len - 1][j] = 0;
            if (block[block_ar_len - 1][j] == 1 && (neighbours == 2 || neighbours == 3))
                next[block_ar_len - 1][j] = 1;
            if (block[block_ar_len - 1][j] == 1 && neighbours >= 4)
                next[block_ar_len - 1][j] = 0;
            if (block[block_ar_len - 1][j] == 0 && neighbours == 3)
                next[block_ar_len - 1][j] = 1;
            if (block[block_ar_len - 1][j] == 0 && neighbours != 3)
                next[block_ar_len - 1][j] = 0;
            if (next[block_ar_len - 1][j] == 1)
                zeros = 1;
            if (next[block_ar_len - 1][j] != block[block_ar_len - 1][j])
                same = 1;
        }

        //in recv_array: north=0, south=1, east=2, west=3, ne=4, nw=5, se=6, sw=7
        //left column, corners are already finished
        for (int i = 1; i < block_ar_len - 1; ++i) {
            int neighbours = 0;

            if (block[i][1] == 1)
                neighbours++;
            if (block[i + 1][1] == 1)
                neighbours++;
            if (block[i][0] == 1)
                neighbours++;
            if (block[i - 1][1] == 1)
                neighbours++;
            if (block[i - 1][0] == 1)
                neighbours++;
            if (recv_array[3][i - 1] == 1)
                neighbours++;
            if (recv_array[3][i] == 1)
                neighbours++;
            if (recv_array[3][i + 1] == 1)
                neighbours++;


            if (block[i][0] == 1 && (neighbours == 0 || neighbours == 1))
                next[i][0] = 0;
            if (block[i][0] == 1 && (neighbours == 2 || neighbours == 3))
                next[i][0] = 1;
            if (block[i][0] == 1 && neighbours >= 4)
                next[i][0] = 0;
            if (block[i][0] == 0 && neighbours == 3)
                next[i][0] = 1;
            if (block[i][0] == 0 && neighbours != 3)
                next[i][0] = 0;
            if (next[i][0] == 1)
                zeros = 1;
            if (next[i][0] != block[i][0])
                same = 1;
        }

        //in recv_array: north=0, south=1, east=2, west=3, ne=4, nw=5, se=6, sw=7
        //right column, corners are already finished
        for (int i = 1; i < block_ar_len - 1; ++i) {
            int neighbours = 0;

            if (block[i][block_ar_len - 2] == 1)
                neighbours++;
            if (block[i + 1][block_ar_len - 2] == 1)
                neighbours++;
            if (block[i + 1][block_ar_len - 1] == 1)
                neighbours++;
            if (block[i - 1][block_ar_len - 2] == 1)
                neighbours++;
            if (block[i - 1][block_ar_len - 1] == 1)
                neighbours++;
            if (recv_array[2][i - 1] == 1)
                neighbours++;
            if (recv_array[2][i] == 1)
                neighbours++;
            if (recv_array[2][i + 1] == 1)
                neighbours++;


            if (block[i][block_ar_len - 1] == 1 && (neighbours == 0 || neighbours == 1))
                next[i][block_ar_len - 1] = 0;
            if (block[i][block_ar_len - 1] == 1 && (neighbours == 2 || neighbours == 3))
                next[i][block_ar_len - 1] = 1;
            if (block[i][block_ar_len - 1] == 1 && neighbours >= 4)
                next[i][block_ar_len - 1] = 0;
            if (block[i][block_ar_len - 1] == 0 && neighbours == 3)
                next[i][block_ar_len - 1] = 1;
            if (block[i][block_ar_len - 1] == 0 && neighbours != 3)
                next[i][block_ar_len - 1] = 0;
            if (next[i][block_ar_len - 1] == 1)
                zeros = 1;
            if (next[i][block_ar_len - 1] != block[i][block_ar_len - 1])
                same = 1;
        }

        if ((sanity == 1) && (m % 10 == 0) && (m != 0)) {       //check if grid is all zeros or is the same as the next generations
            // printf("hi sanity %d\n", sanity);
            MPI_Reduce(&zeros, &zero_count, 1, MPI_INT, MPI_MAX, 0, comm2);
            MPI_Reduce(&same, &same_count, 1, MPI_INT, MPI_MAX, 0, comm2);
            if (rank == 0) {
                if (zero_count == 0 || same_count == 0) {
                    printf("Sanity check failed!\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }

        //swapping
        char **temp;
        temp = block;
        block = next;
        next = temp;
        
        MPI_Waitall(8, reqs, MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);


    }
    MPI_Gatherv(&(block[0][0]), arr_size * arr_size / comm_sz, MPI_CHAR, &(current_arr[0][0]), num, begin, block_type, 0, MPI_COMM_WORLD);      //all gather array from each process into current_arr
    MPI_Pcontrol(0);
    end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    double max, min, average;

    MPI_Reduce(&total_time, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);       //calculate max time
    MPI_Reduce(&total_time, &min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);       //calculate min time
    MPI_Reduce(&total_time, &average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   //calculate average time
    average = average / comm_sz;

    //print final grid
    if (rank == 0) {
        printf("Printing final grid. Generation: %d\n", gens);
        for (int i = 0; i < arr_size; ++i) {
            for (int j = 0; j < arr_size; ++j) {
                if (current_arr[i][j] == 0)
                    printf("0");
                else if (current_arr[i][j] == 1)
                    printf("1");
                else
                    printf("#");
            }
            printf("\n");
        }
    }

    if (rank == 0) {
        printf("Max time is: %f and min time is %f\n", max, min);
        printf("Average time is: %f\n", average);
    }

    
    fflush(stdout);
    MPI_Type_free(&block_type);
    MPI_Comm_free(&comm2);

    MPI_Finalize();
    return 0;
}