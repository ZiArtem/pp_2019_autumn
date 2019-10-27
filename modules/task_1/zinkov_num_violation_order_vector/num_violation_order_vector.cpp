// Copyright 2019 Zinkov Artem
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include "../../../modules/task_1/zinkov_num_violation_order_vector/num_violation_order_vector.h"

std::vector<int> getRandomVector(int length) {
  if (length < 1)
    throw "WRONG_LEN";

  std::vector<int> vec(length);
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));

  for (auto& val : vec) {
    val= gen() % 100;
  }

  return vec;
}

int getNumViolationOrderVector(std::vector<int> vec) {
  int num = 0;

  for (size_t i = 1; i < vec.size(); i++) {
    if (vec[i - 1] > vec[i]) {
      num++;
    }
  }
  return num;
}

int getNumViolationOrderVectorParallel(std::vector<int> global_vec, int size_vector) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int delta = size_vector / size;
  const int residue = size_vector % size;
  int global_num = 0;
  std::vector<int> local_vec;

  if (size_vector < size) {
    if (rank == 0) {
      global_num = getNumViolationOrderVector(global_vec);
    }
    return global_num;
  }

  if (rank==0) {
    local_vec.resize (delta + residue);
  } else {
    local_vec.resize (delta + 1);
  }
	
  int* sendcounts = new int[size];
  int* displs = new int[size];

  for (int i = 0; i < size; i++) {
    displs[i] = 0;
    if (i == 0) {
      sendcounts[i] = delta + residue;
    }	else {
      sendcounts[i] = delta + 1;
    }
    if (i > 0)	{
      displs[i] = displs[i - 1] + sendcounts[i - 1]-1;
    }
  }

  MPI_Scatterv(global_vec.data(), sendcounts, displs, MPI_INT,
    &local_vec.front(), sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  int local_num = getNumViolationOrderVector(local_vec);
  MPI_Reduce(&local_num, &global_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  return global_num;
}
