#include "cublas_v2.h"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <openacc.h>

// Макрос индексации с 0
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#define T double
#define MAX std::fmax
#define STOD std::stod
#define cublascopy cublasDcopy
#define cublasaxpy cublasDaxpy
#define cublasIamax cublasIdamax

// cublas API error checking
#define CUBLAS_CHECK(err)                                                      \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("cublas error");                                \
    }                                                                          \
  } while (0)

// Вывести значения двумерного массива
void print_array(T *A, int size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
#pragma acc kernels present(A)
      printf("%.2f\t", A[IDX2C(i, j, size)]);
    }
    printf("\n");
  }
  printf("\n");
}

// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T *A, int size) {
  // Заполнение углов матрицы значениями
  A[IDX2C(0, 0, size)] = 10.0;
  A[IDX2C(0, size - 1, size)] = 20.0;
  A[IDX2C(size - 1, 0, size)] = 20.0;
  A[IDX2C(size - 1, size - 1, size)] = 30.0;

  // Заполнение периметра матрицы
  T step = 10.0 / (size - 1);

  for (int i = 1; i < size - 1; ++i) {
    T addend = step * i;
    A[IDX2C(0, i, size)] = A[IDX2C(0, 0, size)] + addend; // horizontal
    A[IDX2C(size - 1, i, size)] =
        A[IDX2C(size - 1, 0, size)] + addend;             // horizontal
    A[IDX2C(i, 0, size)] = A[IDX2C(0, 0, size)] + addend; // vertical
    A[IDX2C(i, size - 1, size)] =
        A[IDX2C(0, size - 1, size)] + addend; // vertical
  }

  //Заполнение 20-ю, чтобы сократить количество операций в основном алгоритме
  for (int i = 1; i < size - 1; ++i)
    for (int j = 1; j < size - 1; ++j)
      A[IDX2C(i, j, size)] = 20.0;
}

// Основной алгоритм
void calculate(int net_size = 128, int iter_max = 1e6, T accuracy = 1e-6,
               bool res = false) {
  // acc_set_device_num(2,acc_device_default);
  //  Размер вектора - размер сетки в квадрате
  int vec_size = net_size * net_size;
  // Флаг обновления ошибки на хосте для обработки условием цикла
  bool update_flag = true;
  // Создание 2-х матриц (векторов), одна будет считаться на основе другой. И
  // еще одна для разности
  T *Anew = new T[vec_size], *A = new T[vec_size];
  T *Adif = (T *)acc_malloc(sizeof(T) * vec_size);
  // Инициализация матриц
  initialize_array(A, net_size);
  initialize_array(Anew, net_size);
  // Текущая ошибка
  T error = 0;
  // Счетчик итераций
  int iter;
  // Указатель для swap
  T *temp;
  // Скаляр для вычитания, cublas требует указатель, поэтому выделим и под нее
  // память
  const T alpha = -1;
  // Инкремент для матриц, в этой задаче 1
  const int inc = 1;
  // Индекс максимального элемента
  int max_idx = 0;
  // Создаем указатель на структуру, содержащую контекст
  cublasHandle_t handle;
  // Инициализация контекста
  CUBLAS_CHECK(cublasCreate(&handle));
#pragma acc enter data copyin(A[:vec_size], Anew[:vec_size], error)
  if (res) {
    std::cout << "--Borders--" << std::endl;
    print_array(A, net_size);
  }
  for (iter = 0; iter < iter_max; ++iter) {
    // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять
    // значения.
    update_flag = !(iter % net_size);

// Подсчет матрицы по среднему соседей в другой матрице
#pragma acc kernels loop independent collapse(2) present(A, Anew)              \
    vector_length(128) async(1)
    for (int i = 1; i < net_size - 1; i++)
      for (int j = 1; j < net_size - 1; j++)
        Anew[IDX2C(i, j, net_size)] =
            (A[IDX2C(i + 1, j, net_size)] + A[IDX2C(i - 1, j, net_size)] +
             A[IDX2C(i, j - 1, net_size)] + A[IDX2C(i, j + 1, net_size)]) *
            0.25;
    // swap(A, Anew)
    temp = A, A = Anew, Anew = temp;
    // Проверить ошибку
    if (update_flag) {
// зануление ошибки на GPU
#pragma acc kernels present(error)
      error = 0;
#pragma acc data present(A, Anew) deviceptr(Adif)
      {
#pragma acc host_data use_device(A, Anew)
        {
          // Adif = Anew
          CUBLAS_CHECK(cublascopy(handle, vec_size, Anew, inc, Adif, inc));
          // Adif = -1 * A + Adif
          CUBLAS_CHECK(cublasaxpy(handle, vec_size, &alpha, A, inc, Adif, inc));
          // Получить индекс максимального абсолютного значения в Adif
          CUBLAS_CHECK(cublasIamax(handle, vec_size, Adif, inc, &max_idx));

#pragma acc kernels present(error)
          error = fabs(Adif[max_idx - 1]); // Fortran moment
        }
      }
// Обновление ошибки на хосте
#pragma acc update host(error) wait(1)
      // Если ошибка не превышает точность, прекратить выполнение цикла
      if (error <= accuracy)
        break;
    }
  }
// Синхронизация
#pragma acc wait(1)
  acc_free(Adif);

  std::cout.precision(2);
  if (res) {
    std::cout << "--Result--" << std::endl;
    print_array(Anew, net_size);
  }
  std::cout << "iter=" << iter << ",\terror=" << error << std::endl;

  cublasDestroy(handle);
#pragma acc exit data delete (A[:vec_size], Anew[:vec_size], error)
  delete[] A;
  delete[] Anew;
}

int main(int argc, char *argv[]) {
  auto begin_main = std::chrono::steady_clock::now();
  int net_size = 1024, iter_max = (int)1e6;
  T accuracy = 1e-6;
  bool res = false;
  for (int arg = 1; arg < argc; arg++) {
    std::string str = argv[arg];
    if (!str.compare("-res"))
      res = true;
    else {
      arg++;
      if (!str.compare("-a"))
        accuracy = STOD(argv[arg]);
      else if (!str.compare("-i"))
        iter_max = (int)std::stod(argv[arg]);
      else if (!str.compare("-s"))
        net_size = std::stoi(argv[arg]);
      else {
        std::cout << "Wrong args!";
        return -1;
      }
    }
  }
  calculate(net_size, iter_max, accuracy, res);
  auto end_main = std::chrono::steady_clock::now();
  int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_main - begin_main)
                            .count();
  std::cout << "The elapsed time is:\nmain\t\t\t" << time_spent_main << " ms\n";
  return 0;
}