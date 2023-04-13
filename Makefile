format:
	clang-format -i *.cpp

avg_save:
	git add *
	git commit -m "avg_save"
	git push

run:
	pgc++ main.cpp -fast -acc=gpu -O2 -o main_gpu -Mcudalib=cublas
	./main_gpu
res:
	nsys profile -t openacc,cublas,nvtx ./main_gpu -s 512 -i 100