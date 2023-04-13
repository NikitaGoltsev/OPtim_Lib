format:
	clang-format -i *.cpp
avg_save:
	git add *
	git commit -m "avg_save"
	git push