all:
	g++ -fsingle-precision-constant -lstdc++ -DIKFAST_REAL=float ik.cpp
lib:
	g++ -fsingle-precision-constant -fPIC -lstdc++ -DIKFAST_NO_MAIN -DIKFAST_CLIBRARY -DIKFAST_REAL=float -shared -Wl,-soname,libik.so -o libik.so ik.cpp
clean:
	rm a.out libik.so
