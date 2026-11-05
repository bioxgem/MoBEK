CC	= g++ -Wall
MAIN = Composition.cpp
OUTPUT = ac
LIB	= ../bin/mod_ac
FLAG = -static -O3
IMPLEMENT = Composition.cpp Composition.h Compound.h Edge.h String.h

all:
	${CC} ${FLAG} ${MAIN} -o ${LIB}
clean:
	rm -f *.o ${OUTPUT} $(LIB)
cleanw:
	del /Q *.o ${OUTPUT}.exe $(LIB).exe

