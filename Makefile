# SDL-GL-basic template
#
# See SDL-GL-basic.c for details
#
# written by Hans de Ruiter
#
# License:
# This source code can be used and/or modified without restrictions.
# It is provided as is and the author disclaims all warranties, expressed 
# or implied, including, without limitation, the warranties of
# merchantability and of fitness for any purpose. The user must assume the
# entire risk of using the Software.

CC     = g++
CP     = cp
RM     = rm -rf
KILL   = killall -9
SHELL  = /bin/sh
MAKE   = make

IFLAGS = -I./lib -I./lib/pngwriter/include -DNO_FREETYPE
LFLAGS = -lpng -lz -lpngwriter -L./lib/pngwriter/lib
OPTIMIZE = -O3
ERROR = -Wconversion -Werror
CFLAGS = $(OPTIMIZE) -Wall -ggdb $(ERROR) $(LIBFLAGS)
LDFLAGS = $(OPTIMIZE) $(ERROR) $(LIBFLAGS)

TARGET = SPEEDRACER™
MODEL_DIR = models
MODEL = dragon1K
MODEL_EXT = m
ARGS = $(MODEL_DIR)/$(MODEL).$(MODEL_EXT)

# Additional linker libraries
LIBS = $(LIBFLAGS) -lm

# -------------------------------------------------------------
# Nothing should need changing below this line

# The source files
SRCS = $(wildcard src/*.cpp) $(wildcard src/**/*.cpp)
HEADER = $(wildcard src/*.h) $(wildcard src/**/*.h)

OBJS = $(SRCS:.cpp=.o)

# Rules for building
all: $(TARGET)

$(TARGET): $(OBJS) $(HEADERS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: lib
lib:
	$(SHELL) ./lib.sh

run:
	./$(TARGET) $(ARGS)

gdb:
	gdb ./$(TARGET) --args $(ARGS)

valgrind:
	valgrind --tool=memcheck --leak-check=full ./$(TARGET) $(ARGS)

clean:
	$(RM) $(TARGET) $(OBJS)

killall:
	$(KILL) $(TARGET)
