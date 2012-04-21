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

IFLAGS = -I./src -I./lib -I./lib/pngwriter/include -DNO_FREETYPE -L./lib/pngwriter/lib
LFLAGS = -lpng -lz -lpngwriter -L./lib/pngwriter/lib
OPTIMIZE = -O3
ERROR = -Wconversion -Werror
CFLAGS = $(OPTIMIZE) -Wall -ggdb $(ERROR) $(IFLAGS)
LDFLAGS = $(OPTIMIZE) $(ERROR) $(LFLAGS)

TARGET = SPEEDRACER™
MODEL_DIR = models
MODEL = bunny500
MODEL_EXT = m
IMG_DIR = images
IMG_EXT = png
WIDTH = 800
HEIGHT = 600
ARGS = -i $(MODEL_DIR)/$(MODEL).$(MODEL_EXT) -o $(IMG_DIR)/$(MODEL).$(IMG_EXT) -w $(WIDTH) -h $(HEIGHT)

# Additional linker libraries
LIBS = $(LIBFLAGS) -lm

# -------------------------------------------------------------
# Nothing should need changing below this line

# The source files
SRCS = $(wildcard src/*.cpp) $(wildcard src/**/*.cpp)
HEADERS = $(wildcard src/*.h) $(wildcard src/**/*.h)

OBJS = $(SRCS:.cpp=.o)

# Rules for building
all: $(TARGET)

$(TARGET): $(OBJS) $(HEADERS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: lib
lib:
	$(SHELL) ./lib.sh

run:
	./$(TARGET) $(ARGS)

eog:
	eog ./$(IMG_DIR)/$(MODEL).$(IMG_EXT)

gdb:
	gdb ./$(TARGET) --args $(ARGS)

valgrind:
	valgrind --tool=memcheck --leak-check=full ./$(TARGET) $(ARGS)

clean:
	$(RM) $(TARGET) $(OBJS)

killall:
	$(KILL) $(TARGET)
