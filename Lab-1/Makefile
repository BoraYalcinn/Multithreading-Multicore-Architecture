# ===========================================================================
# Lab-1 — build file
#
#   make            build every C++ example into bin/
#   make java       compile every Java example into bin/java/
#   make ex1        build a single target (see TARGETS below)
#   make run-ex1    build and run a single target
#   make clean      remove bin/ and the legacy in-place binaries
#
# Every binary lands in bin/, which is git-ignored.
# ===========================================================================

CXX       ?= g++
CXXFLAGS  ?= -std=c++20 -O2 -Wall -Wextra -pthread
JAVAC     ?= javac
BIN       := bin

# Examples that need libcurl (Example10, I/O-bound HTTP downloads)
CURL_LIBS := -lcurl

# --- target name  ->  source file ------------------------------------------
SRC_ex1             := Example1/ex1.cpp
SRC_ex2             := Example2/MultithreadingExample.cpp
SRC_ex2_workers     := Example2-Part-BandC/MultithreadingExampleWorkers.cpp
SRC_deadlock        := Example3/Deadlock.cpp
SRC_deadlock_fixed  := Example3/DeadlockFixed.cpp
SRC_race            := Example4/RaceCondition.cpp
SRC_producer        := Example5/ProducerConsumer.cpp
SRC_factorial       := Example6/Factorial.cpp
SRC_matmul_mt       := Example7/matrix_mult.cpp
SRC_matmul_mp       := Example7/matmul_mp.cpp
SRC_barrier         := Example8/Barrier.cpp
SRC_mapreduce       := Example9/map_reduce.cpp
SRC_io_seq          := Example10/sequential_IO.cpp
SRC_io_threads      := Example10/thread_pool_IO.cpp
SRC_io_procs        := Example10/process_pool_IO.cpp

TARGETS := ex1 ex2 ex2_workers deadlock deadlock_fixed race producer \
           factorial matmul_mt matmul_mp barrier mapreduce \
           io_seq io_threads io_procs

# Targets that additionally link against libcurl
CURL_TARGETS := io_seq io_threads io_procs

JAVA_SRCS := $(shell find . -name '*.java')

.PHONY: all java clean help $(TARGETS) $(addprefix run-,$(TARGETS))

all: $(TARGETS)

# Pattern: `make ex1` builds bin/ex1
$(TARGETS): %: $(BIN)/%

$(BIN):
	@mkdir -p $(BIN)

# One rule per target, generated so each knows its own source and libs.
define BUILD_RULE
$(BIN)/$(1): $$(SRC_$(1)) | $$(BIN)
	$$(CXX) $$(CXXFLAGS) $$< -o $$@ $$(if $$(filter $(1),$$(CURL_TARGETS)),$$(CURL_LIBS),)
	@echo "  built -> $$@"
endef
$(foreach t,$(TARGETS),$(eval $(call BUILD_RULE,$(t))))

# `make run-ex1` builds then runs
$(addprefix run-,$(TARGETS)): run-%: $(BIN)/%
	@echo "--- running $< ---"
	@./$<

java: | $(BIN)
	@mkdir -p $(BIN)/java
	$(JAVAC) -d $(BIN)/java $(JAVA_SRCS)
	@echo "  compiled Java classes -> $(BIN)/java"
	@echo "  run with: java -cp $(BIN)/java <ClassName>"

clean:
	@rm -rf $(BIN)
	@find . -name '*.class' -delete
	@rm -f Example1/ex1 Example2/MultithreadingExample \
	       Example2-Part-BandC/MultithreadingExampleWorkers \
	       Example3/Deadlock Example3/DeadlockFixed \
	       Example4/RaceCondition Example5/ProducerConsumer \
	       Example6/Factorial Example7/matrix_mult Example7/matmul_mp \
	       Example8/Barrier Example9/map_reduce \
	       Example10/sequential_IO Example10/thread_pool_IO Example10/process_pool_IO
	@echo "cleaned."

help:
	@echo "Targets: $(TARGETS)"
	@echo ""
	@echo "  make            build all C++ examples into $(BIN)/"
	@echo "  make java       compile all Java examples into $(BIN)/java/"
	@echo "  make <name>     build one example"
	@echo "  make run-<name> build and run one example"
	@echo "  make clean      remove build output"
