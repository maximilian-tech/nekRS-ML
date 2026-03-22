#set(CMAKE_CXX_FLAGS "-I/home/max/repos/distributed-data-queue/_build/install/include -L/home/max/repos/distributed-data-queue/_build/install/lib -lrdq")

target_compile_options(udf PUBLIC "-I/home/max/repos/distributed-data-queue/_build_gcc15/install/include")
target_link_libraries(udf PUBLIC "/home/max/repos/distributed-data-queue/_build_gcc15/install/lib/librdq.a")

#target_compile_definitions(udf PUBLIC SMARTREDIS=1)

