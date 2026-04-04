#set(CMAKE_CXX_FLAGS "-I/home/max/repos/distributed-data-queue/_build/install/include -L/home/max/repos/distributed-data-queue/_build/install/lib -lrdq")

set(RDQ_TEST /home/max/repos/remote-data-queue/_build/install)

#target_compile_options(udf PUBLIC "-I${RDQ_TEST}/include")
#target_link_libraries(udf PUBLIC "${RDQ_TEST}/lib/librdq.a")
target_compile_options(udf PUBLIC "-I${NEKRS_INSTALL_DIR}/3rd_party/rdq/include")
target_link_libraries(udf PUBLIC "${NEKRS_INSTALL_DIR}/3rd_party/rdq/lib/librdq.a")

#target_compile_definitions(udf PUBLIC SMARTREDIS=1)

